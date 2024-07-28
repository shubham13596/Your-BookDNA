import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import openai
import os
import requests

def prepare_model(csv_file_path):
    # Function to get book description from Google Books API
    def get_book_description(title, author):
        base_url = "https://www.googleapis.com/books/v1/volumes"
        query = f"intitle:{title}+inauthor:{author}"
        params = {"q": query, "maxResults": 1}
        
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if "items" in data and len(data["items"]) > 0:
                volume_info = data["items"][0]["volumeInfo"]
                return volume_info.get("description", "No description available")
        return "No description available"

    # Load data from the uploaded CSV
    df = pd.read_csv(csv_file_path)

    total_rows = len(df)
    print(f"Total rows in CSV: {total_rows}")

    if total_rows < 200:
        print("Less than 200 rows. Including all rows for model building.")
    else:
        print("200 or more rows. Excluding rows with 'My Rating' = 0 for model building.")
        df = df[df['My Rating'] != 0]


    # Add description column
    df['description'] = df.apply(lambda row: get_book_description(row['Title'], row['Author']), axis=1)

    # Set up OpenAI client
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Function to get OpenAI embedding
    def get_openai_embedding(text):
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    # Get embeddings for descriptions
    desc_embeddings = np.array([get_openai_embedding(desc) for desc in df['description']])
    desc_embeddings = torch.FloatTensor(desc_embeddings)

    # Prepare data for PyTorch
    WANT_TO_READ_RATING = 3.5  # Pseudo-rating for want-to-read books
    WANT_TO_READ_WEIGHT = 0.5  # Weight for want-to-read pseudo-ratings
    ratings = df['My Rating'].copy()
    weights = pd.Series(1.0, index=df.index)  # Default weight 1 for actual ratings

    # Fill in pseudo-ratings for want-to-read books (My rating = 0)
    if total_rows < 200:
        want_to_read_mask = df['My Rating'] == 0
        ratings[want_to_read_mask] = WANT_TO_READ_RATING
        weights[want_to_read_mask] = WANT_TO_READ_WEIGHT

    ratings = torch.FloatTensor(ratings.values)
    weights = torch.FloatTensor(weights.values)

    # Split data
    train_ratings, test_ratings, train_desc, test_desc, train_weights, test_weights = train_test_split(
        ratings, desc_embeddings, weights, test_size=0.2, random_state=42
    )

    # Define the model
    class BookRecommender(nn.Module):
        def __init__(self, n_factors, n_desc_features):
            super().__init__()
            self.desc_transform = nn.Linear(n_desc_features, n_factors)
            self.output = nn.Linear(n_factors, 1)
            
        def forward(self, description):
            desc_embedding = self.desc_transform(description)
            combined = desc_embedding
            return self.output(combined).squeeze()

    # Set up the model
    n_factors = 50
    n_desc_features = 1536  # OpenAI's text-embedding-ada-002 has 1536 dimensions
    model = BookRecommender(n_factors, n_desc_features)
    criterion = nn.MSELoss(reduction='none')  # Use 'none' to apply weights manually
    optimizer = optim.Adam(model.parameters())

    # Train the model
    n_epochs = 100
    batch_size = 64
    for epoch in range(n_epochs):
        for i in range(0, len(train_desc), batch_size):
            desc_batch = train_desc[i:i+batch_size]
            rating_batch = train_ratings[i:i+batch_size]
            weight_batch = train_weights[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(desc_batch)
            losses = criterion(outputs, rating_batch)
            weighted_loss = (losses * weight_batch).mean()
            weighted_loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {weighted_loss.item():.4f}')

    return model

# Function to get embedding for a single description
def get_embedding(description):
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=description,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding

# Function to predict rating for a single book
def predict_rating(model, description):
    embedding = get_embedding(description)
    embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
    with torch.no_grad():
        prediction = model(embedding_tensor)
    return prediction.item()