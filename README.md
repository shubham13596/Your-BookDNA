# Personalized Book Recommender

## Overview

Choosing my next book to read is an arduous task for me. I want to make sure it's well worth the time and effort. I go through a book's reviews, discussions, ratings etc. to make sure that the book I choose is worthy. And while going through 'NYT's Top 100 Books to Read' I found this process to be a nightmare. Doing this process for 100 books wasn't possible.

So I thought what if I could get some help from a digital twin who understands my reading preferences and could lend me a hand? That's what made me build 'Your BookDNA'. 

Your BookDNA understands your past reading preferences to predict book ratings for any book, as you would rate it. There is also a feature to extract book titles from any webpage and show predicted ratings. Online discussion forums mention multiple books which you would want to check out. The book title extractor makes that easy. 

The limitation with the current product is that I can show predictions only if you have a Goodreads reading history (since my model trains on it in real time). The backend is a neural network that trains on your book ratings and descriptions (converted to GPT-4 embeddings) fetched via Google Books API.

It's now available in production. You can try it on your Goodreads data! 
Website: https://bookrecommendations1-hcr77cz3ga-el.a.run.app

Check out this video demo of the Book Recommender in action:(https://youtu.be/35y6kJRjng4?si=z_RG16PuCc4CY-NK)

Happy to hear feedback, comments, and critiques.

The above was made possible only because I record all my reading in Goodreads and it allows for an export option. What if we could aggregate our digital trail across platforms to be able to build recommenders for ourselves beyond the lock-ins of a large platform? 

For ex. if you could export your watching history of Amazon Prime & Netflix, combine them, and ask a 3rd party to build your movie preference ML model. That would help you with better movie recommendations instead of disparate platforms having a piecemeal idea of what you want to watch basis your watch history only on their platform. Movies are just one example.

## Features

- **CSV Upload**: Upload your Goodreads reading history to create a personalized model.
- **Famous Book Lists**: Currently includes Obama's 2023 Reading List and Pulitzer Prize Masterpieces.
- **Personalized Ratings**: Predicts your potential rating for each book on the lists.
- **Interactive UI**: Easy-to-use interface for uploading data and viewing predictions.
- **Quote Display**: Displays rotating book quotes while the model is being prepared.

## How It Works

1. **Data Upload**: Users upload their Goodreads reading history CSV file.
2. **Model Creation**: The system processes the CSV file to create a personalized recommendation model. For each book title and author, I ping the Google Books API to retrieve a book description. That book description is converted to embeddings using the Gpt-4o API. The neural network is trained on the embeddings for each book description & the book rating to build the model for the user.  
3. **Book List Display**: Users can view famous book lists (e.g., Obama's 2023 picks, Pulitzer winners).
4. **Rating Prediction**: For each book in the lists, the system predicts how the user would rate it based on their reading history.
5. **Results Display**: Predicted ratings are displayed alongside each book in the lists.

## Technology Stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python with Flask
- APIs: Google Books API, GPT 4o (for fetching book descriptions)
- Data Processing: Neural network with GPT 4 embeddings for book descriptions

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/book-recommender.git
   ```
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your Google Books API key:
   - Obtain an API key from [Google Cloud Console](https://console.cloud.google.com/)
   - Add your API key to the configuration file

4. Run the application:
   ```
   python app.py
   ```

## Usage

1. Export your Goodreads reading history:
   - Log in to your Goodreads account
   - Click on the 'Export' option
   - Download the generated CSV file
2. Open the Book Recommender application in your web browser
3. Upload your Goodreads CSV file
4. Wait for the model to be created (enjoy the book quotes!)
5. Explore the book lists and see your predicted ratings

## Future Enhancements

- [ ] Add more curated book lists
- [ ] Implement user accounts for saving preferences
- [ ] Enhance the recommendation algorithm
- [ ] Add a feature to compare predictions with friends

## Contributing

Contributions to improve the Book Recommender are welcome! Please feel free to submit a Pull Request.


## Acknowledgements

- Barack Obama for his annual book recommendations
- Goodreads for providing exportable reading history
- Google Books API for book information
- Claude Sonnet 
