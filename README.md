# Personalized Book Recommender

## Overview

This project is a personalized book recommender system that leverages your Goodreads reading history to predict how you would rate books from famous reading lists. It's designed to help you discover new books that align with your tastes, based on your past reading experiences.

It's now available in production. You can try it on your Goodreads data! 
Website: https://bookrecommendations1-hcr77cz3ga-el.a.run.app

Check out this video demo of the Book Recommender in action:(https://youtu.be/35y6kJRjng4?si=z_RG16PuCc4CY-NK)

## Features

- **CSV Upload**: Upload your Goodreads reading history to create a personalized model.
- **Famous Book Lists**: Currently includes Obama's 2023 Reading List and Pulitzer Prize Masterpieces.
- **Personalized Ratings**: Predicts your potential rating for each book on the lists.
- **Interactive UI**: Easy-to-use interface for uploading data and viewing predictions.
- **Quote Display**: Displays rotating book quotes while the model is being prepared.

## How It Works

1. **Data Upload**: Users upload their Goodreads reading history CSV file.
2. **Model Creation**: The system processes the CSV file to create a personalized recommendation model.
3. **Book List Display**: Users can view famous book lists (e.g., Obama's 2023 picks, Pulitzer winners).
4. **Rating Prediction**: For each book in the lists, the system predicts how the user would rate it based on their reading history.
5. **Results Display**: Predicted ratings are displayed alongside each book in the lists.

## Technology Stack

- Frontend: HTML, CSS, JavaScript
- Backend: [Python with Flask]
- APIs: Google Books API (for fetching book descriptions)
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
   - Go to 'My Books'
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
