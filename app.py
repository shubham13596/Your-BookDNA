from flask import Flask, request, jsonify, send_from_directory
import os
import requests
import json
import numpy as np
import tempfile
from werkzeug.utils import secure_filename
from book_recommender import prepare_model, predict_rating

app = Flask(__name__)

# Global variable to store the trained model
trained_model = None

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    global trained_model
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.csv'):
        filename = secure_filename(file.filename)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            file_path = temp_file.name
            file.save(file_path)
        
        try:
            trained_model = prepare_model(file_path)
            os.unlink(file_path)  # Remove the temporary file
            return jsonify({'message': 'Model prepared successfully'}), 200
        except Exception as e:
            os.unlink(file_path)  # Remove the temporary file
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/predict', methods=['POST'])
def predict():
    global trained_model
    if trained_model is None:
        return jsonify({'error': 'Model not prepared. Please upload CSV first.'}), 400
    
    data = request.json
    if 'description' not in data:
        return jsonify({'error': 'No description provided'}), 400
    
    description = data['description']
    try:
        score = predict_rating(trained_model, description)
        return jsonify({'score': score}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def read_json_file(file_path):
    encodings = ['utf-8', 'latin-1', 'ascii']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to read the file {file_path} with any of the attempted encodings.")
    
# Load book data (you'll need to create these JSON files)
with open('pulitzer_books.json', 'r') as f:
    pulitzer_books = json.load(f)

with open('obama_books.json', 'r') as f:
    obama_books = read_json_file('obama_books.json')

# Add the root route
@app.route('/')
def root():
    return send_from_directory('.', 'index.html')

@app.route('/books/PulitzerBooks', methods=['GET'])
def get_pulitzer_books():
    return jsonify(pulitzer_books)

@app.route('/books/ObamaBooks', methods=['GET'])
def get_obama_books():
    return jsonify(obama_books)

if __name__ == '__main__':
    app.run(debug=False) 



