from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from process import preprocess_image, ocr_image, clean_text, process_text, convert_to_braille

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Perform OCR on the processed image
    text = ocr_image(processed_image)
    
    # Clean the extracted text
    cleaned_text = clean_text(text)
    
    # Process the cleaned text using the LLM
    processed_text = process_text(cleaned_text)
    
    # Convert the processed text to Braille
    braille = convert_to_braille(processed_text)
    
    return jsonify({'braille': braille, 'text': processed_text}), 200

if __name__ == '__main__':
    app.run(debug=True)
