import cv2
import numpy as np
import pytesseract
from transformers import pipeline
import re

# Specify the Tesseract executable path if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Adjust this path

# Initialize the text generation pipeline 
text_processor = pipeline('text-generation', model='distilgpt2')

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply thresholding
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def ocr_image(processed_image):
    # Use Tesseract to read text from the image
    text = pytesseract.image_to_string(processed_image)
    return text

def clean_text(text):
    # Remove non-printable characters
    text = re.sub(r'[^ -~]+', '', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    return text

def process_text(text):
    # Use the text generation pipeline to process the text
    processed_text = text_processor(text, max_new_tokens=50, truncation=True, num_return_sequences=1)[0]['generated_text']
    return processed_text

def convert_to_braille(text):
    from braille import text_to_braille
    braille = text_to_braille(text)
    return braille
