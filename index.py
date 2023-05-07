import logging
from typing import List
from contextlib import closing
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


app = Flask(_name_)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the captioning model and processor from a configuration file
with open('config.json') as f:
    config = json.load(f)

processor = BlipProcessor.from_pretrained(config['processor_path'])
model = BlipForConditionalGeneration.from_pretrained(config['model_path'])

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define a function to preprocess the uploaded image
def preprocess_image(image) -> Image:
    return Image.open(image).convert('RGB')

# Define a function to generate captions for an image
def generate_captions(image: Image) -> List[str]:
    # Process the image
    inputs = processor(preprocess_image(image), return_tensors="pt")

    # Generate the output
    try:
        with closing(model):
            outputs = model.generate(
                **inputs,
                max_length=config['max_length'],
                num_beams=config['num_beams'],
                num_return_sequences=config['num_return_sequences'],
                temperature=config['temperature'],
            )
    except Exception as e:
        logging.error(f"Failed to generate captions: {e}")
        return []

    # Decode the output and return the captions
    captions = []
    for output in outputs:
        caption = processor.decode(output, skip_special_tokens=True)
        captions.append(caption)

    return captions

# Define a route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle image uploads and generate captions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'no image uploaded'})

    # Save the uploaded image to the uploads folder
    image = request.files['image']
    image_path = Path(app.config['UPLOAD_FOLDER']) / image.filename

    try:
        with image_path.open('wb') as f:
            f.write(image.read())
    except Exception as e:
        logging.error(f"Failed to save image: {e}")
        return jsonify({'error': 'failed to save image'})

    # Generate captions for the uploaded image
    captions = generate_captions(image_path)

    # Return the generated captions as JSON
    return jsonify({'captions': captions})