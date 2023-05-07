from flask import Flask, render_template, request, jsonify
from typing import List
from contextlib import closing
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


def preprocess_image(image) -> Image:
    return Image.open(image).convert('RGB')

def generate_captions(image: Image, num_return_sequences: int = 1) -> List[str]:
    inputs = processor(preprocess_image(image), return_tensors="pt")

    # Generate the output    
    outputs = model.generate(
        **inputs,
        max_length=16,
        num_beams=4,
        num_return_sequences=num_return_sequences,
        temperature=1.0,
    )
    
    # Decode the output and return the captions
    captions = []
    for output in outputs:
        caption = processor.decode(output, skip_special_tokens=True)
        captions.append(caption)

    return captions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image']
    image_path = Path(app.config['UPLOAD_FOLDER']) / image.filename

    try:
        with image_path.open('wb') as f:
            f.write(image.read())
    except Exception as e:
        return jsonify({'error': 'Failed to save image'})

    captions_str = []

    if 'generate_button_single' in request.form:
        captions_str = generate_captions(image_path, num_return_sequences=1)
    elif 'generate_button_multiple' in request.form:
        captions_str = generate_captions(image_path, num_return_sequences=3)

    return render_template('index.html', captions=captions_str)

if __name__ == '__main__':
    app.run(debug=True)
