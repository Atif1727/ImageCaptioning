from flask import Flask, render_template, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'


max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")


def preprocess_image(image):
    return Image.open(image).convert('RGB')

def generate_captions(image: Image):
    inputs = processor(preprocess_image(image), return_tensors="pt")

    # Generate the output    
    outputs=model.generate(**inputs,**gen_kwargs)

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
        return 

    image = request.files['image']
    image_path = Path(app.config['UPLOAD_FOLDER']) / image.filename

    try:
        with image_path.open('wb') as f:
            f.write(image.read())
    except Exception as e:
        return jsonify({'error': 'failed to save image'})

    captions_str = generate_captions(image_path)

    return render_template('index.html', captions=captions_str)

if __name__ == '__main__':
    app.run(debug=True)