import os
from flask import Flask, render_template, request, jsonify
from google.cloud import vision
import io

app = Flask(__name__)

# Ensure credentials are found
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'serviceaccount.json'

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Image Labeler</title>
    <h1>Upload Image for Labeling</h1>
    <form method=post enctype=multipart/form-data action="/predict">
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files: return 'No file'
    file = request.files['file']
    content = file.read()

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    
    labels = [f"{label.description} ({label.score:.2%})" for label in response.label_annotations]
    return jsonify(labels=labels)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
