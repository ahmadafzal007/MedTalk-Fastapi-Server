import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# Load the models
covid_model_path = 'covid_model_adv.h5'
pneumonia_model_path = 'pneumonia_model_adv (1).h5'
validation_pipeline_path = 'ValidationPipeLine.h5'

covid_model = load_model(covid_model_path)
pneumonia_model = load_model(pneumonia_model_path)
validation_model = load_model(validation_pipeline_path)

def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    image = image.resize(target_size)
    if image.mode != 'RGB':
        image = image.convert('RGB')  # Ensure image is in RGB mode
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    return image_array

def predict(model, image):
    try:
        pred_prob = model.predict(image)
        return pred_prob[0][0]  # Return the probability of the positive class
    except Exception as e:
        print(f"Error predicting: {str(e)}")
        return -1

@app.route('/predict_disease', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(io.BytesIO(image_file.read()))
        preprocessed_image = preprocess_image(image)
    except Exception as e:
        return jsonify({'error': f"Error processing image: {str(e)}"}), 400

    preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

    # Validation pipeline to check if the image is an X-ray
    validation_threshold = 0.5
    validation_pred_prob = predict(validation_model, preprocessed_image)
    if validation_pred_prob < validation_threshold:
        return jsonify({'error': 'The image is not an X-ray'}), 400

    # Threshold for COVID-19 model
    covid_threshold = 0.5
    covid_pred_prob = predict(covid_model, preprocessed_image)
    covid_result = 'Covid detected' if covid_pred_prob < covid_threshold else 'Not Detected'

    # Threshold for pneumonia model
    pneumonia_threshold = 0.5
    pneumonia_pred_prob = predict(pneumonia_model, preprocessed_image)
    pneumonia_result = 'Pneumonia detected' if pneumonia_pred_prob > pneumonia_threshold else 'Not Detected'

    # Print the findings to the console
    print(f"Covid: {covid_result} Pneumonia: {pneumonia_result}")

    response = {
        'covid': covid_result,
        'pneumonia': pneumonia_result
    }

    return jsonify(response)

@app.route('/')
def index():
    return "Welcome to Disease Detection API!"

if __name__ == '__main__':
    # Print a message indicating that the models are loading
    print("Loading models...")
    app.run(debug=True)
