from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Load models
vgg_category_model = load_model('models/Category_validation.h5')
resnet_lungs_model = load_model('models/resnet_lungs_model.h5')
resnet_kidney_model = load_model('models/resnet_kidney_model.h5')

# Labels
category_labels = {0: 'kidney', 1: 'lungs', 2: 'random'}
lungs_labels = {0: 'COVID', 1: 'Normal', 2: 'PNEUMONIA', 3: 'Tuberculosis'}
kidney_labels = {0: 'Renal Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    return np.argmax(predictions[0])

@app.post("/predict")
async def predict_image_class(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_array = preprocess_image(img_bytes)
    
    # Predict with VGG model
    category_pred = predict_image(vgg_category_model, img_array)
    category_label = category_labels[category_pred]
    
    if category_label == 'random':
        return JSONResponse(content={"result": "random"})
    
    if category_label == 'lungs':
        # Predict with resnet_lungs model
        lung_pred = predict_image(resnet_lungs_model, img_array)
        lung_label = lungs_labels[lung_pred]
        return JSONResponse(content={"result": lung_label})
    
    if category_label == 'kidney':
        # Predict with resnet_kidney model
        kidney_pred = predict_image(resnet_kidney_model, img_array)
        kidney_label = kidney_labels[kidney_pred]
        return JSONResponse(content={"result": kidney_label})

    return JSONResponse(content={"result": "Unknown category"})
