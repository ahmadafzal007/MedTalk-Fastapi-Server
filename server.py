from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import io
from PIL import Image
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')


client = OpenAI(api_key=openai.api_key)


# Load models
vgg_category_model = load_model('models/Category_validation.h5')
resnet_lungs_model = load_model('models/resnet_lungs_model.h5')
resnet_kidney_model = load_model('models/resnet_kidney_model.h5')

# Labels
category_labels = {0: 'kidney', 1: 'lungs', 2: 'random'}
lungs_labels = {0: 'COVID', 1: 'Normal', 2: 'PNEUMONIA', 3: 'Tuberculosis'}
kidney_labels = {0: 'Renal Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}

# Array to store results
results_array = []

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

async def generate_finetunedModel_response(disease, user_prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to the appropriate model
            messages=[
                {"role": "system", "content": "MedTalk is a medical chatbot. It is designed to assist the doctors. That also provide assistance on medical imaging including lungs x-ray and kidney ctscan. there is another function that detects the disease and combines that detected disease with the user prompt. you have to write a detailed response on that detected diease by keeping the relevence with the user prompt. if the detected disease is not relevant to the user prompt then you can write a general response. you can also provide the general information about the detected disease."},
                {"role": "user", "content": f"The detected disease is: {disease}. {user_prompt}"}
            ]
        )
        
        print("Assistant's Response:", completion.choices[0].message.content)
        
        return completion.choices[0].message.content
    except Exception as e:
        return str(e)

@app.post("/predict")
async def predict_image_class(file: UploadFile = File(...), user_prompt: str = Form(...)):
    img_bytes = await file.read()
    img_array = preprocess_image(img_bytes)
    
    # Predict with VGG model
    category_pred = predict_image(vgg_category_model, img_array)
    category_label = category_labels[category_pred]
    
    if category_label == 'random':
        result = "random"
    elif category_label == 'lungs':
        lung_pred = predict_image(resnet_lungs_model, img_array)
        result = lungs_labels[lung_pred]
    elif category_label == 'kidney':
        kidney_pred = predict_image(resnet_kidney_model, img_array)
        result = kidney_labels[kidney_pred]
    else:
        result = "Unknown category"

    # Store the result in the results array
    results_array.append(result)

    # Generate a comprehensive response using the GPT model
    gpt_response = await generate_finetunedModel_response(result, user_prompt)

    return JSONResponse(content={"result": result, "gpt_response": gpt_response})
