from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from openai import OpenAI
import os
import numpy as np
import io
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import cloudinary
import cloudinary.uploader

# Load environment variables
load_dotenv()

# FastAPI instance
app = FastAPI()

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai.api_key)

# Load models
vgg_category_model = load_model('models/Category_validation.h5')
resnet_lungs_model = load_model('models/resnet_lungs_model.h5')
resnet_kidney_model = load_model('models/resnet_kidney_model.h5')

# Load the RandomForest model for heart disease detection
with open('models/ECG.pkl', 'rb') as f:
    ecg_model = pickle.load(f)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET')
)

# Labels
category_labels = {0: 'kidney', 1: 'lungs', 2: 'random'}
lungs_labels = {0: 'COVID', 1: 'Normal', 2: 'PNEUMONIA', 3: 'Tuberculosis'}
kidney_labels = {0: 'Renal Cyst', 1: 'Normal', 2: 'Stone', 3: 'Tumor'}
classes = {
    0: 'Normal beat',
    1: 'Supraventricular premature beat',
    2: 'Premature ventricular contraction',
    3: "Fusion of ventricular and normal beat",
    4: 'Unclassifiable beat'
}

async def generate_completion(prompt: str, previous_message: str = None):
    try:
       
        context = f"previous prompt and response: {previous_message}\n\n" if previous_message else ""
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
              {"role": "system", "content": "MedTalk is a medical chatbot. You will only respond to medical-related queries and provide answers in a well-structured form. If I ask you anything other than medical or healthcare-related, you should not answer. Medtalk is designmed to provide assistance to the doctor while diagnosing their patients and helping the medical science students in their research work"},
              {"role": "user", "content": f"{context}{prompt}"}
              ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(model, img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence_score = np.max(predictions[0])
    confidence_score = confidence_score*100

    return predicted_class, confidence_score

async def generate_finetunedModel_response(disease, user_prompt, confidence_score=None, previous_message=None):
    try:
        confidence_str = f" with a confidence score of {confidence_score:.2f}%" if confidence_score is not None else ""
        context = f"previous prompt and response: {previous_message}\n\n" if previous_message else ""
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
              {"role": "system", "content": "MedTalk is a medical chatbot. It is designed to assist the doctors. That also provide assistance on medical imaging including lungs x-ray and kidney ctscan. there is another function that detects the disease and combines that detected disease with the user prompt. you have to write a detailed response on that detected diease by keeping the relevence with the user prompt. if the detected disease is not relevant to the user prompt then you can write a general response. you can also provide the general information about the detected disease. Provide answers in a well-structured form. If I ask you anything other than medical or healthcare-related you should not answer. If the image is random then you have to tell that the provided image is not a lings x-ray or kidney ct scan"},
              {"role": "user", "content": f"{context}The detected disease is: {disease}{confidence_str}. {user_prompt}"}
              ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return str(e)

# Modify preprocess_image_from_csv to return predicted class and confidence score
def preprocess_image_from_csv(csv_file_path, row_number):
    data = pd.read_csv(csv_file_path, header=None)
    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    single_row = x_data.iloc[row_number, :].values.reshape(1, -1)
    actual_class = int(y_data.iloc[row_number])
    prediction_probs = ecg_model.predict_proba(single_row)[0]
    predicted_class = np.argmax(prediction_probs)
    confidence_score = np.max(prediction_probs)
    confidence_score = confidence_score*100
    return classes.get(predicted_class, "Unknown"), confidence_score




async def create_plot(csv_file_path, row_number):
    data = pd.read_csv(csv_file_path, header=None)
    x_data = data.iloc[:, :-1]
    single_row = x_data.iloc[row_number, :].values.reshape(1, -1)
    plt.figure(figsize=(20, 4))
    plt.plot(single_row.flatten(), color='blue', label=f"ECG Signal (Row {row_number})")
    plt.title(f"ECG Signal for Row {row_number}")
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(loc='upper right')
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()
    return img_io

async def upload_image_to_cloudinary(img_io):
    cloudinary_response = cloudinary.uploader.upload(img_io, resource_type='image')
    return cloudinary_response['secure_url']

async def generate_gpt_response(disease, user_prompt, confidence_score=None,  previous_message=None):
    try:
        confidence_str = f" with a confidence score of {confidence_score:.2f}%" if confidence_score is not None else ""
        context = f"previous prompt and response: {previous_message}\n\n" if previous_message else ""
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
              {"role": "system", "content": "MedTalk is a medical chatbot. It is designed to assist the doctors. That also provide assistance on heart arrhythmia detection through CSV file That contains time series data of ECG. there is another function that detects the arrhythmia and combines that detected arrhythmia with the user prompt. you have to write a detailed response on that detected arrhythmia by keeping the relevence with the user prompt. if the detected arrhythmia is not relevant to the user prompt then you can write a general response. you can also provide the general information about the detected arrhythmia. Provide answers in a well-structured form. If I ask you anything other than medical or healthcare-related you should not answer."},
              {"role": "user", "content": f"{context}The detected arrhythmia is: {disease}{confidence_str}. {user_prompt}"}
              ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return str(e)

@app.post("/generate_response/")
async def generate_response(prompt: str = Form(...), file: UploadFile = File(None), previous_message: str = Form(None)):
    if file:
        file_extension = file.filename.split('.')[-1].lower()


        if file_extension in ['jpg', 'jpeg', 'png']:
            # Handle image prediction
            img_bytes = await file.read()
            img_array = preprocess_image(img_bytes)
            category_pred, category_confidence = predict_image(vgg_category_model, img_array)
            category_label = category_labels[category_pred]

            if category_label == 'random':
                result = "random"
                gpt_response = await generate_finetunedModel_response(result, prompt, previous_message=previous_message)
            elif category_label == 'lungs':
                lung_pred, lung_confidence = predict_image(resnet_lungs_model, img_array)
                result = lungs_labels[lung_pred]
                gpt_response = await generate_finetunedModel_response(result, prompt, lung_confidence, previous_message=previous_message)
            elif category_label == 'kidney':
                kidney_pred, kidney_confidence = predict_image(resnet_kidney_model, img_array)
                result = kidney_labels[kidney_pred]
                gpt_response = await generate_finetunedModel_response(result, prompt, kidney_confidence, previous_message=previous_message)
            else:
                result = "Unknown category"
                gpt_response = await generate_finetunedModel_response(result, prompt, previous_message=previous_message)

            return JSONResponse(content={"result": result, "gpt_response": gpt_response})


        elif file_extension == 'csv':
            # Handle heart disease detection
            temp_csv_path = 'temp_file.csv'
            with open(temp_csv_path, 'wb') as f:
                f.write(await file.read())

            row_number = 0
            try:
                # Try to process the CSV file
                predicted_class_label, ecg_confidence = preprocess_image_from_csv(temp_csv_path, row_number)
                img_io = await create_plot(temp_csv_path, row_number)
                plot_url = await upload_image_to_cloudinary(img_io)
                gpt_response = await generate_gpt_response(predicted_class_label, prompt, ecg_confidence, previous_message=previous_message)

                return JSONResponse(content={
                    "predicted_class": predicted_class_label,
                    "gpt_response": gpt_response,
                    "plot_url": plot_url
                })

            except Exception as e:
                # If there is an error, generate a GPT response indicating invalid data
                gpt_response = await generate_gpt_response("Invalid ECG data", prompt)
                
                return JSONResponse(content={
                    "error": "The uploaded CSV file does not contain valid ECG data.",
                    "gpt_response": gpt_response
                })



        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a JPG, JPEG, PNG, or CSV file.")

    else:
        # If no image or file is provided, generate a completion based on the prompt
        gpt_response = await generate_completion(prompt, previous_message=previous_message)
        return JSONResponse(content={"gpt_response": gpt_response})



