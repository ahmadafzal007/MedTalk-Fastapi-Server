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

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai.api_key)

# # Configure Cloudinary
# cloudinary.config(
#     cloud_name=os.getenv('CLOUD_NAME'),
#     api_key=os.getenv('API_KEY'),
#     api_secret=os.getenv('API_SECRET')
# )



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

        confidence_str = f" with a model's confidence score of {confidence_score:.2f}%" if confidence_score is not None else ""
        print(confidence_str)
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


