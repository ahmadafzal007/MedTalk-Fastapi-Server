from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import io
import numpy as np
import openai
import os
from dotenv import load_dotenv
import cloudinary
import cloudinary.uploader
from openai import OpenAI

# Load environment variables
load_dotenv()

app = FastAPI()

# The class labels
classes = {
    0: 'Normal beat',
    1: 'Supraventricular premature beat',
    2: 'Premature ventricular contraction',
    3: "Fusion of ventricular and normal beat",
    4: 'Unclassifiable beat'
}

# Load the RandomForest model from the pickle file
with open('models/ECG.pkl', 'rb') as f:
    model = pickle.load(f)

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai.api_key)

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET')
)

def preprocess_image_from_csv(csv_file_path, row_number):
    # Load the CSV file
    data = pd.read_csv(csv_file_path, header=None)
    
    # Separate input (features) and output (labels)
    x_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
    
    # Get the specific row for prediction
    single_row = x_data.iloc[row_number, :].values.reshape(1, -1)  # Reshape for prediction
    actual_class = int(y_data.iloc[row_number])

    # Predict the class
    predicted_class = int(model.predict(single_row)[0])

    # Get class labels from the dictionary
    actual_class_label = classes.get(actual_class, "Unknown")
    predicted_class_label = classes.get(predicted_class, "Unknown")

    return predicted_class_label

async def create_plot(csv_file_path, row_number):
    # Load the CSV file
    data = pd.read_csv(csv_file_path, header=None)
    
    # Separate input (features) and output (labels)
    x_data = data.iloc[:, :-1]
    
    # Get the specific row for visualization
    single_row = x_data.iloc[row_number, :].values.reshape(1, -1)  # Reshape for plotting

    # Create a plot
    plt.figure(figsize=(20, 4))  # Stretch horizontally
    plt.plot(single_row.flatten(), color='blue', label=f"ECG Signal (Row {row_number})")
    plt.title(f"ECG Signal for Row {row_number}")
    plt.xlabel('Time Steps')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend(loc='upper right')

    # Save the plot to a BytesIO object
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png')
    img_io.seek(0)
    plt.close()

    return img_io

async def upload_image_to_cloudinary(img_io):
    cloudinary_response = cloudinary.uploader.upload(img_io, resource_type='image')
    return cloudinary_response['secure_url']  # Get the secure URL

async def generate_gpt_response(disease, user_prompt):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",  # Adjust to the appropriate model
            messages=[
                {"role": "system", "content": "MedTalk is a medical chatbot. It is designed to assist the doctors. That also provide assistance on heart arrhythmia detection through CSV file. there is another function that detects the arrhythmia and combines that detected arrhythmia with the user prompt. you have to write a detailed response on that detected arrhythmia by keeping the relevence with the user prompt. if the detected arrhythmia is not relevant to the user prompt then you can write a general response. you can also provide the general information about the detected arrhythmia."},
                {"role": "user", "content": f"The detected arrhythmia is: {disease}. {user_prompt}"}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return str(e)

@app.post("/predict")
async def predict_from_csv(file: UploadFile = File(...), user_prompt: str = Form(...)):
    # Save the uploaded CSV file temporarily
    temp_csv_path = 'temp_file.csv'
    with open(temp_csv_path, 'wb') as f:
        f.write(await file.read())
    
    # Fixed row number for prediction
    row_number = 0
    
    # Get the prediction
    predicted_class_label = preprocess_image_from_csv(temp_csv_path, row_number)
    
    # Create the plot
    img_io = await create_plot(temp_csv_path, row_number)

    # Upload the plot image to Cloudinary
    plot_url = await upload_image_to_cloudinary(img_io)

    # Generate a comprehensive response using the GPT model
    gpt_response = await generate_gpt_response(predicted_class_label, user_prompt)

    # Return the result, the plot image URL, and the GPT response
    return JSONResponse(
        content={
            "predicted_class": predicted_class_label,
            "gpt_response": gpt_response,
            "plot_url": plot_url  # URL to the uploaded plot image
        }
    )

@app.get("/plot_image")
async def get_plot_image():
    return JSONResponse(content={"message": "Use the /predict endpoint to get the plot image URL."})
