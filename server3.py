from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
import os
from openai import OpenAI


# Load environment variables
load_dotenv()

# FastAPI instance
app = FastAPI()

# # Set API key
openai.api_key = os.getenv('OPENAI_API_KEY')
print(openai.api_key)

client = OpenAI(api_key=openai.api_key)

# fine_tuned_model = "ft:gpt-4o-mini-2024-07-18:personal:medtalk:A9dMB3FD"
fine_tuned_model = "gpt-4o-mini"

# Define the Pydantic model for the message (JSON structure)
class Message(BaseModel):
    content: str

@app.post("/generate_completion/")
async def generate_completion(prompt: Message):
    try:
        # Use the content from the message body (JSON)
        content = prompt.content
        print(content)
        
        
        completion = client.chat.completions.create(
            model=fine_tuned_model ,
            messages=[
                {"role": "system", "content": "MedTalk is a medical chatbot. You will only respond to medical-related queries and provide answers in a well-structured form. If I ask you anything other than medical or healthcare-related, you should not answer. Medtalk is designmed to provide assistance to the doctor while diagnosing their patients and helping the medical science students in their research work"},
                {"role": "user", "content": content}
            ]
        )
        print("Assistant's Response:", completion.choices[0].message.content)

        # Return the chatbot's response
        return {"completion": completion.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


