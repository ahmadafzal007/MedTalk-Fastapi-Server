from openai import OpenAI
import os
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Form, Query
from pydantic import BaseModel


# Load environment variables
load_dotenv()
app = FastAPI()

# Set API key
openai.api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=openai.api_key)

class Message(BaseModel):
    content: str

@app.post("/generate_completion/")
async def generate_completion(content: str = Query(None)):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content}
            ]
        )

        return {"completion": completion.choices[0].message.content}

    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")




