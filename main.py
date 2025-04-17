from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import requests
from dotenv import load_dotenv
import os
from typing import Optional
import uvicorn
import torch

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODERATION_KEYWORDS = [
    'hate', 'violence', 'sexual', 'kill', 'attack', 'harass', 'abuse',
    'racist', 'nude', 'porn', 'nsfw', 'slut', 'fuck', 'shit', 'bitch',
    'asshole', 'dick', 'pussy', 'cunt', 'bastard', 'motherfucker'
]

class SignRequest(BaseModel):
    query: str
    similarity_threshold: Optional[float] = 0.3
    top_k: Optional[int] = 3

class SignResponse(BaseModel):
    query: str
    is_moderated: bool
    best_match: Optional[dict]
    similar_signs: Optional[list]
    ai_response: Optional[str]
    success: bool
    message: Optional[str]

def moderate_query(query: str) -> bool:
    return any(word in query.lower() for word in MODERATION_KEYWORDS)

# Load data with better error handling
def load_data():
    try:
        df = pd.read_csv("videos.csv")
        if not all(col in df.columns for col in ['Relations', 'URL']):
            raise ValueError("CSV must have 'Relations' and 'URL' columns.")
        # Reset index to ensure we have proper integer indices
        return df.reset_index(drop=True)
    except Exception as e:
        raise ValueError(f"Error loading data: {str(e)}")

try:
    df = load_data()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    video_texts = df['Relations'].tolist()
    video_embeddings = model.encode(video_texts, convert_to_tensor=True)
except Exception as e:
    print(f"Initialization error: {str(e)}")
    raise

@app.post("/search", response_model=SignResponse)
async def search_sign(request: SignRequest):
    response_data = {
        "query": request.query,
        "is_moderated": False,
        "best_match": None,
        "similar_signs": [],
        "ai_response": None,
        "success": True,
        "message": None
    }
    
    if moderate_query(request.query):
        response_data.update({
            "is_moderated": True,
            "success": False,
            "message": "Sorry, I can only help with sign language learning."
        })
        return response_data
    
    try:
        # Convert query to embedding
        query_embedding = model.encode(request.query, convert_to_tensor=True)
        
        # Calculate similarity scores
        scores = util.cos_sim(query_embedding, video_embeddings)[0]
        
        # Convert tensor to numpy array if needed
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # Get top matches
        best_idx = scores.argmax()
        best_score = scores[best_idx]
        
        if best_score >= request.similarity_threshold:
            # Safely access DataFrame row
            best_row = df.iloc[best_idx].to_dict()
            response_data["best_match"] = {
                "sign": best_row['Relations'],
                "url": best_row['URL'],
                "similarity_score": float(best_score)
            }
            
            # Get similar signs
            top_indices = scores.argsort()[::-1][:request.top_k+1]  # +1 to include best match
            for idx in top_indices:
                if idx != best_idx and scores[idx] > 0.1:
                    row = df.iloc[idx].to_dict()
                    response_data["similar_signs"].append({
                        "sign": row['Relations'],
                        "url": row['URL'],
                        "similarity_score": float(scores[idx])
                    })
        
        else:
            # Fallback to DeepSeek API if no good matches
            deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
            if deepseek_api_key:
                try:
                    headers = {
                        'Authorization': f'Bearer {deepseek_api_key}',
                        'Content-Type': 'application/json'
                    }
                    data = {
                        "question": f"Explain the sign for '{request.query}'",
                        "model": "deepseek-large",
                        "temperature": 0.3,
                        "max_tokens": 100
                    }
                    
                    api_response = requests.post(
                        "https://api.deepseek.com/v1/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=10
                    )
                    
                    if api_response.status_code == 200:
                        reply = api_response.json().get("choices", [{}])[0].get("message", {}).get("content", "I can help with sign language!")
                        if moderate_query(reply):
                            reply = "Let's focus on sign language learning."
                        response_data["ai_response"] = reply
                
                except Exception as api_error:
                    response_data["ai_response"] = "Try asking about common signs like 'hello' or 'thank you'."
            
            # Still suggest some similar signs
            top_indices = scores.argsort()[::-1][:request.top_k]
            for idx in top_indices:
                if scores[idx] > 0.1:
                    row = df.iloc[idx].to_dict()
                    response_data["similar_signs"].append({
                        "sign": row['Relations'],
                        "url": row['URL'],
                        "similarity_score": float(scores[idx])
                    })
    
    except Exception as e:
        response_data.update({
            "success": False,
            "message": f"An error occurred: {str(e)}"
        })
        # Log the full error for debugging
        print(f"Error processing query '{request.query}': {str(e)}")
    
    return response_data

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)