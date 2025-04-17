from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load model and data
model = SentenceTransformer("model/all-MiniLM-L6-v2")
df = pd.read_csv("videos.csv")

# Load video sentences into embeddings
df["embedding"] = df["sentence"].apply(lambda x: model.encode(x, convert_to_tensor=True))

# Init app
app = FastAPI(title="Sign Language Video API")

# Request schema
class Query(BaseModel):
    text: str

@app.post("/get_sign_video")
def get_sign_video(query: Query):
    try:
        input_embedding = model.encode(query.text, convert_to_tensor=True)
        similarities = [util.pytorch_cos_sim(input_embedding, emb).item() for emb in df["embedding"]]
        best_index = similarities.index(max(similarities))
        return {
            "input": query.text,
            "matched_sentence": df.iloc[best_index]["sentence"],
            "video_url": df.iloc[best_index]["video_url"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
