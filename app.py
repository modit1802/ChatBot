import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(page_title="Sign Language Chatbot", page_icon="🤟", layout="centered")

# --- CSS Styling for Chatbot UI ---
st.markdown("""
    <style>
    body {
        background-color: #f7f9fc;
    }
    .chat-bubble {
        padding: 15px;
        border-radius: 15px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .user {
        background-color: #DCF8C6;
        align-self: flex-end;
        margin-left: auto;
    }
    .bot {
        background-color: #f0f2f6;
        align-self: flex-start;
        margin-right: auto;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h2 style='text-align:center;'>🤖 Learn Sign Language with AI</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask about any sign and watch it come alive in a video!</p>", unsafe_allow_html=True)

# --- Content Moderation ---
MODERATION_KEYWORDS = [
    'hate', 'violence', 'sexual', 'kill', 'attack', 'harass', 'abuse',
    'racist', 'nude', 'porn', 'nsfw', 'slut', 'fuck', 'shit', 'bitch',
    'asshole', 'dick', 'pussy', 'cunt', 'bastard', 'motherfucker'
]

def moderate_query(query):
    return any(word in query.lower() for word in MODERATION_KEYWORDS)

# --- DeepSeek API ---
deepseek_api_url = "https://api.deepseek.com"
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    st.error("API key not found. Please check your .env file.")
    st.stop()

# --- Load Data ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("videos.csv")
        if not all(col in df.columns for col in ['Relations', 'URL']):
            st.error("CSV must have 'Relations' and 'URL' columns.")
            st.stop()
        return df
    except FileNotFoundError:
        st.error("videos.csv not found.")
        st.stop()

df = load_data()

# --- Load Embedding Model ---
@st.cache_resource
def load_model():
    with st.spinner("Loading AI brain..."):
        return SentenceTransformer("models/all-MiniLM-L6-v2")

model = load_model()

# --- Precompute Embeddings ---
@st.cache_data
def get_embeddings():
    video_texts = [row['Relations'] for _, row in df.iterrows()]
    st.progress(50, text="Processing sign database...")
    embeddings = model.encode(video_texts, convert_to_tensor=True)
    st.progress(100, text="All set to learn!")
    return embeddings

video_embeddings = get_embeddings()

# --- Chat Input ---
query = st.text_input("👤 You:", placeholder="Type a word/phrase to see the sign...")

if query:
    st.markdown(f"<div class='chat-container'><div class='chat-bubble user'>👤 {query}</div></div>", unsafe_allow_html=True)

    if moderate_query(query):
        st.markdown(f"<div class='chat-container'><div class='chat-bubble bot'>🚫 Sorry, I can only help with sign language learning.</div></div>", unsafe_allow_html=True)
    else:
        with st.spinner("🔍 Finding the best match..."):
            query_embedding = model.encode(query, convert_to_tensor=True)
            scores = util.cos_sim(query_embedding, video_embeddings)[0]
            best_idx = scores.argmax().item()
            best_score = scores[best_idx].item()

        similarity_threshold = 0.3

        if best_score >= similarity_threshold:
            row = df.iloc[best_idx]
            st.markdown(f"<div class='chat-container'><div class='chat-bubble bot'>✅ Here's the sign for: <strong>{row['Relations']}</strong></div></div>", unsafe_allow_html=True)
            st.video(row['URL'])

            with st.expander("💡 Practice Tips"):
                st.markdown("""
                    - Watch hand movements closely  
                    - Practice in front of a mirror  
                    - Repeat it 5–10 times  
                    - Record and compare yourself  
                """)
        else:
            try:
                headers = {
                    'Authorization': f'Bearer {deepseek_api_key}',
                    'Content-Type': 'application/json'
                }
                prompt = f"""
                You are a helpful assistant for sign language. The user asked: '{query}'.
                Respond with short, helpful guidance on sign language only. Be kind and concise.
                """
                data = {
                    "question": prompt,
                    "model": "deepseek-large",
                    "temperature": 0.3,
                    "max_tokens": 100
                }

                with st.spinner("🧠 Thinking..."):
                    response = requests.post(deepseek_api_url, headers=headers, json=data, timeout=10)

                if response.status_code == 200:
                    reply = response.json().get("response", "I'm here to help you learn sign language!")
                else:
                    reply = "Try asking about basic signs like 'hello' or 'thank you'."

                if moderate_query(reply):
                    reply = "Let's stay focused on learning sign language."

                st.markdown(f"<div class='chat-container'><div class='chat-bubble bot'>🤖 {reply}</div></div>", unsafe_allow_html=True)

                # Suggest closest signs
                st.subheader("🔍 Similar signs to explore:")
                top_k = min(3, len(df))
                top_indices = scores.argsort(descending=True)[:top_k]

                cols = st.columns(top_k)
                for i, idx in enumerate(top_indices):
                    if scores[idx] > 0.1:
                        match_row = df.iloc[idx]
                        with cols[i]:
                            st.markdown(f"**{match_row['Relations']}**")
                            st.video(match_row['URL'])

            except requests.Timeout:
                st.warning("⏳ It's taking too long. Try something simpler.")
            except:
                st.error("Something went wrong. Let's try again!")

# --- Footer ---
st.markdown("---")
st.markdown("""
    <div style='text-align:center;'>
        <h4>🌟 Community Guidelines</h4>
        <p>✅ Only sign language-related questions</p>
        <p>✅ Respectful and kind interaction</p>
        <p>✅ Practice and have fun learning! 🤟</p>
    </div>
""", unsafe_allow_html=True)
