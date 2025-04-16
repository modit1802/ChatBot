import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import requests
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set page config MUST be the first Streamlit command
st.set_page_config(page_title="Sign Language Video Chatbot", page_icon="🤖")

# --- Content Moderation Setup ---
MODERATION_KEYWORDS = [
    'hate', 'violence', 'sexual', 'kill', 'attack', 'harass', 'abuse',
    'racist', 'nude', 'porn', 'nsfw', 'slut', 'fuck', 'shit', 'bitch',
    'asshole', 'dick', 'pussy', 'cunt', 'bastard', 'motherfucker'
]

def moderate_query(query):
    """Check if query contains inappropriate content"""
    query_lower = query.lower()
    return any(bad_word in query_lower for bad_word in MODERATION_KEYWORDS)

# --- Main App ---
st.markdown("<h1 style='text-align: center;'>🤖 Sign Language Video Chatbot</h1>", unsafe_allow_html=True)

# DeepSeek API setup - Now using environment variable
deepseek_api_url = 'https://api.deepseek.com'
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')  # Securely loaded from .env

if not deepseek_api_key:
    st.error("API key not found. Please configure your .env file.")
    st.stop()

# Load CSV data with error handling
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("videos.csv")
        required_columns = ['Relations', 'URL']
        if not all(col in df.columns for col in required_columns):
            st.error(f"CSV file must contain these columns: {required_columns}")
            st.stop()
        return df
    except FileNotFoundError:
        st.error("videos.csv file not found. Please ensure it exists in your project directory.")
        st.stop()

df = load_data()

# Load embedding model with progress indicator
@st.cache_resource
def load_model():
    with st.spinner('Loading AI model... This may take a moment...'):
        return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Precompute embeddings with progress bar
@st.cache_data
def get_embeddings():
    video_texts = [f"{row['Relations']}" for _, row in df.iterrows()]
    progress_bar = st.progress(0, text="Preparing sign language database...")
    embeddings = model.encode(video_texts, convert_to_tensor=True)
    progress_bar.progress(100, text="Ready to help you learn!")
    return embeddings

video_embeddings = get_embeddings()

# --- User Input Section ---
query = st.text_input("You:", placeholder="Ask about any word or phrase in sign language...")

if query:
    # Content moderation check
    if moderate_query(query):
        st.error("Sorry, I can't respond to that request. Please ask about sign language instead.")
        st.stop()
    
    # Processing indicator
    with st.spinner('Finding the best sign language match...'):
        # Embed user query
        query_embedding = model.encode(query, convert_to_tensor=True)

        # Similarity matching
        scores = util.cos_sim(query_embedding, video_embeddings)[0]
        best_idx = scores.argmax().item()
        best_score = scores[best_idx].item()
    
    # Adjusted threshold
    similarity_threshold = 0.3
    
    if best_score >= similarity_threshold:
        # Show matching content
        selected_row = df.iloc[best_idx]
        relation = selected_row['Relations']
        video_url = selected_row['URL']
        
        st.success(f"Found a match for: {relation}")
        st.video(video_url)
        
        # Add learning tips
        with st.expander("💡 Learning Tips"):
            st.markdown("""
            1. **Watch** the hand shapes carefully
            2. **Notice** the movement direction
            3. **Practice** in front of a mirror
            4. **Repeat** 5-10 times for muscle memory
            5. **Record** yourself to compare
            """)
        
    else:
        # No good match found - use DeepSeek API with moderation
        try:
            headers = {
                'Authorization': f'Bearer {deepseek_api_key}',
                'Content-Type': 'application/json'
            }
            
            # Improved prompt with better instructions
            prompt = f"""You are a sign language teaching assistant. The user asked: '{query}'
            
            Guidelines for your response:
            - If this is sign language related, provide helpful information
            - If inappropriate, say: "I can only help with sign language"
            - If unrelated, suggest learning basic signs
            - Keep responses under 3 sentences
            - Always be encouraging
            
            Response:"""
            
            data = {
                "question": prompt,
                "model": "deepseek-large",
                "temperature": 0.3,
                "max_tokens": 100
            }
            
            with st.spinner('Consulting sign language resources...'):
                response = requests.post(deepseek_api_url, headers=headers, json=data, timeout=15)
            
            if response.status_code == 200:
                response_json = response.json()
                bot_response = response_json.get('response', "I can help you learn sign language. What would you like to know?")
                
                # Secondary moderation check on API response
                if moderate_query(bot_response):
                    bot_response = "Let's focus on learning sign language today!"
            else:
                bot_response = "Let's learn some sign language! Try asking about common signs."
                
            st.markdown(f"""
            <div style='background-color:#f0f2f6; padding:15px; border-radius:10px; margin:10px 0;'>
            <strong>🤖 Bot:</strong> {bot_response}
            </div>
            """, unsafe_allow_html=True)
            
            # Show closest matches with improved UI
            st.subheader("🔍 Similar signs you might want to learn:")
            top_k = min(3, len(df))
            top_indices = scores.argsort(descending=True)[:top_k]
            
            cols = st.columns(min(3, len(top_indices)))
            for i, idx in enumerate(top_indices):
                if scores[idx] > 0.1:
                    row = df.iloc[idx]
                    with cols[i]:
                        st.markdown(f"**{row['Relations']}**")
                        st.video(row['URL'])
            
        except requests.Timeout:
            st.warning("Taking longer than expected. Try asking about common signs like: Hello, Thank you, I love you")
        except Exception as e:
            st.error("Let's focus on learning sign language. Try asking about basic signs.")

# Add footer with improved guidelines
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <h4>Community Guidelines</h4>
    <p>✓ For sign language learning only</p>
    <p>✓ Be respectful and kind</p>
    <p>✓ Enjoy your learning journey! 🤟</p>
</div>
""", unsafe_allow_html=True)