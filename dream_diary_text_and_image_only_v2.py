
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

# Check if API key is provided
if not api_key:
    st.error("❌ API Key is missing. Please check your .env file.")
    st.stop()

# Configure Generative AI
genai.configure(api_key=api_key)

# Initialize model
model_text = genai.GenerativeModel("gemini-1.5-flash")
model_image = genai.GenerativeModel("gemini-1.5-flash")

# Streamlit app UI
st.title("🌙 Dream-to-Visual Diary")
dream_input = st.text_area("Describe your dream here:", height=200)

if st.button("Interpret & Visualize"):
    if not dream_input.strip():
        st.warning("Please enter your dream description.")
    else:
        with st.spinner("🧠 Interpreting your dream..."):
            try:
                interpretation_prompt = f"""
                I had the following dream:

                "{dream_input}"

                This was just a dream. Please interpret what this dream might mean and any psychological insights.
                """
                response_text = model_text.generate_content(interpretation_prompt)
                interpretation = response_text.candidates[0].content.parts[0].text
                st.subheader("📝 Dream Interpretation")
                st.write(interpretation)
            except Exception as e:
                st.error(f"Failed to interpret dream: {e}")

        with st.spinner("🎨 Generating dream image..."):
            try:
                response_image = model_image.generate_content(
                    [f"Create a surreal image based on this dream: {dream_input}"]
                )
                for part in response_image.parts:
                    if hasattr(part, "inline_data") and part.inline_data.data:
                        image = Image.open(BytesIO(base64.b64decode(part.inline_data.data)))
                        st.subheader("🖼️ Dream Visualization")
                        st.image(image, caption="AI-generated image from your dream")
            except Exception as e:
                st.error(f"Failed to generate image: {e}")
