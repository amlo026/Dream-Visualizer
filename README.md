# Dream-Visualizer

This Streamlit app uses **Google's Gemini AI** to interpret dreams, generate a surreal dream-inspired image, and suggest a **therapeutic music theme** turning your dream into a multimodal experience.


## Features

-  **Dream Interpretation:** Understand emotional and symbolic meanings using Gemini 1.5 Flash.
-  **Image Generation:** Get a surreal image based on your dream narrative.
-  **Therapeutic Music Prompt:** Receive a customized music description based on your dream for a calming experience.

## How It Works
1. **User types a dream** (e.g., "I was running through a forest being chased by shadows").
2. **AI responds with**:
   - Interpretation of the dream
   - AI-generated visual art
   - A music theme to represent the emotional tone of the dream
  

## 🔧 Technologies

- [Streamlit](https://streamlit.io/)
- [Google Generative AI](https://ai.google.dev/)
- [Gemini 1.5 Flash](https://ai.google.dev/gemini-api/docs)
- Python, dotenv, Pillow

##  Setup Instructions

1. Clone or download the project.

2. Create a `.env` file in the same directory as the app:

```
API_KEY=your_google_api_key_here
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the app:

```bash
streamlit run dream_diary_app_fixed_v4.py
```

---
