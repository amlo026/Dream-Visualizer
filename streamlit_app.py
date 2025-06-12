import streamlit as st
import asyncio
import io as sys_io
import wave
import nest_asyncio
import time

from google import genai
from google.genai import types
from google.genai.errors import ServerError
from PIL import Image
from io import BytesIO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Dream-to-Visual & Music", layout="wide")

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

# Initialize both clients
client = genai.Client(api_key=GOOGLE_API_KEY)
music_client = genai.Client(
    api_key=GOOGLE_API_KEY,
    http_options={"api_version": "v1alpha"}
)

# Patch the already-running event loop so we can use asyncio.run & top-level await
nest_asyncio.apply()

# ─── RETRY HELPER ───────────────────────────────────────────────────────────────
def generate_with_retry(fn, *args, max_retries=3, backoff=2, **kwargs):
    """
    Calls fn(*args, **kwargs) and retries up to max_retries times
    if a 503 UNAVAILABLE ServerError is raised.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except ServerError as e:
            if e.status_code == 503 and attempt < max_retries:
                st.warning(f"Service unavailable (503). Retrying in {backoff}s… (Attempt {attempt}/{max_retries})")
                time.sleep(backoff)
                backoff *= 2
                continue
            raise

# ─── MUSIC GENERATION ──────────────────────────────────────────────────────────
async def generate_music_from_text(prompt: str,
                                   bpm: int = 85,
                                   temperature: float = 1.0,
                                   duration: int = 15) -> bytes:
    """
    Streams music from Lyria, returns raw WAV bytes.
    """
    buffer = sys_io.BytesIO()

    async def _receive(session):
        async for msg in session.receive():
            chunk = msg.server_content.audio_chunks[0].data
            buffer.write(chunk)
            # stop once we've got enough
            if buffer.tell() > 48000 * 2 * 2 * duration:
                break
        buffer.seek(0)

    async with (
        music_client.aio.live.music.connect(model="models/lyria-realtime-exp") as sess,
        asyncio.TaskGroup() as tg
    ):
        tg.create_task(_receive(sess))
        await sess.set_weighted_prompts([
            types.WeightedPrompt(text=prompt, weight=1.0)
        ])
        await sess.set_music_generation_config(
            config=types.LiveMusicGenerationConfig(
                bpm=bpm,
                temperature=temperature
            )
        )
        await sess.play()

    return buffer.read()

# ─── STREAMLIT UI ─────────────────────────────────────────────────────────────
st.title("🌙 Dream-to-Visual & Soundtrack")

dream_text = st.text_area("Describe your dream", height=150)

if st.button("Generate"):
    if not dream_text.strip():
        st.warning("Please enter your dream above.")
        st.stop()

    # Interpretation
    with st.spinner("🔮 Interpreting your dream..."):
        interp_resp = generate_with_retry(
            client.models.generate_content,
            model="gemini-2.0-flash-lite",
            contents=dream_text,
            config=types.GenerateContentConfig(response_modalities=["TEXT"])
        )
        interp = "\n\n".join(
            part.text for part in interp_resp.candidates[0].content.parts
            if part.text
        )

    st.subheader("🌟 Interpretation")
    st.write(interp)

    # Image Generation
    with st.spinner("🖼️ Generating image..."):
        img_resp = generate_with_retry(
            client.models.generate_content,
            model="gemini-2.0-flash-preview-image-generation",
            contents=dream_text,
            config=types.GenerateContentConfig(response_modalities=["IMAGE"])
        )
        img_data = next(
            part.inline_data.data
            for part in img_resp.candidates[0].content.parts
            if part.inline_data
        )
        img = Image.open(BytesIO(img_data))

    st.subheader("🖼️ Dream Visualization")
    st.image(img, use_column_width=True)

    # Music Generation
    with st.spinner("🎵 Generating soundtrack..."):
        wav_bytes = asyncio.run(
            generate_music_from_text(
                prompt=dream_text,
                bpm=85,
                temperature=1.0,
                duration=15
            )
        )

    st.subheader("🎧 Dream Soundtrack")
    st.audio(wav_bytes, format="audio/wav")

    st.success("All done! Scroll up to experience your dream in text, image, and sound.")