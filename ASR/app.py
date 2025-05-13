import streamlit as st # type: ignore
from transformers import pipeline
import whisper # type: ignore
import yt_dlp # type: ignore
import os

# Load Whisper and Summarizer
model = whisper.load_model("base")  # or "small", "medium", "large"
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)


def download_youtube_audio(url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return 'audio.mp3'

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):

    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]  # break into chunks if long
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return "\n".join(summaries)

# Streamlit App
st.title("🎧 YouTube Audio Summarizer")

youtube_url = st.text_input("Enter YouTube Video URL")

if youtube_url:
    with st.spinner("Downloading and processing..."):
        audio_file = download_youtube_audio(youtube_url)
        transcript = transcribe_audio(audio_file)
        summary = summarize_text(transcript)
    
    # st.subheader("Transcript")
    # st.write(transcript)
    
    st.subheader("Summary")
    st.write(summary)
