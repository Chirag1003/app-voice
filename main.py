import streamlit as st
import whisper
import sounddevice as sd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from TTS.api import TTS
import queue
import threading
import re
import torch
import contextlib
import sys
import os

# =============== SETUP ===============
@st.cache_resource(show_spinner=False)
def load_models():
    asr_model = whisper.load_model("tiny")
    llm_model_id = "google/gemma-2b-it"
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
    llm_model = AutoModelForCausalLM.from_pretrained(
        llm_model_id,
        torch_dtype=None,
        device_map="cpu"
    )
    llm_pipeline = pipeline(
        "text-generation",
        model=llm_model,
        tokenizer=llm_tokenizer,
        max_new_tokens=256,
        temperature=0.6,
        do_sample=True
    )
    tts = TTS("tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    return asr_model, llm_pipeline, tts

asr_model, llm_pipeline, tts = load_models()

# =============== SESSION STATE ===============
if "history" not in st.session_state:
    st.session_state["history"] = []
if "is_paused" not in st.session_state:
    st.session_state["is_paused"] = False
if "assistant_thread" not in st.session_state:
    st.session_state["assistant_thread"] = None

# =============== AUDIO RECORDING ===============
def record_audio(duration=5, fs=16000):
    st.info("Recording for {} seconds...".format(duration))
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    audio = np.squeeze(recording)
    return audio

def transcribe(audio):
    with st.spinner("Transcribing..."):
        result = asr_model.transcribe(audio, language='en', fp16=False)
    return result["text"]

# =============== LLM REPLY ===============
def get_llm_reply(prompt, history):
    conv = ""
    for turn in history[-3:]:
        conv += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
    conv += f"User: {prompt}\nAssistant:"
    gen = llm_pipeline(conv)[0]['generated_text']
    reply = gen[len(conv):].strip()
    return reply

# =============== TTS PLAYBACK ===============
def split_sentences(text):
    return re.findall(r'[^.!?]+[.!?]', text) or [text]

def speak_stream(text, pause_event):
    sentences = split_sentences(text)
    for sentence in sentences:
        if pause_event.is_set():
            break
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            tts.tts_to_file(text=sentence, file_path="reply.wav")
        data, samplerate = sf.read("reply.wav", dtype='float32')
        sd.play(data, samplerate)
        sd.wait()
        os.remove("reply.wav")

# =============== SIDEBAR (CHAT HISTORY) ===============
with st.sidebar:
    st.title("🗂️ Conversation History")
    for idx, turn in enumerate(st.session_state.history):
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Assistant:** {turn['assistant']}")
    if st.button("Clear history"):
        st.session_state.history = []

# =============== MAIN UI ===============
st.title("🎙️ AI Voice Assistant (ChatGPT Style)")

st.write("#### How to use:")
st.markdown("""
- Press **Record** to capture your voice, or type your prompt below.
- Pause/resume voice output anytime.
- All chat history appears on the sidebar.
""")

# =============== INPUT ROW ===============
col1, col2, col3 = st.columns([1, 5, 1])
record_btn = col1.button("🎤 Record", key="rec_btn")
user_input = col2.text_input("Type your prompt here...", key="prompt")
send_btn = col3.button("Send", key="send_btn")

# PAUSE BUTTON
pause_btn = st.button("⏸️ Pause Speech", key="pause_btn")

if pause_btn:
    st.session_state.is_paused = True

# =============== HANDLE CHAT / AUDIO ===============
def run_assistant(prompt):
    reply = get_llm_reply(prompt, st.session_state.history)
    st.session_state.history.append({'user': prompt, 'assistant': reply})
    st.rerun()  # Show immediately
    # Speak (can be interrupted with pause)
    pause_event = threading.Event()
    st.session_state.is_paused = False
    st.session_state.pause_event = pause_event
    speak_thread = threading.Thread(target=speak_stream, args=(reply, pause_event))
    speak_thread.start()
    st.session_state.assistant_thread = speak_thread

if record_btn:
    audio = record_audio()
    text = transcribe(audio)
    st.session_state.history.append({'user': text, 'assistant': "(processing...)"} )
    st.rerun()  # Show user voice input right away
    run_assistant(text)

if send_btn and user_input.strip():
    run_assistant(user_input)

# =============== PAUSE THREAD IF NEEDED ===============
if st.session_state.get("is_paused", False):
    if st.session_state.get("pause_event"):
        st.session_state["pause_event"].set()
        st.session_state["is_paused"] = False

# =============== DISPLAY CHAT ===============
st.markdown("---")
st.write("### 💬 Chat")
for turn in st.session_state.history[::-1]:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Assistant:** {turn['assistant']}")

