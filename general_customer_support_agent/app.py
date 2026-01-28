# ---------------------------------------------
# Imports & Setup
# ---------------------------------------------
import streamlit as st
import os, time, json, base64, wave, asyncio
import speech_recognition as sr
import boto3
import edge_tts
from pydub import AudioSegment
from pydub.playback import play
from datetime import datetime
import shutil

# Directories
AUDIO_DIR = "audios"
MEMORY_DIR = "memory"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(MEMORY_DIR, exist_ok=True)

# AWS + Bedrock (LLaMA 3 70B Instruct)
AWS_REGION = "ap-south-1"
LLAMA_3_MODEL_ID = "meta.llama3-70b-instruct-v1:0"
session = boto3.Session(region_name=AWS_REGION)
bedrock = session.client("bedrock-runtime")

# ---------------------------------------------
# UI Setup
# ---------------------------------------------
st.set_page_config(page_title="Customer Support Agent")
st.title("🎧 Voice Assistant for Customer Support")

lang_map = {
    "English (India)": {
        "voice": "en-IN-NeerjaNeural",
        "stt": "en-IN",
        "greeting": "Hi! Thank you for calling us, How can I assist you today?",
        "goodbye": "Thank you for calling us, we hope to assist you in the future! <END_CALL>"
    },
    "Hindi": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "hi-IN",
        "greeting": "नमस्ते! हमें कॉल करने के लिए धन्यवाद, मैं आज आपकी कैसे सहायता कर सकती हूँ?",
        "goodbye": "हमें कॉल करने के लिए धन्यवाद, हम भविष्य में आपकी सहायता करने की आशा करते हैं! <END_CALL>"
    },
    "Marathi": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "mr-IN",
        "greeting": "नमस्कार, आम्हाला कॉल केल्याबद्दल धन्यवाद! मी आज आपली कशी मदत करू?",
        "goodbye": "आम्हाला कॉल केल्याबद्दल धन्यवाद, आम्ही भविष्यातही आपली मदत करण्याची आशा करतो! <END_CALL>"
    },
    "Gujarati": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "gu-IN",
        "greeting": "નમસ્તે, અમને કોલ કરવા બદલ આભાર! આજે હું કેવી રીતે તમારી મદદ કરી શકું?",
        "goodbye": "અમને કોલ કરવા બદલ આભાર, ભવિષ્યમાં પણ તમારી સહાય કરવાની આશા રાખીએ છીએ! <END_CALL>"
    },
    "Bengali": {
        "voice": "bn-IN-TanishaaNeural",
        "stt": "bn-IN",
        "greeting": "নমস্কার, আমাদের কল করার জন্য ধন্যবাদ! আজ আমি আপনাকে কীভাবে সাহায্য করতে পারি?",
        "goodbye": "আমাদের কল করার জন্য ধন্যবাদ, ভবিষ্যতে আপনাকে সাহায্য করতে পারব বলে আশা করি! <END_CALL>"
    },
    "Tamil": {
        "voice": "ta-IN-PallaviNeural",
        "stt": "ta-IN",
        "greeting": "வணக்கம், எங்களை அழைத்ததற்கு நன்றி! இன்று நான் உங்களுக்கு எப்படி உதவலாம்?",
        "goodbye": "எங்களை அழைத்ததற்கு நன்றி, எதிர்காலத்தில் உங்களுக்கு உதவ விரும்புகிறோம்! <END_CALL>"
    },
    "Telugu": {
        "voice": "te-IN-ShrutiNeural",
        "stt": "te-IN",
        "greeting": "నమస్తే, మమ్మల్ని కాల్ చేసినందుకు ధన్యవాదాలు! నేను ఈ రోజు మీకు ఎలా సహాయం చేయగలను?",
        "goodbye": "మమ్మల్ని కాల్ చేసినందుకు ధన్యవాదాలు, భవిష్యత్తులో మేము మిమ్మల్ని సహాయం చేయగలమని ఆశిస్తున్నాము! <END_CALL>"
    },
    "Punjabi": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "pa-IN",
        "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਸਾਨੂੰ ਕਾਲ ਕਰਨ ਲਈ ਧੰਨਵਾਦ! ਅੱਜ ਮੈਂ ਤੁਹਾਡੀ ਕਿਵੇਂ ਮਦਦ ਕਰ ਸਕਦਾ ਹਾਂ?",
        "goodbye": "ਸਾਨੂੰ ਕਾਲ ਕਰਨ ਲਈ ਧੰਨਵਾਦ, ਅਸੀਂ ਭਵਿੱਖ ਵਿੱਚ ਵੀ ਤੁਹਾਡੀ ਮਦਦ ਕਰਨ ਦੀ ਉਮੀਦ ਰੱਖਦੇ ਹਾਂ! <END_CALL>"
    },
    "Odia": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "or-IN",
        "greeting": "ନମସ୍କାର, ଆମକୁ କଲ୍ କରିବାକୁ ଧନ୍ୟବାଦ! ଆଜି ମୁଁ ଆପଣଙ୍କୁ କିପରି ସହଯୋଗ କରିପାରିବି?",
        "goodbye": "ଆମକୁ କଲ୍ କରିଥିବା ପାଇଁ ଧନ୍ୟବାଦ, ଆମେ ଆଗାମୀ ଦିନରେ ଆପଣଙ୍କୁ ସହଯୋଗ କରିବାକୁ ଆଶା କରୁଛୁ! <END_CALL>"
    }
}

lang = st.selectbox("Choose Language", list(lang_map.keys()))
lang_data = lang_map[lang]
voice_id = lang_data["voice"]
stt_lang = lang_data["stt"]

# ---------------------------------------------
# Session State
# ---------------------------------------------
if "chat" not in st.session_state: st.session_state.chat = []
if "phase" not in st.session_state: st.session_state.phase = "init"
if "last_msg_hash" not in st.session_state: st.session_state.last_msg_hash = None

# ---------------------------------------------
# LLM (Native Format for LLaMA 3)
# ---------------------------------------------
def get_llm_response(history):
    system_prompt = """You are a polite, multilingual customer support agent. Your job is to 
-assist customer when they call you, provide appropriate solutions to any and all queries, answer generally what you think might help them
-do sentiment analysis and respond accordingly, always be empathetic.
The user will select the language(before starting the call), you have to respond in that language only.
If english language is selected(default) continue in that, do not ask once again to change languages.
Do not change to default messages or languages, continue the same language and be empathetic throughout.
If needed or if they seem unsatisfied, offer to connect them to a human.
Keep the replies short and natural(about 50 words).
Always thank the customer and update the records.
Once the call is concluded and you have thanked the customer, end the call."""

    formatted = "<|begin_of_text|>"
    formatted += f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>\n"

    for msg in history:
        role = "user" if msg["role"] == "user" else "assistant"
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n{msg['content']}<|eot_id|>\n"

    formatted += "<|start_header_id|>assistant<|end_header_id|>\n"

    body = json.dumps({
        "prompt": formatted,
        "max_gen_len": 100,
        "temperature": 0.7
    })

    try:
        response = bedrock.invoke_model(body=body, modelId=LLAMA_3_MODEL_ID)
        output = json.loads(response["body"].read())
        return output["generation"].strip()
    except Exception:
        return "Sorry, something went wrong. <END_CALL>"

def summarize_chat(chat_history):
    formatted = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant. Summarize this call in a detailed story format for internal recordkeeping. Be natural and descriptive.<|eot_id|>\n"
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "assistant"
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n{msg['content']}<|eot_id|>\n"
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n"

    body = json.dumps({
        "prompt": formatted,
        "max_gen_len": 400,
        "temperature": 0.6
    })

    try:
        response = bedrock.invoke_model(body=body, modelId=LLAMA_3_MODEL_ID)
        output = json.loads(response["body"].read())
        return output["generation"].strip()
    except Exception:
        return "Summary generation failed."

# ---------------------------------------------
# Utilities
# ---------------------------------------------
def hash_message(content):
    return str(abs(hash(content.strip().lower())))

async def synthesize_audio(text, path, voice):
    mp3_path = path.replace(".wav", ".mp3")
    if not os.path.exists(path):
        tts = edge_tts.Communicate(text.split("<")[0], voice=voice, rate="+30%")
        await tts.save(mp3_path)
        AudioSegment.from_file(mp3_path, format="mp3").export(path, format="wav")

def play_audio_blocking(path):
    audio = AudioSegment.from_wav(path)
    play(audio)

def record_user_voice():
    rec = sr.Recognizer()
    with sr.Microphone() as source:
        rec.adjust_for_ambient_noise(source)
        st.toast("🎧 Listening...")
        audio = rec.listen(source, timeout=5, phrase_time_limit=8)
        return rec.recognize_google(audio, language=stt_lang)

def cleanup_audio():
    for file in os.listdir(AUDIO_DIR):
        os.remove(os.path.join(AUDIO_DIR, file))

# ---------------------------------------------
# Status Indicator
# ---------------------------------------------
status = {
    "speak": "🗣️ Speaking...",
    "listen": "🎧 Listening...",
    "process": "💬 Thinking...",
    "done": "✅ Call Ended."
}.get(st.session_state.phase, "")
if status: st.info(status)

# ---------------------------------------------
# Start Call
# ---------------------------------------------
if not st.session_state.chat:
    if st.button("Start Call 🎙️"):
        st.session_state.chat.append({"role": "assistant", "content": lang_data["greeting"]})
        st.session_state.phase = "speak"
        st.rerun()

# ---------------------------------------------
# Display Chat
# ---------------------------------------------
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].split("<")[0])

# ---------------------------------------------
# End Call Manually
# ---------------------------------------------
if st.session_state.phase not in ["init", "done"]:
    if st.button("❌ End Call"):
        st.session_state.phase = "done"

        summary = summarize_chat(st.session_state.chat)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"{MEMORY_DIR}/summary_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        with open(f"{MEMORY_DIR}/summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat, f, ensure_ascii=False, indent=2)

        cleanup_audio()
        st.rerun()

# ---------------------------------------------
# Assistant Speaks
# ---------------------------------------------
if st.session_state.phase == "speak":
    last = st.session_state.chat[-1]
    msg_hash = hash_message(last["content"])
    filename = os.path.join(AUDIO_DIR, f"{msg_hash}.wav")

    if st.session_state.last_msg_hash != msg_hash:
        asyncio.run(synthesize_audio(last["content"], filename, voice_id))
        st.session_state.last_msg_hash = msg_hash

    play_audio_blocking(filename)
    st.session_state.phase = "listen"
    st.rerun()

# ---------------------------------------------
# User Speaks
# ---------------------------------------------
if st.session_state.phase == "listen":
    try:
        user_input = record_user_voice()
    except:
        fallback = "Sorry, I couldn't hear you. Could you please repeat that?"
        st.session_state.chat.append({"role": "assistant", "content": fallback})
        st.session_state.phase = "speak"
        st.rerun()

    if user_input and user_input.strip():
        st.session_state.chat.append({"role": "user", "content": user_input})
        st.session_state.phase = "process"
    else:
        fallback = "I didn't catch that. Could you say it again?"
        st.session_state.chat.append({"role": "assistant", "content": fallback})
        st.session_state.phase = "speak"
    st.rerun()

# ---------------------------------------------
# LLM Processes
# ---------------------------------------------
if st.session_state.phase == "process":
    reply = get_llm_response(st.session_state.chat)
    st.session_state.chat.append({"role": "assistant", "content": reply})

    if "<END_CALL>" in reply:
        st.session_state.phase = "done"
        st.success("✅ Call Ended.")

        summary = summarize_chat(st.session_state.chat)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        with open(f"{MEMORY_DIR}/summary_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write(summary)
        with open(f"{MEMORY_DIR}/summary_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat, f, ensure_ascii=False, indent=2)

        cleanup_audio()
    else:
        st.session_state.phase = "speak"
    st.rerun()
