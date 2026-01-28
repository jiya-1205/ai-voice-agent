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
st.set_page_config(page_title="Eva — Your Voice Assistant")
st.title("🎧 Eva — Your Voice Assistant")

lang_map = {
    "English (India)": {
        "voice": "en-IN-NeerjaNeural",
        "stt": "en-IN",
        "greeting": "Hi! This call is for product feedback. May I know if you recently purchased one of our products?",
        "goodbye": "Thanks for your time. Goodbye! <END_CALL>"
    },
    "Hindi": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "hi-IN",
        "greeting": "नमस्ते! यह कॉल उत्पाद प्रतिक्रिया के लिए है। क्या आपने हाल ही में हमारा कोई उत्पाद खरीदा है?",
        "goodbye": "आपके समय के लिए धन्यवाद। अलविदा! <END_CALL>"
    },
    "Marathi": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "mr-IN",
        "greeting": "नमस्कार! हा कॉल उत्पादन अभिप्रायासाठी आहे. तुम्ही अलीकडे आमचं काही उत्पादन विकत घेतलं का?",
        "goodbye": "तुमच्या वेळेसाठी धन्यवाद. गुडबाय! <END_CALL>"
    },
    "Gujarati": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "gu-IN",
        "greeting": "નમસ્તે! આ કોલ પ્રોડક્ટ પ્રતિસાદ માટે છે. શું તમે તાજેતરમાં નું કોઈ પ્રોડક્ટ ખરીદ્યું છે?",
        "goodbye": "તમારા સમય માટે આભાર. અલવિદા! <END_CALL>"
    },
    "Bengali": {
        "voice": "bn-IN-TanishaaNeural",
        "stt": "bn-IN",
        "greeting": "নমস্কার! এই কলটি পণ্যের প্রতিক্রিয়া নেওয়ার জন্য। আপনি কি সম্প্রতি আমাদের কোনো পণ্য কিনেছেন?",
        "goodbye": "আপনার সময়ের জন্য ধন্যবাদ। বিদায়! <END_CALL>"
    },
    "Tamil": {
        "voice": "ta-IN-PallaviNeural",
        "stt": "ta-IN",
        "greeting": "வணக்கம்! இந்த அழைப்பு தயாரிப்பு பின்னூட்டத்திற்காக. நீங்கள் சமீபத்தில் எங்கள் தயாரிப்புகளை வாங்கியிருக்கிறீர்களா?",
        "goodbye": "உங்கள் நேரத்திற்காக நன்றி. குட்பை! <END_CALL>"
    },
    "Telugu": {
        "voice": "te-IN-ShrutiNeural",
        "stt": "te-IN",
        "greeting": "నమస్తే! ఈ కాల్ ఉత్పత్తి అభిప్రాయం కోసం. మీరు ఇటీవల మా ఉత్పత్తిని కొనుగోలు చేశారా?",
        "goodbye": "మీ సమయానికి ధన్యవాదాలు. గుడ్ బై! <END_CALL>"
    },
    "Punjabi": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "pa-IN",
        "greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਇਹ ਕਾਲ ਉਤਪਾਦ ਪ੍ਰਤੀਕਿਰਿਆ ਲਈ ਹੈ। ਕੀ ਤੁਸੀਂ ਹਾਲ ਹੀ ਵਿੱਚ ਸਾਡਾ ਕੋਈ ਉਤਪਾਦ ਖਰੀਦਿਆ ਹੈ?",
        "goodbye": "ਤੁਹਾਡਾ ਸਮਾਂ ਦੇਣ ਲਈ ਧੰਨਵਾਦ। ਗੁੱਡਬਾਏ! <END_CALL>"
    },
    "Odia": {
        "voice": "hi-IN-SwaraNeural",
        "stt": "or-IN",
        "greeting": "ନମସ୍କାର! ଏହି କଲ୍‌ ପ୍ରୋଡକ୍ଟ ଫିଡ୍‌ବ୍ୟାକ୍ ପାଇଁ। ଆପଣ ସମ୍ପ୍ରତି କିଛି ଉତ୍ପାଦ କିଣିଛନ୍ତି କି?",
        "goodbye": "ଆପଣଙ୍କ ସମୟ ପାଇଁ ଧନ୍ୟବାଦ। ଗୁଡ୍‌ବାଏ! <END_CALL>"
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
    system_prompt = """You are a polite, multilingual customer service agent. 
Your job is to call customers for a product feedback, also to sentiment analysis and respond accordingly.
The user will select the language(before starting the call), you have to respond in that language only.
If english language is selected(default) continue in that, do not ask once again to change languages.
Do not change to default messages or languages, continue the same language and be empathetic throughout.
-Start by greeting them and asking if they recently bought anything,
if not, apologise and conclude the call.
-If they have, check if they’ve used it and ask about their experience,
 if they haven't used it yet, politely urge them to use it, and schedule a call for later.
-If they’re happy, ask for a review(on the call itself),
If not, ask what could be improved and listen carefully.
If needed, offer to connect them to a human.
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
