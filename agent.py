import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import os
import json
import aiohttp
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from linebot import LineBotApi

# === Secure Groq Setup ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable not set!")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    timeout=10,
    max_retries=1,
)

# === OpenAI API Key (for Whisper API) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

# === Secure Google Sheets Setup ===
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds_json = os.getenv("GOOGLE_CREDENTIALS")
if not creds_json:
    raise ValueError("GOOGLE_CREDENTIALS environment variable not set!")
creds_info = json.loads(creds_json)
creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
client = gspread.authorize(creds)

SHEET_ID = "1bDQuJTF-ene3Z8lXBKkFowwKKxAYcerpSRnbeFt38sg"
sheet = client.open_by_key(SHEET_ID).sheet1

# === LINE Bot API for Profile Fetching ===
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN environment variable not set!")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)

# === Persistent Memory on /data Disk ===
MEMORY_FILE = Path("/data/memory.json")

# User profile tracking
user_profiles: dict[str, dict] = {}

# Load memory from disk on startup
if MEMORY_FILE.exists():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        conversations = raw_data.get("conversations", {})
        user_profiles = raw_data.get("profiles", {})
        # Convert raw messages back to LangChain objects
        for user_id in conversations:
            conversations[user_id] = [
                {"role": "system", "content": msg["content"]} if isinstance(msg, dict) and msg.get("role") == "system"
                else HumanMessage(content=msg["content"]) if msg.get("type") == "human"
                else AIMessage(content=msg["content"]) if msg.get("type") == "ai"
                else msg
                for msg in conversations[user_id]
            ]
        print(f"Loaded persistent memory for {len(conversations)} users from disk")
    except Exception as e:
        print(f"Failed to load memory from disk (starting fresh): {e}")
        conversations = {}
        user_profiles = {}
else:
    print("No memory file found — starting fresh")
    conversations = {}
    user_profiles = {}

def save_memory():
    """Save conversations and user profiles to disk"""
    try:
        serializable = {
            "conversations": {},
            "profiles": user_profiles
        }
        for user_id, history in conversations.items():
            serializable["conversations"][user_id] = []
            for msg in history:
                if isinstance(msg, HumanMessage):
                    serializable["conversations"][user_id].append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    serializable["conversations"][user_id].append({"type": "ai", "content": msg.content})
                elif isinstance(msg, dict) and msg.get("role") == "system":
                    serializable["conversations"][user_id].append(msg)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Saved persistent memory for {len(conversations)} users to disk")
    except Exception as e:
        print("Failed to save memory to disk:", str(e))

async def transcribe_audio(message_id: str) -> str:
    """Download LINE voice message and transcribe using OpenAI Whisper API"""
    try:
        print(f"DEBUG: Starting OpenAI Whisper transcription for message_id: {message_id}")
        audio_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                audio_url,
                headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}
            ) as resp:
                print(f"DEBUG: LINE download status: {resp.status}")
                if resp.status != 200:
                    return "無法下載語音訊息，請稍後再試。"
                audio_data = await resp.read()
                print(f"DEBUG: Audio downloaded, size: {len(audio_data)} bytes")

        MAX_AUDIO_SIZE = 30 * 1024 * 1024  # 30 MB ≈ 10–12 minutes
        if len(audio_data) > MAX_AUDIO_SIZE:
            return "語音太長了（超過10分鐘），請分段錄製或用文字分享，謝謝！"

        print("DEBUG: Sending to OpenAI Whisper API...")
        async with aiohttp.ClientSession() as session:
            form = aiohttp.FormData()
            form.add_field("file", audio_data, filename="voice.m4a", content_type="audio/m4a")
            form.add_field("model", "whisper-1")
            form.add_field("language", "zh")
            form.add_field("response_format", "text")

            async with session.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                data=form
            ) as resp:
                print(f"DEBUG: OpenAI response status: {resp.status}")
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"Whisper API error {resp.status}: {error_text}")
                    raise Exception(f"Whisper API error {resp.status}: {error_text}")
                text = await resp.text()
                print(f"DEBUG: Raw OpenAI response: {text[:200]}{'...' if len(text) > 200 else ''}")

        return text.strip() if text else "語音內容空白，請再試一次。"

    except Exception as e:
        import traceback
        print("Whisper error:", str(e))
        traceback.print_exc()
        return "語音轉文字失敗，請用文字分享或再試一次。"

# ... (the rest of your code remains unchanged: save_memory, get_agent_response, etc.)
