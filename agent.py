import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import os
import json
import aiohttp
import base64
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from linebot import LineBotApi
import traceback

# === Secure Groq Setup - Llama 4 Scout (Vision Capable) ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set!")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.75,   # Slightly higher for more fun personality
    timeout=25,
    max_retries=2,
)

# === OpenAI Whisper ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set!")

# === Google Sheets (metadata) ===
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds_info = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key("1bDQuJTF-ene3Z8lXBKkFowwKKxAYcerpSRnbeFt38sg").sheet1

# === LINE ===
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not LINE_CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_ACCESS_TOKEN not set!")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)

# === Persistent Memory ===
USER_MEMORY_FILE = Path("/data/memory.json")        # 1:1 private
GROUP_MEMORY_FILE = Path("/data/group_memory.json") # Shared group history

user_conversations: dict[str, list] = {}
user_profiles: dict[str, dict] = {}
group_conversations: dict[str, list] = {}

# Load / Save functions (same as your previous stable version - omitted for brevity, keep yours)

# Load memory
if MEMORY_FILE.exists():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        conversations = raw.get("conversations", {})
        user_profiles = raw.get("profiles", {})
        for uid, msgs in conversations.items():
            conversations[uid] = [
                {"role": "system", "content": m["content"]} if isinstance(m, dict) and m.get("role") == "system"
                else HumanMessage(content=m["content"]) if m.get("type") == "human"
                else AIMessage(content=m["content"]) if m.get("type") == "ai"
                else m
                for m in msgs
            ]
        print(f"Loaded memory for {len(conversations)} users")
    except Exception as e:
        print(f"Memory load failed: {e}")
        conversations = {}
        user_profiles = {}
else:
    print("No memory file — starting fresh")
    conversations = {}

def save_memory():
    try:
        serializable = {"conversations": {}, "profiles": user_profiles}
        for uid, hist in conversations.items():
            serializable["conversations"][uid] = [
                {"type": "human", "content": m.content} if isinstance(m, HumanMessage)
                else {"type": "ai", "content": m.content} if isinstance(m, AIMessage)
                else m
                for m in hist
            ]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Saved memory for {len(conversations)} users")
    except Exception as e:
        print(f"Memory save failed: {e}")

async def analyze_image(message_id: str) -> str:
    try:
        url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}) as resp:
                if resp.status != 200:
                    return "無法下載照片～"
                image_bytes = await resp.read()
                base64_image = base64.b64encode(image_bytes).decode('utf-8')

        messages = [
            {"role": "system", "content": "你是一位溫暖、幽默、像臺灣阿姨一樣的記憶守護者。請用溫馨有趣的方式描述照片，並邀請用戶分享這張照片背後的故事。"},
            {"role": "user", "content": [
                {"type": "text", "text": "請幫我看看這張照片，用可愛的方式描述它，並問用戶這張照片對他們有什麼特別的回憶或故事。我想幫他們把照片和故事一起記錄下來。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]}
        ]

        response = await llm.ainvoke(messages)
        return response.content

    except Exception as e:
        print(f"Image analysis error: {e}")
        return "我看到照片了！這張照片看起來很有故事～你願意告訴我這張照片背後的回憶嗎？我會好好幫你記錄下來哦～"

async def get_agent_response(user_message: str, user_id: str, is_voice: bool = False, message_id: str = None, group_id: str = None, is_image: bool = False) -> str:
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    msg_lower = user_message.lower()

    # === Phase 1: Image Analysis ===
    if is_image and message_id:
        vision_desc = await analyze_image(message_id)
        user_message = f"[Photo uploaded] {vision_desc}"

    # === Phase 2: Fun Group Commands ===
    is_group = group_id is not None
    bot_mentioned = False
    if is_group and user_message:
        bot_name = line_bot_api.get_bot_info().display_name or "Echo"
        bot_mentioned = f"@{bot_name}" in user_message or f"@{bot_name.lower()}" in user_message.lower()

    if is_group and bot_mentioned:
        if any(x in msg_lower for x in ["總結", "summary", " recap"]):
            return "好的！讓我幫大家總結最近的聊天～（開發中，很快就會有完整版！）"
        if any(x in msg_lower for x in ["throwback", "回憶", "以前", "舊照"]):
            return "來點美好的回憶吧！上次大家分享的照片裡...（開發中）"
        if any(x in msg_lower for x in ["遊戲", "game", "玩"]):
            return "要玩什麼遊戲呢？故事接龍？猜台灣小吃？還是『誰最像阿姨』？😆 告訴我你想玩哪一個！"
        if any(x in msg_lower for x in ["分享", "share this to group"]):
            return "好的！我會把你私下告訴我的故事用溫暖的方式分享到群組～確認要分享嗎？"

    # === Normal flow (your existing logic + charming personality) ===
    # ... (keep your join/help, initialization, re-engagement, reply logic, etc.)

    # For brevity in this response, the rest of the function is the same as your last stable version, 
    # but with the new image handling and group command block inserted.

    return "我聽到了～讓我幫你記錄下來！有什麼想分享的嗎？❤️"
