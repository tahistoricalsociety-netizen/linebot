import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import os
import json
import aiohttp
import base64
import random
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from linebot import LineBotApi
import traceback

# === Secure Groq Setup - Llama 4 Scout ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set!")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.78,
    timeout=25,
    max_retries=2,
)

# === OpenAI Whisper ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set!")

# === Google Sheets ===
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
USER_MEMORY_FILE = Path("/data/memory.json")
GROUP_MEMORY_FILE = Path("/data/group_memory.json")

user_conversations: dict[str, list] = {}
user_profiles: dict[str, dict] = {}
group_conversations: dict[str, list] = {}

# Load memory (keep your existing load/save functions here - omitted for brevity)

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

async def generate_group_joke(group_id: str) -> str:
    recent = group_conversations.get(group_id, [])[-15:]
    if not recent:
        return "群組還太新，沒有足夠的回憶可以開玩笑呢～下次再來！😊"
    
    # Simple humorous respectful tease
    jokes = [
        "最近有人在群組裡一直說要減肥，結果昨天又偷偷叫了三杯手搖飲！是誰啊～開玩笑的，大家都很可愛啦！",
        "有人每次說要早睡，結果凌晨三點還在傳訊息～我都看到了哦～😆",
        "這個群組的聊天記錄顯示：有人超會聊天，但一到分享故事就害羞！是誰呢～"
    ]
    return random.choice(jokes) + "\n有什麼想分享的回憶嗎？"

async def generate_group_poke(group_id: str) -> str:
    recent = group_conversations.get(group_id, [])[-15:]
    if not recent:
        return "來 poke 一下～大家最近都好安靜哦，是不是在偷偷準備驚喜？快告訴我！"
    
    pokes = [
        "哎呀～有人最近很活躍，但一到分享故事就害羞！是誰呢～😏 來，勇敢一點！",
        "我看到有人在群組裡一直偷笑～快說，是不是有什麼好玩的事？",
        "poke poke～有人最近很安靜，是不是在想心事？來分享一下嘛～"
    ]
    return random.choice(pokes)

async def get_agent_response(user_message: str, user_id: str, is_voice: bool = False, message_id: str = None, group_id: str = None, is_image: bool = False) -> str:
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    msg_lower = user_message.lower()

    # Image Analysis
    if is_image and message_id:
        vision_desc = await analyze_image(message_id)
        user_message = f"[Photo uploaded] {vision_desc}"

    is_group = group_id is not None
    bot_mentioned = False
    if is_group and user_message:
        bot_name = line_bot_api.get_bot_info().display_name or "Echo"
        bot_mentioned = f"@{bot_name}" in user_message or f"@{bot_name.lower()}" in user_message.lower()

    # Fun Group Commands
    if is_group and bot_mentioned:
        if "joke" in msg_lower:
            return await generate_group_joke(group_id)
        if "poke" in msg_lower:
            return await generate_group_poke(group_id)
        if any(x in msg_lower for x in ["總結", "summary", "recap"]):
            return "好的！讓我幫大家總結最近的聊天～（開發中）"
        if any(x in msg_lower for x in ["throwback", "回憶", "以前"]):
            return "來點美好的回憶吧！上次大家分享的照片裡...（開發中）"
        if any(x in msg_lower for x in ["遊戲", "game", "玩"]):
            return "要玩什麼遊戲呢？故事接龍？猜台灣小吃？還是『誰最像阿姨』？😆 告訴我你想玩哪一個！"

    # Normal flow (your existing logic remains here)
    # For brevity, the rest of the function is your previous stable version with the above integrations.

    return "我聽到了～讓我幫你記錄下來！有什麼想分享的嗎？❤️"
