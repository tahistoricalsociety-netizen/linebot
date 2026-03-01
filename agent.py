import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import os
import json
import aiohttp
import base64
import random
from pathlib import Path
from langchain_xai import ChatXAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from linebot import LineBotApi
import traceback

# === Secure xAI Grok 4 Reasoning Setup ===
XAI_API_KEY = os.getenv("XAI_API_KEY")
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not set!")

llm = ChatXAI(
    xai_api_key=XAI_API_KEY,
    model="grok-4",
    temperature=0.70,
    max_tokens=4096,
)

# === OpenAI Whisper for Voice Transcription ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set!")

# === Google Sheets Setup ===
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds_info = json.loads(os.getenv("GOOGLE_CREDENTIALS"))
creds = Credentials.from_service_account_info(creds_info, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key("1bDQuJTF-ene3Z8lXBKkFowwKKxAYcerpSRnbeFt38sg").sheet1

# === LINE Bot API ===
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

# Load user memory
if USER_MEMORY_FILE.exists():
    try:
        with open(USER_MEMORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        user_conversations = raw.get("conversations", {})
        user_profiles = raw.get("profiles", {})
        print(f"Loaded user memory for {len(user_conversations)} users")
    except Exception as e:
        print(f"User memory load failed: {e}")

# Load group memory
if GROUP_MEMORY_FILE.exists():
    try:
        with open(GROUP_MEMORY_FILE, "r", encoding="utf-8") as f:
            group_conversations = json.load(f)
        print(f"Loaded group memory for {len(group_conversations)} groups")
    except Exception as e:
        print(f"Group memory load failed: {e}")

def save_user_memory():
    try:
        serializable = {"conversations": {}, "profiles": user_profiles}
        for uid, hist in user_conversations.items():
            serializable["conversations"][uid] = [
                {"type": "human", "content": m.content} if isinstance(m, HumanMessage)
                else {"type": "ai", "content": m.content} if isinstance(m, AIMessage)
                else m
                for m in hist
            ]
        with open(USER_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Saved user memory for {len(user_conversations)} users")
    except Exception as e:
        print(f"User memory save failed: {e}")

def save_group_memory():
    try:
        with open(GROUP_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(group_conversations, f, ensure_ascii=False, indent=2)
        print(f"Saved group memory for {len(group_conversations)} groups")
    except Exception as e:
        print(f"Group memory save failed: {e}")

async def transcribe_audio(message_id: str) -> str:
    try:
        audio_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
        async with aiohttp.ClientSession() as session:
            async with session.get(audio_url, headers={"Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"}) as resp:
                if resp.status != 200:
                    return "無法下載語音訊息，請稍後再試。"
                audio_data = await resp.read()
        if len(audio_data) > 30 * 1024 * 1024:
            return "語音太長了（超過10分鐘），請分段錄製或用文字分享，謝謝！"
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
                if resp.status != 200:
                    error = await resp.text()
                    print(f"Whisper error {resp.status}: {error}")
                    raise Exception(f"Whisper error {resp.status}: {error}")
                text = await resp.text()
        return text.strip() or "語音內容空白，請再試一次。"
    except Exception as e:
        print("Whisper error:", str(e))
        traceback.print_exc()
        return "語音轉文字失敗，請用文字分享或再試一次。"

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
        return "我看到照片了！這張照片看起來很有故事～"

async def get_agent_response(user_message: str, user_id: str, is_voice: bool = False, message_id: str = None, group_id: str = None, group_name: str = None, is_image: bool = False, is_private_dm: bool = False) -> str:
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    msg_lower = user_message.lower()

    # === PRIVATE DM FOR GROUP PHOTO (NEW LOGIC) ===
    if is_image and is_private_dm and group_name:
        return (
            f"我剛在「{group_name}」看到您分享的照片了～\n\n"
            f"這張照片看起來很有故事！\n"
            f"您願意告訴我這張照片背後的回憶嗎？\n\n"
            "如果這張照片有歷史價值（家族、臺灣、美國移民相關等），我們很希望能幫您永久存檔到 TAHS 檔案庫。\n"
            "最簡單的方式：請把照片寄到 tahistoricalsociety@gmail.com，郵件主旨寫上您的 LINE ID（例如：您的LINEID - 家族照片），我們會立刻處理並連結到您的故事。\n\n"
            "謝謝您讓我看到這張珍貴的照片！期待您的分享～❤️"
        )

    # Image Analysis (for 1:1 or @mentioned in group)
    if is_image and message_id:
        vision_desc = await analyze_image(message_id)
        user_message = f"[Photo uploaded] {vision_desc}"

    # Voice transcription
    if is_voice:
        transcribed = await transcribe_audio(message_id)
        user_message = f"[Voice message transcribed]: {transcribed}"
        print(f"DEBUG: Voice transcribed → {user_message}")

    # Initialize conversation
    if user_id not in user_conversations:
        user_conversations[user_id] = [{
            "role": "system",
            "content": """
You are Echo (歲月有聲), the official AI Historiographer for the Taiwanese American Historical Society (TAHS / 台美人歷史協會). 

You run on the LINE messaging platform using Grok 4 reasoning by xAI.

Your sole purpose is to collect, preserve, and record the personal stories of Taiwanese Americans and their families’ connections to Taiwan and the United States.

Core Identity (Always State This Clearly When Asked):
- You are the official TAHS Historiographer on LINE, powered by Grok 4 reasoning.
- In groups you stay silent unless @mentioned.
- In private 1:1 chats you are fully available for deep story collection.
- You can analyze photos sent to you and ask for the stories behind them.
- You can tell gentle jokes and poke people playfully based on group history.

When asked "who are you?" or "what can you do?", always reply with:
“我是臺灣美國歷史學會（TAHS）的官方AI故事夥伴 Echo（歲月有聲）。我在 LINE 上運作，使用 Grok 4 reasoning 模型，主要幫助大家保存家族故事與回憶。在群組裡我會保持安靜，只有被 @Echo 提到時才會回應。我可以分析照片、講溫馨笑話、回顧群組回憶，並在私訊中深度記錄您的個人故事。”

Critical Rules:
- Never invent or guess information not shared by the user.
- Never offer to search or look up external information.
- Always state your knowledge limitation when asked about facts or capabilities: 
  "我的知識來自 Grok 4 訓練資料，沒有即時資訊或外部搜尋功能。我的角色是收集和保存您的個人故事，而不是提供或驗證歷史事實。"
- Always redirect to collecting the user's own stories.

Conversation Flow Guidelines
- Start gently: First few messages — ask simple, low-pressure questions.
- Build gradually: Once sharing freely, ask one thoughtful, open-ended question at a time.
- Support stories: When user continues across messages, respond with warm encouragement without redirecting.
- Responses: Keep to 1–3 sentences, warm, natural, deeply appreciative.
- Introduction: Only in first message — introduce yourself and TAHS mission.
- Language: Match user’s language (Traditional Chinese default; switch to English if requested and stay there).
- Tone: Calm, respectful, caring — like a trusted friend and archivist honoring memories.

Group Chat Behavior
- In LINE groups: Stay completely silent unless directly @mentioned.
- If @mentioned: Reply in the group only for that message.
- Non-@mentioned messages: Reply privately (1:1) only if user friended you.
- Ignore spam, ads, stickers, locations, non-text/voice content.

Voice & Transcription
- Transcribe voice with OpenAI Whisper (cloud-based).
- Acknowledge warmly, show transcription clearly.
- If fail or too long: Politely guide to retry shorter or use text.

Photos & Documents
- LINE cannot permanently save media/files.
- When user sends photo in group without @Echo: Send private DM only.
- In private DM for photo: Reference the group, ask for story, and offer archiving.

Re-engagement After Inactivity
- Acknowledge time passed warmly.
- Reference specific past details from memory.
- Personalize naturally based on real history.

Memory & Tone
- Always reference shared details naturally.
- Never repeat or summarize unless asked.
- Calm, respectful, caring tone.
"""
        }]
        user_profiles[user_id] = {
            "first_interaction": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_message_time": current_time.isoformat(),
            "total_messages": 0,
            "language_preference": "繁體中文",
            "display_name": "Fetching...",
            "username": "",
            "picture_url": "",
            "last_followup_time": None
        }

    history = user_conversations[user_id]

    user_profiles[user_id]["last_message_time"] = current_time.isoformat()
    user_profiles[user_id]["total_messages"] = user_profiles[user_id].get("total_messages", 0) + 1

    if user_profiles[user_id]["display_name"] == "Fetching...":
        try:
            profile = line_bot_api.get_profile(user_id)
            user_profiles[user_id].update({
                "display_name": profile.display_name,
                "username": getattr(profile, "username", ""),
                "picture_url": profile.picture_url or ""
            })
        except Exception as e:
            print("Failed to fetch LINE profile:", str(e))
            user_profiles[user_id].update({
                "display_name": "Unknown",
                "username": "",
                "picture_url": ""
            })

    # Re-engagement
    last_time_str = user_profiles[user_id].get("last_message_time")
    reengage_prefix = ""
    if last_time_str:
        try:
            last_time = datetime.fromisoformat(last_time_str)
            time_diff = current_time - last_time
            if time_diff > timedelta(days=30):
                reengage_prefix = f"已經一個多月沒聽到您的故事了！上次您提到"
            elif time_diff > timedelta(days=7):
                reengage_prefix = f"已經一星期多了——我還在想您上次分享的"
            elif time_diff > timedelta(days=2):
                reengage_prefix = f"歡迎回來！已經幾天沒聽到您的故事了，"
            if reengage_prefix:
                user_message = f"[User returning after {time_diff.days} days] {reengage_prefix} {user_message}"
        except:
            pass

    history.append(HumanMessage(content=user_message))

    # Group story detection & private follow-up
    if group_id and not is_voice:
        story_keywords = ["故事", "家族", "回憶", "過去", "臺灣", "美國", "移民", "經歷", "小時候", "爺爺", "奶奶", "爸爸", "媽媽", "老家", "童年", "歷史", "分享", "講", "說", "以前", "當年"]
        matching = [kw for kw in story_keywords if kw in msg_lower]
        is_story_like = len(matching) >= 2 and len(msg_lower) > 50

        if is_story_like:
            last_followup = user_profiles[user_id].get("last_followup_time")
            can_followup = not last_followup or (current_time - datetime.fromisoformat(last_followup)) > timedelta(minutes=5)

            if can_followup:
                try:
                    line_bot_api.push_message(
                        user_id,
                        TextSendMessage(text=(
                            "謝謝您在群組分享的故事片段！聽起來很有意義～\n"
                            "如果方便的話，能否私下多告訴我一些細節？例如當時的心情、周圍環境，或您家人的反應？\n"
                            "我會用心記錄，幫助您把故事完整保存下來。期待您的分享！"
                        ))
                    )
                    user_profiles[user_id]["last_followup_time"] = current_time.isoformat()
                    print(f"DEBUG: Sent private story follow-up DM to {user_id}")
                except Exception as e:
                    print(f"Private DM failed (likely not friended): {e}")

    # Normal LLM reply
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("history")])
    chain = prompt | llm

    try:
        response = await asyncio.wait_for(chain.ainvoke({"history": history}), timeout=25.0)
        bot_reply = response.content or "我在這裡傾聽您的故事。如果有什麼想分享的，請繼續告訴我，好嗎？"
        history.append(AIMessage(content=bot_reply))

        profile = user_profiles[user_id]
        row_data = [
            timestamp,
            user_id,
            "User" if not is_voice else "Voice (transcribed)",
            user_message,
            "[Voice]" if is_voice else "",
            profile.get("display_name", "Unknown"),
            profile.get("username", ""),
            profile.get("picture_url", ""),
            profile.get("first_interaction", ""),
            profile.get("total_messages", 0),
            profile.get("language_preference", "繁體中文"),
            group_id or ""
        ]
        bot_row_data = row_data.copy()
        bot_row_data[2] = "Bot"
        bot_row_data[3] = bot_reply
        bot_row_data[4] = "TAHS Interview"

        sheet.append_row(row_data)
        sheet.append_row(bot_row_data)
        print("DEBUG: Logged to Sheets")

        save_user_memory()
        return bot_reply

    except asyncio.TimeoutError:
        timeout_reply = "感謝您的耐心等待——我在這裡。請繼續分享您的故事。"
        history.append(AIMessage(content=timeout_reply))
        save_user_memory()
        return timeout_reply

    except Exception as e:
        print("Agent error:", str(e))
        traceback.print_exc()
        fallback = "我在傾聽。請隨時分享您的故事。"
        history.append(AIMessage(content=fallback))
        save_user_memory()
        return fallback
