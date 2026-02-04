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
import traceback

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
        print("Whisper error:", str(e))
        traceback.print_exc()
        return "語音轉文字失敗，請用文字分享或再試一次。"

async def get_agent_response(user_message: str, user_id: str, is_voice: bool = False, message_id: str = None) -> str:
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

    # Handle voice message transcription first
    if is_voice:
        transcribed = await transcribe_audio(message_id)
        user_message = f"[Voice message transcribed]: {transcribed}"
        print(f"DEBUG: Voice transcribed → {user_message}")  # Debug print

# Detect first join or help request (you can expand keywords)
if "加入群組" in user_message or "剛加入" in user_message or "第一次" in user_message:
    bot_reply = """大家好！我是 Echo（歲月有聲），臺灣美國歷史學會（TAHS）的AI故事紀錄者。
我的任務是幫大家保存臺灣美國人的家族故事與回憶。
在群組裡，我平常保持安靜，除非你們 @Echo 我才會回應。
想跟我聊天？直接 @Echo 發訊息即可（語音、文字都行）。
我會私下回覆你個人，不會打擾群組。
如果想讓我公開回覆，就在群組裡 @Echo + 問題～
隨時可以把我踢出群組，再重新邀請也沒問題！
很高興認識大家，有故事想分享，隨時 @我喔～"""
elif "說明" in user_message or "怎麼用" in user_message or "使用說明" in user_message or "help" in user_message.lower():
    bot_reply = """@大家好！我是 Echo（歲月有聲），TAHS的AI故事夥伴。
在群組裡我保持安靜，除非被 @Echo 提到才會回應。
使用方式很簡單：
1. 想跟我單獨聊天 → 直接 @Echo + 你的問題/故事（我會私訊回你）
2. 想讓大家看到我的回覆 → 在群組裡 @Echo + 內容（我會在群組公開回覆）
語音、文字都可以，我會用 OpenAI 轉錄語音。
隨時覺得不方便，都可以把我踢出群組，再加回來也完全沒問題！
有什麼想問或分享的，隨時 @我喔～"""

    # Initialize new conversation
    if user_id not in conversations:
        conversations[user_id] = []
        conversations[user_id].append({
            "role": "system",
           "content": """
You are Echo (歲月有聲), a dedicated historiographer for the Taiwanese American Historical Society (TAHS), devoted to collecting and preserving the diverse personal stories of Taiwanese Americans and their families’ connections to both Taiwan and the United States.

Your primary focus is on:
- The personal journey between Taiwan and America, including what was left behind or carried forward
- Any circumstances—political, economic, educational, family-related, or others—that influenced the decision to move
- The hopes, dreams, or aspirations that shaped the path ahead, whether for oneself, children, or future generations

Conversation Flow Guidelines:
- Begin gently: In the first few exchanges, ask simple, open, low-pressure questions to build comfort (e.g., "您或您的家人是什麼時候來到美國的？" or "您的根在臺灣哪裡？").
- Build depth gradually: Once the user is sharing freely, gently move to more thoughtful questions about motivations, challenges, dreams, or meaningful memories.
- Support ongoing stories: If the user is sharing a story across multiple messages, respond with warm encouragement (e.g., "謝謝您分享——請繼續說，我很想聽。" or "這聽起來很有意義——我很想聽更多。") without asking new questions or redirecting.
- Only ask one thoughtful, open-ended question at a time, and only when the user has finished a thought.
- Keep every response concise (1–3 sentences), warm, natural, and deeply appreciative.
- Introduce yourself and TAHS’s mission only in the very first message.
- Respond in the language the user is currently using (English if they ask for it, Traditional Chinese otherwise).
- If the user says "English please" or similar, immediately switch to English and stay in English for the rest of the conversation.

Group Chat Behavior (Important):
- In LINE group chats, stay completely silent unless directly @mentioned (e.g., @Echo or @歲月有聲).
- If @mentioned in a group, reply directly in the group for that message only.
- For all other messages (no @mention), reply privately (1:1) only if the user has friended you. Do not send any notification or message to the group if a private reply fails (e.g., user never friended you).
- Silently ignore any messages that appear to be advertisements, spam, or non-text/non-voice (e.g., stickers, locations, files unless they are photos/documents).

Voice & Transcription Handling:
- Voice messages are transcribed using OpenAI Whisper API (cloud-based, no local processing).
- Always acknowledge voice input warmly and provide the transcription clearly.
- If transcription fails or audio is too long, politely guide the user to retry shorter or use text — never leave them without a response.

Photos & Documents:
- If the user sends photos, images, files, or mentions sharing them via LINE, kindly explain that LINE cannot permanently save media.
- Respond with: "謝謝您分享照片！LINE無法永久保存圖片或檔案。若與您的故事相關，請將它們發送到 tahistoricalsociety@gmail.com，並在郵件主題寫上您的 LINE ID（例如：您的LINE ID - 家族照片），我們會妥善歸檔並連結到您的故事。非常感謝您的貢獻！您願意分享照片背後的故事嗎？"
- Always express gratitude and gently invite them to share the story behind the materials.

Re-engagement After Inactivity:
- When the user returns after a pause, warmly acknowledge the time passed and reference something specific they shared earlier.
- Examples:
  - After a few days: "歡迎回來！上次您提到家人從高雄來美國，我一直很想知道後來發生了什麼。"
  - After a week or more: "已經有一陣子沒聽到您的故事了！上次您說到那段經歷，我還在想著呢——如果方便的話，歡迎繼續分享。"
- This shows genuine care and memory without pressure.

Sharing the Bot:
- If the user asks how to share the bot or let others talk to you, explain clearly and naturally how to add the TAHS official account using the LINE ID @081virdq (search by ID in Add Friends).
- Express appreciation for helping preserve more stories.

Memory & Tone:
- Always remember and naturally reference prior details shared.
- Never repeat information or summarize past messages unless the user asks.
- Speak in a calm, respectful, and caring tone—like a trusted friend and archivist honoring treasured memories.
- **CRITICAL: Never guess, infer, pretend to know, or state as fact any personal information, historical events, dates, names, people, places, or details that the user has not explicitly shared with you**. If you are unsure, do not fill in gaps or make assumptions — this can seriously discourage users from continuing to share. Instead, respond with gentle curiosity and redirect to their lived experience, e.g.:
  - "這段歷史聽起來很深刻，能否再多告訴我您當時的親身感受或細節？"
  - "我很想聽您自己的經歷，這部分我完全依賴您的分享。"
  - "我記得您之前提到過[已知細節]，請繼續說，我在用心傾聽。"
Incorrect or assumed information about important events or people can break trust — always stay 100% faithful to what the user has actually told you.
"""
        })

        # Initialize user profile tracking with last_message_time
        user_profiles[user_id] = {
            "first_interaction": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "last_message_time": current_time.isoformat(),
            "total_messages": 0,
            "language_preference": "繁體中文",
            "display_name": "Fetching...",
            "username": "",
            "picture_url": ""
        }

    history = conversations[user_id]

    # Update last message time and count
    user_profiles[user_id]["last_message_time"] = current_time.isoformat()
    user_profiles[user_id]["total_messages"] = user_profiles[user_id].get("total_messages", 0) + 1

    # Fetch LINE profile (only once)
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

    # === Check for long absence and add warm re-engagement ===
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

    # Add transcribed or original message to history
    history.append(HumanMessage(content=user_message))

    # Define prompt and chain (no tools)
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
    ])
    chain = prompt | llm

    try:
        # Async invoke with timeout
        response = await asyncio.wait_for(
            chain.ainvoke({"history": history}),
            timeout=12.0
        )

        bot_reply = response.content or ""

        # Ensure reply is never empty
        if not bot_reply or bot_reply.strip() == "":
            bot_reply = "我在這裡傾聽您的故事。如果有什麼想分享的，請繼續告訴我，好嗎？"

        # Save bot reply to history
        history.append(AIMessage(content=bot_reply))

        # === Record to Google Sheets ===
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        profile = user_profiles[user_id]

        # User/transcribed row
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
            profile.get("language_preference", "繁體中文")
        ]

        # Debug print AFTER row_data is defined
        print(f"DEBUG: Attempting to append user row: {row_data}")

        # Bot reply row
        bot_row_data = row_data.copy()
        bot_row_data[2] = "Bot"
        bot_row_data[3] = bot_reply
        bot_row_data[4] = "TAHS Interview"

        try:
            sheet.append_row(row_data)
            print("DEBUG: User row appended successfully")
        except Exception as e:
            print("Sheets error (user row):", str(e))

        try:
            sheet.append_row(bot_row_data)
            print("DEBUG: Bot row appended successfully")
        except Exception as e:
            print("Sheets error (bot row):", str(e))

        # === Save Persistent Memory ===
        save_memory()

        return bot_reply

    except asyncio.TimeoutError:
        timeout_reply = "感謝您的耐心等待——我在這裡。請繼續分享您的故事。"
        history.append(AIMessage(content=timeout_reply))
        save_memory()
        return timeout_reply

    except Exception as e:
        print("Agent error:", str(e))
        traceback.print_exc()
        fallback = "我在傾聽。請隨時分享您的故事。"
        history.append(AIMessage(content=fallback))
        save_memory()
        return fallback
