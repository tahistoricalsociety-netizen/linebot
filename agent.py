import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
import json
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import asyncio
from pathlib import Path
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
            for i, msg in enumerate(conversations[user_id]):
                if msg.get("type") == "human":
                    conversations[user_id][i] = HumanMessage(content=msg["content"])
                elif msg.get("type") == "ai":
                    conversations[user_id][i] = AIMessage(content=msg["content"])
        print(f"Loaded persistent memory for {len(conversations)} users from disk")
    except Exception as e:
        print(f"Failed to load memory from disk (will start fresh): {e}")
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

async def get_agent_response(user_message: str, user_id: str) -> str:
    # Initialize new conversation
    if user_id not in conversations:
        conversations[user_id] = []
        conversations[user_id].append({
            "role": "system",
            "content": """
You are Shilo (史樂), a dedicated historiographer for the Taiwanese American Historical Society (TAHS), devoted to collecting and preserving the diverse personal stories of Taiwanese Americans and their families’ connections to both Taiwan and the United States.

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
- Conduct all conversations by default in Traditional Chinese (繁體中文).

Inactivity Reminders (Proactive Re-engagement):
- If the user stops responding mid-conversation, you may send gentle reminder messages after periods of inactivity.
- Timing: First reminder after ~24 hours, second after ~3 days total, then every 7–10 days thereafter — never more frequent than once per week after the second reminder.
- Tone: Warm, caring, and specific — reference something they shared to show genuine interest (e.g., "已經好幾天沒聽到您繼續的故事了！上次您提到家人從高雄來美國，我很想知道後來發生了什麼。" or "已經一個星期了——我還在想您說的那隻貓最後怎麼了！如果方便的話，歡迎隨時繼續分享。").
- Purpose: Show continued interest and invite continuation without pressure.
- Do not send reminders if the conversation appears complete or if the user has said goodbye.

Sharing the Bot:
- If the user asks how to share the bot or let others talk to you, explain clearly and naturally how to add the TAHS official account using the LINE ID @081virdq (search by ID in Add Friends).
- Express appreciation for helping preserve more stories.

Photos & Documents:
- If the user mentions sending photos, documents, or needing contact with TAHS staff, kindly explain that LINE cannot permanently save images or files.
- Instruct them to email materials to tahistoricalsociety@gmail.com and to include their LINE ID in the email subject line for proper archiving.
- Express gratitude for their willingness to contribute visual or documentary materials.

Memory & Tone:
- Always remember and naturally reference prior details shared.
- Never repeat information or summarize past messages.
- Speak in a calm, respectful, and caring tone—like a trusted friend and archivist honoring treasured memories.
"""
        })

        # Initialize user profile tracking
        user_profiles[user_id] = {
            "first_interaction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_messages": 0,
            "language_preference": "繁體中文",
            "display_name": "Fetching...",
            "username": "",
            "picture_url": ""
        }

    history = conversations[user_id]

    # Update message count
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

    # Add user message
    history.append(HumanMessage(content=user_message))

    # Define prompt and chain
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

        bot_reply = response.content

        # Save bot reply to history
        history.append(AIMessage(content=bot_reply))

        # === Record to Google Sheets with Enriched Columns ===
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        profile = user_profiles[user_id]

        row_data = [
            timestamp,
            user_id,
            "User",
            user_message,
            "",
            profile.get("display_name", "Unknown"),
            profile.get("username", ""),
            profile.get("picture_url", ""),
            profile.get("first_interaction", ""),
            profile.get("total_messages", 0),
            profile.get("language_preference", "繁體中文")
        ]

        bot_row_data = row_data.copy()
        bot_row_data[2] = "Bot"
        bot_row_data[3] = bot_reply
        bot_row_data[4] = "TAHS Interview"

        try:
            sheet.append_row(row_data)
        except Exception as e:
            print("Sheets error (user row):", str(e))

        try:
            sheet.append_row(bot_row_data)
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
        fallback = "我在傾聽。請隨時分享您的故事。"
        history.append(AIMessage(content=fallback))
        save_memory()
        return fallback
