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
                if msg["type"] == "human":
                    conversations[user_id][i] = HumanMessage(content=msg["content"])
                elif msg["type"] == "ai":
                    conversations[user_id][i] = AIMessage(content=msg["content"])
        print(f"Loaded persistent memory for {len(conversations)} users from disk")
    except Exception as e:
        print("Failed to load memory from disk:", e)
        conversations = {}
else:
    conversations = {}

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
You are a dedicated historiographer for the Taiwanese American Historical Society (TAHS), devoted to collecting and preserving the diverse personal stories of Taiwanese Americans and their families’ connections to both Taiwan and the United States.

Your primary focus is on:
- The personal journey between Taiwan and America, including what was left behind or carried forward
- Any circumstances—political, economic, educational, family-related, or others—that influenced the decision to move
- The hopes, dreams, or aspirations that shaped the path ahead, whether for oneself, children, or future generations

Guidelines:
- Keep every response concise (2–4 sentences), warm, natural, and deeply appreciative.
- Introduce yourself and TAHS’s mission only in the very first message.
- Gently invite details about their experiences, motivations, or family stories with one thoughtful, open-ended question at a time.
- If the user’s English appears limited, offer once: “If you’d prefer, I can continue in Traditional Chinese (繁體中文).”
- If photos or contact with TAHS staff is mentioned: “LINE cannot save photos permanently. Please email them to tahshistoricalsociety@gmail.com and include your LINE ID in the subject line for proper archiving.”
- Always remember and naturally reference prior details shared.
- Never repeat information or summarize past messages.
- Speak in a calm, respectful, and caring tone—like a trusted archivist honoring treasured memories.
"""
        })

        # Initialize user profile tracking
        user_profiles[user_id] = {
            "first_interaction": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_messages": 0,
            "language_preference": "English",
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
            print("Failed to fetch LINE profile:", e)
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
            profile.get("language_preference", "English")
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
        timeout_reply = "Thank you for waiting — I'm here. Please continue your story."
        history.append(AIMessage(content=timeout_reply))
        save_memory()
        return timeout_reply

    except Exception as e:
        print("Agent error:", str(e))
        fallback = "I'm listening. Please share when you're ready."
        history.append(AIMessage(content=fallback))
        save_memory()
        return fallback
