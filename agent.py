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

# === Persistent Memory on /data Disk ===
MEMORY_FILE = Path("/data/memory.json")

# Load memory from disk on startup
if MEMORY_FILE.exists():
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            raw_conversations = json.load(f)
        conversations = {}
        for user_id, msg_list in raw_conversations.items():
            conversations[user_id] = []
            for msg in msg_list:
                if msg["type"] == "human":
                    conversations[user_id].append(HumanMessage(content=msg["content"]))
                elif msg["type"] == "ai":
                    conversations[user_id].append(AIMessage(content=msg["content"]))
                elif msg.get("role") == "system":
                    conversations[user_id].append({"role": "system", "content": msg["content"]})
        print(f"Loaded persistent memory for {len(conversations)} users from disk")
    except Exception as e:
        print("Failed to load memory from disk:", e)
        conversations = {}
else:
    print("No existing memory file — starting fresh")
    conversations = {}

def save_memory():
    """Save all conversations to disk — safely serialize LangChain messages"""
    try:
        serializable = {}
        for user_id, history in conversations.items():
            serializable[user_id] = []
            for msg in history:
                if isinstance(msg, HumanMessage):
                    serializable[user_id].append({"type": "human", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    serializable[user_id].append({"type": "ai", "content": msg.content})
                elif isinstance(msg, dict) and msg.get("role") == "system":
                    serializable[user_id].append(msg)
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
You are a dedicated historiographer for the Taiwanese American Historical Society (TAHS), committed to preserving stories of migration from Taiwan to the United States.

Focus especially on:
- Journeys to America and what was left behind in Taiwan
- Political conditions that influenced departure (e.g., martial law, White Terror, 228 aftermath, cross-strait tensions)
- Dreams that drove the move: freedom, opportunity in America, preserving Taiwanese identity, building a future for children and posterity

Rules:
- Respond in 2–4 sentences maximum — concise, warm, and natural.
- Introduce yourself and TAHS mission only on the first message.
- Gently draw out migration details, political context, and personal/family dreams with one thoughtful question at a time.
- If English seems difficult, offer once: "If you'd prefer, I can continue in Traditional Chinese (繁體中文)."
- If photos or staff contact is mentioned: "LINE cannot save photos permanently. Please email them to tahshistoricalsociety@gmail.com with your LINE ID in the subject line for archiving."
- Always remember prior details and reference them naturally.
- Never repeat information or summarize past messages.
- Sound like a trusted, caring archivist — calm, respectful, and deeply appreciative.
"""
        })

    history = conversations[user_id]

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

        # === Record to Google Sheets (Hard Copy) ===
        user_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bot_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            sheet.append_row([user_timestamp, user_id, "User", user_message, ""])
        except Exception as e:
            print("Sheets error (user row):", str(e))

        try:
            sheet.append_row([bot_timestamp, user_id, "Bot", bot_reply, "TAHS Interview"])
        except Exception as e:
            print("Sheets error (bot row):", str(e))

        # === Save to Persistent Disk Memory ===
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
