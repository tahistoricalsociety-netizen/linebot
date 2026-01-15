import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime, timedelta
import os
import json
from langchain_groq import ChatGroq
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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

# === Wikipedia Tool ===
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

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

async def get_agent_response(user_message: str, user_id: str) -> str:
    current_time = datetime.now()

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
- For any historical facts, events, dates, or names related to Taiwan, Taiwanese history, or Taiwanese American topics, ALWAYS use the Wikipedia tool first to ensure accuracy and include a brief citation.
- Use the Wikipedia tool when needed for accurate historical context about Taiwan or Taiwanese American history.

Re-engagement After Inactivity:
- When the user returns after a pause, warmly acknowledge the time passed and reference something specific they shared earlier.
- Examples:
  - After a few days: "歡迎回來！上次您提到家人從高雄來美國，我一直很想知道後來發生了什麼。"
  - After a week or more: "已經有一陣子沒聽到您的故事了！上次您說到那段經歷，我還在想著呢——如果方便的話，歡迎繼續分享。"
- This shows genuine care and memory without pressure.

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

    # Add user message
    history.append(HumanMessage(content=user_message))

    # Define prompt and chain with Wikipedia tool
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
    ])
    chain = prompt | llm.bind_tools([wikipedia_tool], tool_choice="auto")

    try:
        # Async invoke with timeout
        response = await asyncio.wait_for(
            chain.ainvoke({"history": history}),
            timeout=12.0
        )

        # Start with response content (may be empty when tool calls present)
        bot_reply = response.content or ""

        # Handle Wikipedia tool calls
        if response.tool_calls:
            print("Tool call detected:", response.tool_calls)  # Debug log
            for tool_call in response.tool_calls:
                if tool_call["name"].lower() == "wikipedia_query_run":
                    query = tool_call["args"].get("query", "")
                    try:
                        result = wikipedia_tool.run(query)
                        short_result = result[:500] + "..." if len(result) > 500 else result
                        tool_reply = f"根據維基百科：{short_result}\n\n這對您的家族經歷有什麼相關之處呢？"
                        history.append(AIMessage(content=tool_reply))
                        bot_reply = tool_reply  # Explicitly set from tool
                    except Exception as e:
                        print("Wikipedia tool error:", str(e))
                        bot_reply = "我正在查詢相關歷史資訊，請稍等..."

        # Ensure reply is never empty (LINE requires 1+ char)
        if not bot_reply or bot_reply.strip() == "":
            bot_reply = "我在這裡傾聽您的故事。如果有什麼想分享的，請繼續告訴我，好嗎？"

        # Save bot reply to history
        history.append(AIMessage(content=bot_reply))

        # === Record to Google Sheets with Enriched Columns ===
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
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
        timeout_reply = "感謝您的耐心等待——我在這裡。請繼續分享您的故事，好嗎？"
        history.append(AIMessage(content=timeout_reply))
        save_memory()
        return timeout_reply

    except Exception as e:
        print("Agent error:", str(e))
        fallback = "我在傾聽。請隨時分享您的故事。"
        history.append(AIMessage(content=fallback))
        save_memory()
        return fallback
