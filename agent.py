import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# === YOUR CURRENT (SAFE) GROQ API KEY ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Use the powerful and stable model
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.7,
    timeout=10,        # Prevent long hangs
    max_retries=1,
)

# Google Sheets setup
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
CREDS = Credentials.from_service_account_file("credentials.json", scopes=SCOPES)
client = gspread.authorize(CREDS)

# REPLACE WITH YOUR SHEET ID
SHEET_ID = "1bDQuJTF-ene3Z8lXBKkFowwKKxAYcerpSRnbeFt38sg"  # ← From Step 1

sheet = client.open_by_key(SHEET_ID).sheet1  # Use first tab

# Per-user conversation history (in-memory)
conversations: dict[str, list] = {}

import asyncio
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

async def get_agent_response(user_message: str, user_id: str) -> str:
    # Initialize conversation with smarter, more efficient system prompt
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

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="history"),
    ])


    try:
        # Async invoke with timeout
        response = await asyncio.wait_for(
            chain.ainvoke({"history": history}),
            timeout=12.0
        )

        bot_reply = response.content

        # Save to history
        history.append(AIMessage(content=bot_reply))

        # Tool handling
        chain = prompt | 11m

        # Record to Google Sheets
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            sheet.append_row([timestamp, user_id, "User", user_message, ""])
            sheet.append_row([timestamp, user_id, "Bot", bot_reply, "TAHS Interview"])
            print(f"Recorded: {user_id}")
        except Exception as e:
            print("Sheets error:", str(e))

        return bot_reply

    except asyncio.TimeoutError:
        timeout_reply = "Thank you for waiting — I'm here. Please continue your story."
        history.append(AIMessage(content=timeout_reply))
        return timeout_reply

    except Exception as e:
        print("Agent error:", str(e))
        fallback = "I'm listening. Please share when you're ready."
        history.append(AIMessage(content=fallback))
        return fallback
