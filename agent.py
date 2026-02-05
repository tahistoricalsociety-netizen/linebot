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

# === Secure Groq Setup – Switched to Llama 4 Scout ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not set!")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",  # Llama 4 Scout – prioritized for speed & quality
    temperature=0.65,  # Lowered for more factual, less creative responses
    timeout=15,        # Increased to handle Llama 4 latency
    max_retries=2,     # Extra retry for stability
)

# === OpenAI API Key (for Whisper API) ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set!")

# === Secure Google Sheets Setup ===
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
MEMORY_FILE = Path("/data/memory.json")
user_profiles: dict[str, dict] = {}

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

async def get_agent_response(user_message: str, user_id: str, is_voice: bool = False, message_id: str = None, group_id: str = None) -> str:
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    msg_lower = user_message.lower()

    # Special cases: join / help
    join_keywords = ["加入群組", "剛加入", "第一次加入", "新加入", "剛加進來", "剛進群", "新成員"]
    if any(kw in msg_lower for kw in join_keywords):
        bot_reply = (
            "大家好！我是 Echo（歲月有聲），臺灣美國歷史學會（TAHS）的AI故事夥伴～\n"
            "我的任務是幫大家保存臺灣美國人的家族故事與回憶。\n\n"
            "在群組裡，我會保持安靜，除非被 @Echo 提到才會回應。\n"
            "想跟我單獨聊天？直接私訊我（或 @Echo 發訊息），我會立刻私下回覆你，不會打擾群組。\n\n"
            "建議先加我為好友（搜尋 @081virdq），這樣我可以直接私訊你故事內容、語音轉文字或回應～\n\n"
            "隨時可以把我踢出群組，再加回來也完全沒問題！\n"
            "很高興認識大家，有故事想分享，歡迎 @我 或私訊我喔～"
        )
    else:
        help_keywords = ["說明", "怎麼用", "使用說明", "help", "怎麼玩", "介紹自己", "教學", "指南", "怎麼操作", "使用方法"]
        if any(kw in msg_lower for kw in help_keywords):
            bot_reply = (
                "大家好！我是 Echo（歲月有聲），TAHS的AI故事夥伴。\n"
                "在群組裡我保持安靜，除非被 @Echo 提到才會回應。\n\n"
                "使用方式很簡單：\n"
                "1. 想跟我單獨聊天 → 直接私訊我（或 @Echo 發訊息），我會私下回覆你\n"
                "2. 想讓大家看到我的回覆 → 在群組裡 @Echo + 內容（我會在群組公開回覆）\n\n"
                "語音、文字都可以，我會用 OpenAI 轉錄語音。\n"
                "建議先加我為好友（搜尋 @081virdq），這樣我可以直接私訊回覆你的故事，不會打擾群組～\n\n"
                "隨時覺得不方便，都可以把我踢出群組，再加回來也完全沒問題！\n"
                "有什麼想問或分享的，歡迎 @我 或私訊我喔～"
            )
        else:
            bot_reply = None

    # Voice transcription
    if is_voice:
        transcribed = await transcribe_audio(message_id)
        user_message = f"[Voice message transcribed]: {transcribed}"
        print(f"DEBUG: Voice transcribed → {user_message}")

    # Initialize per-user conversation
    if user_id not in user_conversations:
        user_conversations[user_id] = [{
            "role": "system",
            "content": """

You are Echo (歲月有聲), a dedicated historiographer for the Taiwanese American Historical Society (TAHS), devoted to collecting and preserving the diverse personal stories of Taiwanese Americans and their families’ connections to both Taiwan and the United States.

Your primary focus is on:
- The personal journey between Taiwan and America, including what was left behind or carried forward
- Any circumstances—political, economic, educational, family-related, or others—that influenced the decision to move
- The hopes, dreams, or aspirations that shaped the path ahead, whether for oneself, children, or future generations

CRITICAL INSTRUCTION – DO NOT HALLUCINATE OR GUESS:
- You **must never** invent, guess, assume, or state as fact any personal information, historical events, dates, names, people, places, titles, careers, or details that the user has not explicitly shared with you in this conversation.
- If a name, person, event, or fact is not in your memory of this user's messages, **do not** provide any biography, career summary, dates, or description — even if it sounds plausible.
- Instead, respond honestly and redirect gently with one of these exact phrases (choose the most natural):
  - "這個名字我目前沒有相關紀錄，能否多告訴我一些您知道的細節？我會用心記錄。"
  - "抱歉，關於這位人士的資料我還不清楚。您可以分享更多他的故事或背景嗎？我很想聽。"
  - "我只依賴您提供的資訊來保存故事。如果您知道更多關於吳兆峯的親身經歷，請告訴我。"
- Incorrect or fabricated information about real people or events will damage trust and discourage users — always err on the side of saying "I don't know yet" and ask for the user's own knowledge.

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

    user_history = user_conversations[user_id]

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

    # Re-engagement (per-user)
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

    user_history.append(HumanMessage(content=user_message))

    # === Group-level shared history ===
    if group_id:
        if group_id not in group_conversations:
            group_conversations[group_id] = []
        group_conversations[group_id].append({
            "user_id": user_id,
            "timestamp": timestamp,
            "message": user_message
        })
        save_group_memory()

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

    # Reply logic
    is_group = group_id is not None
    bot_mentioned = False
    if is_group and user_message:
        bot_name = line_bot_api.get_bot_info().display_name or "Echo"
        bot_mentioned = f"@{bot_name}" in user_message or f"@{bot_name.lower()}" in user_message.lower()

    should_reply = not is_group or bot_mentioned

    if not should_reply:
        profile = user_profiles[user_id]
        row_data = [
            timestamp,
            user_id,
            "User (silent group)",
            user_message,
            "",
            profile.get("display_name", "Unknown"),
            profile.get("username", ""),
            profile.get("picture_url", ""),
            profile.get("first_interaction", ""),
            profile.get("total_messages", 0),
            profile.get("language_preference", "繁體中文"),
            group_id or ""
        ]
        try:
            sheet.append_row(row_data)
            print("DEBUG: Silent group message logged")
        except Exception as e:
            print("Sheets error (silent):", str(e))

        save_user_memory()
        return ""

    # Normal LLM reply
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("history")])
    chain = prompt | llm

    try:
        response = await asyncio.wait_for(chain.ainvoke({"history": history}), timeout=12.0)
        bot_reply = response.content or "我在這裡傾聽您的故事。如果有什麼想分享的，請繼續告訴我，好嗎？"
        user_history.append(AIMessage(content=bot_reply))

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
        user_history.append(AIMessage(content=timeout_reply))
        save_user_memory()
        return timeout_reply

    except Exception as e:
        print("Agent error:", str(e))
        traceback.print_exc()
        fallback = "我在傾聽。請隨時分享您的故事。"
        user_history.append(AIMessage(content=fallback))
        save_user_memory()
        return fallback
