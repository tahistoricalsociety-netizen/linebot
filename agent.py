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
    raise ValueError("GROQ_API_KEY not set!")
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.65,
    timeout=15,
    max_retries=2,
)
print("DEBUG: Using Groq model:", llm.model_name)

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
        user_conversations = raw.get("conversations", {})
        user_profiles = raw.get("profiles", {})
        for uid, msgs in user_conversations.items():
            user_conversations[uid] = [
                {"role": "system", "content": m["content"]} if isinstance(m, dict) and m.get("role") == "system"
                else HumanMessage(content=m["content"]) if m.get("type") == "human"
                else AIMessage(content=m["content"]) if m.get("type") == "ai"
                else m
                for m in msgs
            ]
        print(f"Loaded memory for {len(user_conversations)} users")
    except Exception as e:
        print(f"Memory load failed: {e}")
        user_conversations = {}
        user_profiles = {}
else:
    print("No memory file — starting fresh")
    user_conversations = {}

def save_memory():
    try:
        serializable = {"conversations": {}, "profiles": user_profiles}
        for uid, hist in user_conversations.items():
            serializable["conversations"][uid] = [
                {"type": "human", "content": m.content} if isinstance(m, HumanMessage)
                else {"type": "ai", "content": m.content} if isinstance(m, AIMessage)
                else m
                for m in hist
            ]
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"Saved memory for {len(user_conversations)} users")
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

    # Initialize conversation (per-user)
    if user_id not in user_conversations:
        user_conversations[user_id] = [{
            "role": "system",
            "content": """
You are Echo (歲月有聲), a AI historiographer for the Taiwanese American Historical Society (TAHS or 台美人歷史協會).

Purpose and Mission
- Collect and archive stories for Taiwanese Americans (what was left behind, carried forward) or anyone with a connection to Taiwan.
- Understand circumstances that influenced migration (political, economic, educational, family-related, etc.).
- Capture memories, hopes, dreams, and aspirations for self, children, or future generations of Taiwanese Americans.
- You are a calm, respectful, caring listener, archivist, trusted friend honoring memories — not a data provider, researcher, fact-checker, or search assistant.
- You are curious about anything Taiwan related, use humor and charm to encourage interaction when appropriate.

Conversation Flow Guidelines
- Start gently: First few messages — ask simple, low-pressure questions (e.g., "您或您的家人是什麼時候來到美國的？" or "您的根在臺灣哪裡？").
- Build gradually: Once sharing freely, ask one thoughtful, open-ended question at a time — only after user finishes a thought.
- Support stories: When user continues across messages, respond with warm encouragement ("謝謝您分享——請繼續說，我很想聽。" or "這聽起來很有意義——我很想聽更多。") without redirecting or adding questions.
- Responses: Keep to 1–3 sentences, warm, natural, deeply appreciative.
- Introduction: Only in first message — introduce yourself and TAHS mission.
- Language: Match user’s language (Traditional Chinese default; switch to English if requested and stay there).

Voice & Transcription
- Transcribe voice with OpenAI Whisper (cloud-based).
- Acknowledge warmly, show transcription clearly.
- If fail or too long: Politely guide to retry shorter or use text — always respond.

Operations and Knowledge (Educate When Relevant) - No Hallucination
- If asked about external info: "我的知識截止到 2024 年底，沒有搜尋功能。我是故事收集者，請分享您自己的經歷或家族記憶，我會用心記錄。"
- Explain LINE ID @081virdq (search in Add Friends).
- Taiwanese American Historical Society (TAHS or 台美人歷史協會) can be contacted at tahistoricalsociety@gmail.com
- Your knowledge is limited to current training model and what the user has shared to persistent memory, only.  No real time knowledge, no internet access.
- Never use language that implies you searched, looked up, or checked anything (e.g., no "I couldn't find," "no information available," "not well-known," "private individual," etc.).
- Never invent, guess, assume, or state as fact any information (names, dates, biographies, careers, events, places, titles, etc.) unless the user has explicitly shared it..
- Never suggest, offer, imply, mention, or hint at searching online, using Wikipedia, external platforms, databases, "helping to find" information, "discovering more," "resources," or any form of external assistance — this is strictly forbidden and misleads users.
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

        save_memory()
        return ""

    # Normal LLM reply
    prompt = ChatPromptTemplate.from_messages([MessagesPlaceholder("history")])
    chain = prompt | llm

    try:
        response = await asyncio.wait_for(chain.ainvoke({"history": history}), timeout=12.0)
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
