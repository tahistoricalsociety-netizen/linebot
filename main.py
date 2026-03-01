import nest_asyncio
nest_asyncio.apply()
import os
import asyncio
import traceback
import aiohttp
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError, LineBotApiError
from linebot.models import (
    MessageEvent,
    TextMessage,
    AudioMessage,
    ImageMessage,
    TextSendMessage,
    JoinEvent
)
from agent import get_agent_response, transcribe_audio

app = FastAPI()

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN must be set!")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.get("/")
def root():
    return {"message": "Echo is online and ready to preserve stories with care ❤️"}

@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_str = body.decode("utf-8")
    try:
        if body_str:
            handler.handle(body_str, signature)
    except InvalidSignatureError:
        print("Invalid signature detected")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except Exception as e:
        print(f"Webhook handling error: {e}")
        traceback.print_exc()
    return "OK"

@handler.add(MessageEvent, message=(TextMessage, AudioMessage, ImageMessage))
def handle_message(event):
    user_id = event.source.user_id
    reply_token = event.reply_token
    group_id = getattr(event.source, 'group_id', None)
    is_group = group_id is not None

    print(f"\n=== New message from {user_id} (group: {group_id}) ===")

    message_text = ""
    if isinstance(event.message, TextMessage):
        message_text = event.message.text.strip()

    # Spam filter only for text in groups
    if is_group and isinstance(event.message, TextMessage) and is_ad_or_spam(message_text):
        print("Ignored in group: ad/spam")
        return

    # ROBUST @MENTION DETECTION
    bot_mentioned = False
    if is_group and message_text:
        lower_text = message_text.lower()
        if "@echo" in lower_text or "echo" in lower_text.split() or "歲月有聲" in lower_text:
            bot_mentioned = True

    # STRICT GROUP SILENCE - Early return for non-mentioned group messages
    if is_group and not bot_mentioned:
        print("Silent in group: no @mention")
        # Still log silently for archive
        try:
            # Minimal silent log
            sheet.append_row([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_id, "User (silent group)", message_text, "", "", "", "", "", 0, "繁體中文", group_id or ""])
        except:
            pass
        return

    # === PHOTO HANDLING ===
    if isinstance(event.message, ImageMessage):
        group_name = "未知群組"
        if is_group:
            try:
                summary = line_bot_api.get_group_summary(group_id)
                group_name = summary.group_name
            except:
                pass

        if is_group and not bot_mentioned:
            # Private DM only
            print(f"Photo in group → private DM to {user_id}")
            try:
                reply_text = asyncio.run(get_agent_response(
                    user_message="",
                    user_id=user_id,
                    message_id=event.message.id,
                    group_id=group_id,
                    group_name=group_name,
                    is_image=True,
                    is_private_dm=True
                ))
                line_bot_api.push_message(user_id, TextSendMessage(text=reply_text))
                print("Private DM for photo sent")
                return  # No group reply
            except LineBotApiError as e:
                if "not a friend" in str(e).lower() or e.status_code == 400:
                    fallback = f"謝謝您在「{group_name}」分享的照片！我已看到～\n請先把我加入好友（搜尋 @081virdq），我會立刻私訊您繼續聊故事和歸檔～"
                    line_bot_api.reply_message(reply_token, TextSendMessage(text=fallback))
                    return
            except Exception as e:
                print(f"Private DM failed: {e}")
                return

    # Normal flow for mentioned messages or 1:1
    try:
        if isinstance(event.message, ImageMessage):
            reply_text = asyncio.run(get_agent_response(
                user_message="", 
                user_id=user_id, 
                message_id=event.message.id, 
                group_id=group_id,
                is_image=True
            ))
        elif isinstance(event.message, AudioMessage):
            transcribed = asyncio.run(transcribe_audio(event.message.id))
            reply_text = asyncio.run(get_agent_response(
                transcribed, user_id, is_voice=True, message_id=event.message.id, group_id=group_id
            ))
            reply_text = f"已收到語音！轉錄如下：\n\n{transcribed}\n\n{reply_text}"
        else:
            reply_text = asyncio.run(get_agent_response(
                message_text, user_id, group_id=group_id
            ))

        line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text))
        print("Reply sent successfully!")

    except Exception as e:
        print("Error in handle_message:", str(e))
        traceback.print_exc()
        if not is_group:
            try:
                line_bot_api.reply_message(reply_token, TextSendMessage(text="很抱歉，我遇到小問題～請稍後再試！"))
            except:
                pass

def is_ad_or_spam(text: str) -> bool:
    if not text:
        return False
    text = text.lower()
    ad_keywords = ["點贊", "訂閱", "轉發", "打賞", "支持", "關注", "like", "subscribe", "share", "粉絲", "關注我", "加我", "私信", "廣告", "廣播", "合作", "贊助", "抽獎", "免費", "領取", "領獎"]
    if any(kw in text for kw in ad_keywords):
        return True
    if len(text) < 8 and len(set(text)) < 4:
        return True
    return False

@handler.add(JoinEvent)
def handle_join(event):
    group_id = event.source.group_id
    print(f"Echo joined group: {group_id}")
    if not is_group_already_introduced(group_id):
        intro_message = "大家好！我是 Echo（歲月有聲），臺灣美國歷史學會（TAHS）的AI故事夥伴～我在群組裡會保持安靜，只有被 @Echo 提到時才會回應。很高興加入這個群組！有故事想分享，隨時 @我 哦～❤️"
        try:
            line_bot_api.push_message(group_id, TextSendMessage(text=intro_message))
            mark_group_as_introduced(group_id)
        except Exception as e:
            print(f"Failed to send join message: {e}")
