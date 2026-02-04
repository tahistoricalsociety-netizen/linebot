import nest_asyncio
nest_asyncio.apply()
import os
import asyncio
import traceback
import aiohttp
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent,
    TextMessage,
    AudioMessage,
    ImageMessage,
    TextSendMessage
)
from agent import get_agent_response, transcribe_audio

app = FastAPI()

# === Load secrets ===
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN must be set!")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.get("/")
def root():
    return {"message": "TAHS Historiographer Bot is online and ready to preserve Taiwanese American stories."}

@app.post("/webhook")
async def webhook(request: Request):
    signature = request.headers.get("X-Line-Signature", "")
    body = await request.body()
    body_str = body.decode("utf-8")
    try:
        if body_str:  # Skip empty body during LINE verification
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
    group_id = getattr(event.source, 'group_id', None)  # None if 1:1
    is_group = group_id is not None

    print(f"\n=== New message from {user_id} (group: {group_id}) ===")

    # Get message text if it's text
    message_text = ""
    if isinstance(event.message, TextMessage):
        message_text = event.message.text.strip()

    # Ignore ads, spam, or non-meaningful messages
    if is_ad_or_spam(message_text):
        print("Ignored: ad/spam/non-text message")
        return

    # Check if bot was @mentioned (only relevant in groups)
    bot_mentioned = False
    if is_group and message_text:
        bot_name = line_bot_api.get_bot_info().display_name or "Echo"
        bot_mentioned = f"@{bot_name}" in message_text or f"@{bot_name.lower()}" in message_text.lower()

    # Decide whether to reply in group or private
    reply_in_group = is_group and bot_mentioned

    try:
        if isinstance(event.message, ImageMessage):
            print("Image/photo message detected")
            reply_text = (
                "謝謝您分享照片！LINE無法永久保存圖片或檔案。\n"
                "如果照片與您的家族故事相關，請將它們發送到 tahistoricalsociety@gmail.com，"
                "並在郵件主題寫上您的 LINE ID（例如：您的LINE ID - 家族照片），我們會妥善歸檔並連結到您的故事。\n"
                "非常感謝您的貢獻——這些珍貴影像會成為臺灣美國歷史的重要一部分！"
                "您願意繼續分享照片背後的故事嗎？"
            )
        elif isinstance(event.message, AudioMessage):
            print("Voice message detected, transcribing...")
            transcribed = asyncio.run(transcribe_audio(event.message.id))
            print(f"Transcribed: {transcribed[:200]}{'...' if len(transcribed) > 200 else ''}")
            reply_text = asyncio.run(get_agent_response(transcribed, user_id, is_voice=True, message_id=event.message.id))
            reply_text = f"已收到您的語音訊息！轉錄文字如下：\n\n{transcribed}\n\n{reply_text}"
        else:
            user_message = message_text
            print(f"Message: {user_message}")
            reply_text = asyncio.run(get_agent_response(user_message, user_id))

        # Send reply in the right place
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=reply_text)
        )
        print("Reply sent successfully!")

    except Exception as e:
        print("Error in handle_message:", str(e))
        traceback.print_exc()
        # Only send fallback in 1:1 chat (never in group)
        if not is_group:
            try:
                line_bot_api.reply_message(
                    reply_token,
                    TextSendMessage(text="很抱歉，我遇到技術問題。請稍後再試——您的故事對我們很重要。")
                )
            except:
                pass  # Silent if even reply fails

def is_ad_or_spam(text: str) -> bool:
    if not text:
        return True
    text = text.lower()
    ad_keywords = ["點贊", "訂閱", "轉發", "打賞", "支持", "關注", "like", "subscribe", "share"]
    return any(kw in text for kw in ad_keywords) or len(text) < 5
