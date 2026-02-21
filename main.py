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
    return {"message": "Echo is online and ready to make groups more fun and memorable!"}

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

    if is_group and is_ad_or_spam(message_text):
        print("Ignored in group: ad/spam")
        return

    bot_mentioned = False
    if is_group and message_text:
        bot_name = line_bot_api.get_bot_info().display_name or "Echo"
        bot_mentioned = f"@{bot_name}" in message_text or f"@{bot_name.lower()}" in message_text.lower()

    should_reply = not is_group or bot_mentioned

    if not should_reply:
        print("Silent in group: no @mention")
        return

    try:
        if isinstance(event.message, ImageMessage):
            print("Image received → sending to vision")
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

        line_bot_api.reply_message(
            reply_token, TextSendMessage(text=reply_text)
        )
        print("Reply sent successfully!")

    except Exception as e:
        print("Error in handle_message:", str(e))
        traceback.print_exc()
        if not is_group:
            try:
                line_bot_api.reply_message(
                    reply_token,
                    TextSendMessage(text="很抱歉，我遇到小問題～請稍後再試！")
                )
            except:
                pass

def is_ad_or_spam(text: str) -> bool:
    if not text:
        return True
    text = text.lower()
    ad_keywords = ["點贊", "訂閱", "轉發", "打賞", "支持", "關注", "like", "subscribe", "share"]
    return any(kw in text for kw in ad_keywords) or len(text) < 5

@handler.add(JoinEvent)
def handle_join(event):
    group_id = event.source.group_id
    print(f"Echo joined group: {group_id}")

    if not is_group_already_introduced(group_id):
        intro_message = (
            "大家好！我是 Echo（歲月有聲），你們的群組記憶守護者兼開心果～\n"
            "我在群組裡會安靜聽大家聊天，只有被 @Echo 提到時才會出來玩。\n"
            "我可以幫大家記錄珍貴故事、辦小遊戲、回顧舊回憶、分析照片，還會講點溫馨冷笑話。\n"
            "想跟我私下聊天？直接私訊我即可。\n"
            "很高興加入這個群組！有故事想分享，或想玩遊戲，隨時 @我 哦～❤️"
        )
        try:
            line_bot_api.push_message(group_id, TextSendMessage(text=intro_message))
            mark_group_as_introduced(group_id)
        except Exception as e:
            print(f"Failed to send join message: {e}")
