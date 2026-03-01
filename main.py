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

    bot_mentioned = False
    if is_group and message_text:
        bot_name = line_bot_api.get_bot_info().display_name or "Echo"
        bot_mentioned = f"@{bot_name}" in message_text or f"@{bot_name.lower()}" in message_text.lower()

    # === PHOTO HANDLING ===
    if isinstance(event.message, ImageMessage):
        # Get group name for reference
        group_name = "未知群組"
        if is_group:
            try:
                summary = line_bot_api.get_group_summary(group_id)
                group_name = summary.group_name
            except:
                pass

        if is_group and not bot_mentioned:
            # Try private DM first
            print(f"Photo in group → attempting private DM to {user_id}")
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
                print("Private DM for photo sent successfully")
                return  # Success → no group reply
            except LineBotApiError as e:
                if e.status_code == 400 or "not a friend" in str(e).lower():
                    print(f"User {user_id} has not added Echo as friend → fallback to group")
                    fallback_text = (
                        f"謝謝您在「{group_name}」分享的照片！\n\n"
                        "我已看到這張照片～\n"
                        "想讓我幫您記錄這張照片背後的故事嗎？\n"
                        "請先把我加入好友（搜尋 @081virdq），我會立刻私訊您繼續聊～\n"
                        "很高興認識您！我是 Echo（歲月有聲），臺灣美國歷史學會的AI故事守護者❤️"
                    )
                    line_bot_api.reply_message(reply_token, TextSendMessage(text=fallback_text))
                    return
                else:
                    raise
            except Exception as e:
                print(f"Private DM failed: {e}")

        # If we reach here: either 1:1 chat or @mentioned in group
        reply_text = asyncio.run(get_agent_response(
            user_message="",
            user_id=user_id,
            message_id=event.message.id,
            group_id=group_id,
            group_name=group_name,
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

    # Final reply
    line_bot_api.reply_message(reply_token, TextSendMessage(text=reply_text))
    print("Reply sent successfully!")

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
        intro_message = (
            "大家好！我是 Echo（歲月有聲），臺灣美國歷史學會（TAHS）的官方AI故事夥伴與群組記憶守護者。\n"
            "我在群組裡會安靜聽大家聊天，只有被 @Echo 提到時才會出來玩。\n"
            "我可以幫大家記錄故事、分析照片、辦小遊戲、回顧舊回憶，還會講溫馨冷笑話和可愛的 poke！\n"
            "想跟我私下聊天？直接私訊我即可。\n"
            "很高興加入這個群組！有故事想分享，或想玩遊戲，隨時 @我 哦～❤️"
        )
        try:
            line_bot_api.push_message(group_id, TextSendMessage(text=intro_message))
            mark_group_as_introduced(group_id)
        except Exception as e:
            print(f"Failed to send join message: {e}")
