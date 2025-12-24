import nest_asyncio
nest_asyncio.apply()

import os
import asyncio
import traceback
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from agent import get_agent_response  # Your async agent function

app = FastAPI()

# === SECURE: Load secrets from environment variables ===
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")

if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN must be set as environment variables!")

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

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_message = event.message.text
    user_id = event.source.user_id
    reply_token = event.reply_token

    print(f"\n=== New message from {user_id} ===")
    print(f"Message: {user_message}")

    try:
        # Run the async agent function safely
        reply_text = asyncio.run(get_agent_response(user_message, user_id))

        print(f"Bot reply: {reply_text[:200]}{'...' if len(reply_text) > 200 else ''}")

        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=reply_text)
        )
        print("Reply sent successfully!")

    except Exception as e:
        print("Error in handle_message:", str(e))
        traceback.print_exc()
        try:
            line_bot_api.reply_message(
                reply_token,
                TextSendMessage(text="I'm experiencing a brief technical issue. Please try again soon — your story is important to us.")
            )
        except Exception as reply_error:
            print("Failed to send fallback message:", reply_error)
