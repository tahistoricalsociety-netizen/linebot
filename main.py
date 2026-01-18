import nest_asyncio
nest_asyncio.apply()
import os
import asyncio
import traceback
import aiohttp
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, AudioMessage, TextSendMessage
from agent import get_agent_response  # Your async agent function
from faster_whisper import WhisperModel

app = FastAPI()

# === Load secrets ===
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
if not CHANNEL_SECRET or not CHANNEL_ACCESS_TOKEN:
    raise ValueError("LINE_CHANNEL_SECRET and LINE_CHANNEL_ACCESS_TOKEN must be set!")

line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

# Load Whisper model once at startup (small = fast & accurate for Hokkien/Mandarin)
whisper_model = WhisperModel("small", device="cpu", compute_type="int8")

async def transcribe_audio(message_id: str) -> str:
    """Download LINE voice message and transcribe using Whisper"""
    try:
        audio_url = f"https://api-data.line.me/v2/bot/message/{message_id}/content"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                audio_url,
                headers={"Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"}
            ) as resp:
                if resp.status != 200:
                    return "無法下載語音訊息，請稍後再試。"
                audio_data = await resp.read()

        # Save temporarily to disk
        temp_file = Path("/tmp/voice_message.m4a")
        with open(temp_file, "wb") as f:
            f.write(audio_data)

        # Transcribe (Whisper auto-detects language, good for Hokkien/Mandarin mix)
        segments, _ = await asyncio.to_thread(
            whisper_model.transcribe,
            str(temp_file),
            language="zh",  # Force Chinese/Hokkien detection
            vad_filter=True
        )

        transcribed = " ".join(segment.text for segment in segments).strip()

        # Clean up
        temp_file.unlink()

        return transcribed if transcribed else "語音內容空白，請再試一次。"

    except Exception as e:
        print("Transcription error:", str(e))
        return "語音轉文字失敗，請用文字分享或再試一次。"

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

@handler.add(MessageEvent, message=(TextMessage, AudioMessage))
def handle_message(event):
    user_id = event.source.user_id
    reply_token = event.reply_token

    print(f"\n=== New message from {user_id} ===")

    try:
        if isinstance(event.message, AudioMessage):
            # Voice message → transcribe first
            print("Voice message detected, transcribing...")
            transcribed = asyncio.run(transcribe_audio(event.message.id))
            print(f"Transcribed: {transcribed[:200]}{'...' if len(transcribed) > 200 else ''}")
            reply_text = f"已收到您的語音訊息！轉錄文字如下：\n\n{transcribed}\n\n現在請繼續分享您的故事～"
        else:
            # Normal text message
            user_message = event.message.text
            print(f"Message: {user_message}")
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
                TextSendMessage(text="很抱歉，我遇到技術問題。請稍後再試——您的故事對我們很重要。")
            )
        except Exception as reply_error:
            print("Failed to send fallback message:", reply_error)
