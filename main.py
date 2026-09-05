import base64
import io
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import docx
except ImportError:
    docx = None

from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
import json
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# We keep a mapping of common names to OpenRouter models
# Add more models to this dictionary to support them in the app.
MODEL_MAP = {
    "Auto ✨": "openai/chatgpt-4o-latest",
    "ChatGPT": "openai/chatgpt-4o-latest",
    "Claude": "anthropic/claude-3.5-sonnet",
    "Gemini": "google/gemini-flash-1.5",
    "Grok": "x-ai/grok-2",
    "Claude Sonnet 3.5": "anthropic/claude-3.5-sonnet",
    "ChatGPT 5.6 Luna": "openai/gpt-4o-mini",
    "Gemini 3.7 Flash": "google/gemini-flash-1.5",
    "Grok 4.6": "x-ai/grok-2",
    "Claude Opus 5": "anthropic/claude-3-opus",
    "GPT 5.6 Sol": "openai/chatgpt-4o-latest",
    "Gemini 3.1 Pro": "google/gemini-pro-1.5",
    "Grok 4.6 (Premium)": "x-ai/grok-2"
}

app = FastAPI(title="Ekyra AI Backend")

@app.get("/")
async def keep_awake():
    """Health check endpoint for cron jobs to ping"""
    return {"status": "awake", "message": "Ekyra AI Backend is running!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    history: list = []
    model: str = "Auto ✨"
    user_id: str

class ImageGenRequest(BaseModel):
    prompt: str
    model: str = ""
    user_id: str

# ── QUOTA MANAGEMENT ─────────────────────────────────────────

QUOTA_FILE = Path(__file__).parent / "quota_data.json"

def load_quotas():
    if QUOTA_FILE.exists():
        try:
            with open(QUOTA_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_quotas(data):
    with open(QUOTA_FILE, "w") as f:
        json.dump(data, f)

# Tiers configuration (aligns with Flutter app)
TIERS = {
    "Free": {"limit": 400},
    "Pro": {"limit": 1000},
    "Ultra": {"limit": 3000},
    "Ultimate": {"limit": 999999}
}

TOKEN_COSTS = {
    "chat": 20,
    "analyze": 20,
    "image": 40
}

def get_user_quota(user_id: str):
    data = load_quotas()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    if user_id not in data:
        data[user_id] = {
            "tier": "Free",
            "date": today,
            "used": 0
        }
    
    user_data = data[user_id]
    
    if user_data.get("date") != today:
        user_data["date"] = today
        user_data["used"] = 0
        
    save_quotas(data)
    
    limit = TIERS.get(user_data["tier"], TIERS["Free"])["limit"]
    remaining = max(0, limit - user_data["used"])
    
    return {
        "tier": user_data["tier"],
        "used": user_data["used"],
        "limit": limit,
        "remaining": remaining
    }

def consume_tokens(user_id: str, action: str):
    data = load_quotas()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    if user_id not in data:
        data[user_id] = {
            "tier": "Free",
            "date": today,
            "used": 0
        }
        
    user_data = data[user_id]
    if user_data.get("date") != today:
        user_data["date"] = today
        user_data["used"] = 0
        
    cost = TOKEN_COSTS.get(action, 1)
    limit = TIERS.get(user_data["tier"], TIERS["Free"])["limit"]
    
    if user_data["used"] + cost > limit:
        return {"error": "Quota exceeded. Upgrade your plan."}
        
    user_data["used"] += cost
    save_quotas(data)
    
    remaining = max(0, limit - user_data["used"])
    return {
        "tier": user_data["tier"],
        "used": user_data["used"],
        "limit": limit,
        "remaining": remaining
    }

@app.post("/set_tier")
async def set_tier(user_id: str = Form(...), tier: str = Form(...)):
    """Admin endpoint to upgrade a user"""
    if tier not in TIERS:
        raise HTTPException(status_code=400, detail="Invalid tier")
        
    data = load_quotas()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    
    if user_id not in data:
        data[user_id] = {"used": 0, "date": today}
        
    data[user_id]["tier"] = tier
    save_quotas(data)
    return {"status": "success", "user_id": user_id, "tier": tier}

@app.get("/quota/{user_id}")
async def check_quota(user_id: str):
    return get_user_quota(user_id)


# ── MAIN CHAT ENDPOINT ───────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise HTTPException(status_code=500, detail="API Key not configured on the backend.")

    # ── CHECK QUOTA ──
    token_result = consume_tokens(req.user_id, "chat")
    if "error" in token_result:
        return token_result

    # Format history for OpenRouter
    messages = []
    for msg in req.history:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", "")
        })
        
    messages.append({"role": "user", "content": req.message})

    # Model Routing
    actual_model = MODEL_MAP.get(req.model, MODEL_MAP["ChatGPT"])
    display_key = req.model

    # The payload configures the request
    payload = {
        "model": actual_model,
        "messages": messages,
        # Allow the model to call the image generation tool if it detects the user wants an image
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image based on a text prompt. Call this ONLY if the user explicitly asks to generate, draw, or create a picture/image.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "A highly detailed visual description of the image to generate."
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ekyra.app",
        "X-Title": "Ekyra AI",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            res = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            res.raise_for_status()
            data = res.json()
            
            message_obj = data["choices"][0]["message"]
            
            # Check if the AI decided to call the image generation tool
            if message_obj.get("tool_calls"):
                tool_call = message_obj["tool_calls"][0]
                if tool_call["function"]["name"] == "generate_image":
                    args = json.loads(tool_call["function"]["arguments"])
                    image_prompt = args.get("prompt", req.message)
                    
                    import urllib.parse
                    safe_prompt = urllib.parse.quote(image_prompt)
                    image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?model=flux&nologo=true&enhance=true&width=1024&height=1024"
                    
                    # Feed the generated image URL back to the AI so it can describe it
                    messages.append(message_obj)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": "generate_image",
                        "content": f"Image generated successfully. URL: {image_url}"
                    })
                    
                    payload["messages"] = messages
                    payload.pop("tools", None)
                    
                    res2 = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload
                    )
                    res2.raise_for_status()
                    data2 = res2.json()
                    ai_text = data2["choices"][0]["message"]["content"] or ""
                    
                    # Make sure the image URL is in the response
                    if image_url not in ai_text:
                        ai_text += f"\n\n![Generated Image]({image_url})"
                    
                    return {
                        "response": ai_text,
                        "model_used": display_key,
                        "reason": f"Generated image: {image_prompt[:50]}",
                        "image_url": image_url,
                        "quota": get_user_quota(req.user_id),
                    }
            
            ai_text = message_obj.get("content", "")
            quota = get_user_quota(req.user_id)
            
            return {
                "response": ai_text,
                "model_used": display_key,
                "reason": "Generated",
                "quota": quota,
            }
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to connect to AI provider.")


# ── FILE ANALYSIS ENDPOINT ───────────────────────────────────
@app.post("/analyze")
async def analyze_file(
    prompt: str = Form(...),
    model: str = Form(...),
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise HTTPException(status_code=500, detail="API Key not configured on the backend.")

    # ── CHECK QUOTA ──
    token_result = consume_tokens(user_id, "analyze")
    if "error" in token_result:
        return token_result

    content = await file.read()
    actual_model = MODEL_MAP["ChatGPT"]
    
    # Safely identify if it's an image
    mime_type = file.content_type or ""
    filename_lower = file.filename.lower() if file.filename else ""
    is_image = mime_type.startswith("image/") or filename_lower.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp'))
    
    if is_image:
        if not mime_type.startswith("image/"):
            ext = filename_lower.split('.')[-1]
            if ext == 'jpg': ext = 'jpeg'
            mime_type = f"image/{ext}"
            
        b64_image = base64.b64encode(content).decode('utf-8')
        image_url = f"data:{mime_type};base64,{b64_image}"
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    else:
        text_content = ""
        
        try:
            if filename_lower.endswith(".pdf"):
                if PdfReader is None:
                    raise Exception("PDF parsing library not installed on backend.")
                pdf = PdfReader(io.BytesIO(content))
                text_content = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            elif filename_lower.endswith(".docx"):
                if docx is None:
                    raise Exception("Word DOCX parsing library not installed on backend.")
                doc = docx.Document(io.BytesIO(content))
                text_content = "\n".join([p.text for p in doc.paragraphs])
            else:
                text_content = content.decode('utf-8')
                
            # Limit text content to roughly 50,000 characters to prevent API limits
            if len(text_content) > 50000:
                text_content = text_content[:50000] + "\n...[TRUNCATED DUE TO LENGTH]..."
                
            messages = [
                {"role": "system", "content": "Analyze the following file contents."},
                {"role": "user", "content": f"File Name: {file.filename}\n\nContents:\n{text_content}\n\nPrompt: {prompt}"}
            ]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to read document: {str(e)}. Ensure it is a valid text, PDF, or DOCX file.")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ekyra.app",
        "X-Title": "Ekyra AI",
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": actual_model, "messages": messages}
            )
            response.raise_for_status()
            data = response.json()
            quota = get_user_quota(user_id)
            return {
                "response": data["choices"][0]["message"]["content"],
                "model_used": "GPT-4o Vision",
                "quota": quota,
            }
    except Exception as e:
        print(f"Analyze error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to analyze file.")


# ── IMAGE GENERATION ENDPOINT ────────────────────────────────
@app.post("/image")
async def generate_image(req: ImageGenRequest):
    import urllib.parse, os, base64
    
    # ── CHECK QUOTA ──
    token_result = consume_tokens(req.user_id, "image")
    if "error" in token_result:
        return token_result

    safe_prompt = urllib.parse.quote(req.prompt)
    model_requested = (req.model or "").lower()

    # ── Try OpenAI image models if key is available ──
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if openai_key and ("dall" in model_requested or "gpt image" in model_requested or model_requested == ""):
        # Pick the actual model: DALL-E 3 or GPT Image 2.0 (gpt-image-1)
        if "gpt image" in model_requested or "gpt-image" in model_requested:
            api_model = "gpt-image-1"
            source_label = "GPT Image 2.0"
        else:
            api_model = "dall-e-3"
            source_label = "DALL-E 3"
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/images/generations",
                    headers={
                        "Authorization": f"Bearer {openai_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": api_model,
                        "prompt": req.prompt,
                        "n": 1,
                        "size": "1024x1024",
                        "quality": "standard",
                        "response_format": "url",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    image_url = data["data"][0].get("url") or data["data"][0].get("b64_json")
                    if image_url and image_url.startswith("http"):
                        return {"image_url": image_url, "source": source_label}
                    elif image_url:
                        return {"image_url": f"data:image/png;base64,{image_url}", "source": source_label}
        except Exception as e:
            print(f"{source_label} error, falling back to Pollinations: {e}")

    # ── Fallback: Pollinations.ai (free, no key needed) ──
    image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?model=flux&nologo=true&enhance=true&width=1024&height=1024"
    return {
        "image_url": image_url,
        "source": "Pollinations AI (Flux)"
    }


if __name__ == "__main__":
    import uvicorn
    # When deployed on Render or Railway, it uses the PORT environment variable.
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
