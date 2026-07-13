import os
import json
import base64
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Ekyra AI Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ── TIER SYSTEM ──────────────────────────────────────────────
TIERS = {
    "free":     {"limit": 1000,  "label": "Free"},
    "starter":  {"limit": 5000,  "label": "Starter"},
    "pro":      {"limit": 10000, "label": "Pro"},
    "ultimate": {"limit": 10000, "label": "Ultimate"},
}

# Token costs per action (designed so free users get ~20 chats or ~10 images)
TOKEN_COSTS = {
    "chat":    50,   # ~20 chats on free tier
    "analyze": 75,   # ~13 file analyses on free tier
    "image":   100,  # ~10 images on free tier
}

# ── TOKEN MANAGER (JSON file storage) ────────────────────────
QUOTA_FILE = Path("quota_data.json")

def _load_quota() -> dict:
    if QUOTA_FILE.exists():
        try:
            return json.loads(QUOTA_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_quota(data: dict):
    QUOTA_FILE.write_text(json.dumps(data, indent=2))

def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")

def get_user_quota(user_id: str) -> dict:
    """Get or create a user's quota record. Resets daily."""
    data = _load_quota()
    today = _today()
    
    if user_id not in data:
        data[user_id] = {"tier": "free", "date": today, "used": 0}
        _save_quota(data)
    
    user = data[user_id]
    
    # Daily reset: if the date is different, reset usage
    if user.get("date") != today:
        user["used"] = 0
        user["date"] = today
        _save_quota(data)
    
    tier = user.get("tier", "free")
    limit = TIERS.get(tier, TIERS["free"])["limit"]
    remaining = max(0, limit - user.get("used", 0))
    
    return {
        "user_id": user_id,
        "tier": tier,
        "tier_label": TIERS.get(tier, TIERS["free"])["label"],
        "limit": limit,
        "used": user.get("used", 0),
        "remaining": remaining,
    }

def consume_tokens(user_id: str, action: str) -> dict:
    """Consume tokens for an action. Returns quota info or raises if exceeded."""
    cost = TOKEN_COSTS.get(action, 50)
    quota = get_user_quota(user_id)
    
    if quota["remaining"] < cost:
        tier = quota["tier"]
        if tier == "free":
            return {
                "error": "quota_exceeded",
                "message": "You've used all your free messages for today! 🐙 Upgrade to Starter or Pro for more. See you tomorrow!",
                "quota": quota,
            }
        else:
            return {
                "error": "quota_exceeded",
                "message": f"Daily limit reached for your {quota['tier_label']} plan. Your quota resets tomorrow at midnight UTC!",
                "quota": quota,
            }
    
    # Deduct tokens
    data = _load_quota()
    data[user_id]["used"] = data[user_id].get("used", 0) + cost
    _save_quota(data)
    
    return {"ok": True, "quota": get_user_quota(user_id)}


# ── AI MODEL CONFIG ──────────────────────────────────────────
ACTIVE_MODELS = [
    {
        "id": "Auto",
        "name": "Auto ✨",
        "desc": "Picks best AI for you",
        "theme": "auto",
        "api_model": "openai/gpt-4o"
    },
    {
        "id": "Gemini",
        "name": "Gemini 3.1 Pro",
        "desc": "Emails & Google Services",
        "theme": "gemini",
        "api_model": "google/gemini-3.1-pro-preview"
    },
    {
        "id": "Claude",
        "name": "Claude Sonnet 5",
        "desc": "Coding, Essays & Reading",
        "theme": "claude",
        "api_model": "anthropic/claude-sonnet-5"
    },
    {
        "id": "ChatGPT",
        "name": "GPT-4o",
        "desc": "Quick basics & small tasks",
        "theme": "gpt",
        "api_model": "openai/gpt-4o"
    },
    {
        "id": "Grok",
        "name": "Grok 4.3",
        "desc": "Real-time news from X",
        "theme": "grok",
        "api_model": "x-ai/grok-4.3"
    }
]

MODEL_MAP = { m["id"]: m["api_model"] for m in ACTIVE_MODELS }


# ── REQUEST MODELS ───────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str
    content: str
    isFileMessage: Optional[bool] = False

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    model: str
    user_id: str

class ImageGenRequest(BaseModel):
    prompt: str
    model: Optional[str] = "GPT Image 2.0"
    user_id: str


# ── HEALTH & INFO ENDPOINTS ──────────────────────────────────
@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "online"}

@app.get("/models")
def get_models():
    """Returns the list of active models for the frontend to render"""
    return {"models": ACTIVE_MODELS}

@app.get("/quota/{user_id}")
def get_quota(user_id: str):
    """Returns the user's current quota status"""
    return get_user_quota(user_id)


# ── SMART ROUTER ─────────────────────────────────────────────
async def route_prompt(prompt: str, api_key: str) -> str:
    """Uses a fast model to intelligently route the user's prompt to the best AI."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://ekyra.app", 
        "X-Title": "Ekyra AI Routing",
        "Content-Type": "application/json"
    }
    
    system_instruction = (
        "You are an intelligent routing agent. Analyze the user's prompt and determine which AI model is best suited to handle it, based on these exact rules:\n"
        "- CLAUDE: Use for coding tasks, writing essays, writing letters, deep reading, or improving text.\n"
        "- GEMINI: Use for 'how-to' questions, learning tasks, explanations about how to do things, or questions related to Google Workspace (email, Gmail, PowerPoint, etc).\n"
        "- GROK: Use for questions about real-time news, current events, or Twitter/X.\n"
        "- GPT: Use for simple explanations ('explain this to me'), basic general knowledge, casual chat, or if the prompt doesn't clearly fit the others.\n\n"
        "Output ONLY ONE WORD from this list: CLAUDE, GEMINI, GROK, GPT"
    )
    
    payload = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 5
    }
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            res = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
            res.raise_for_status()
            data = res.json()
            decision = data["choices"][0]["message"]["content"].strip().upper()
            
            if "CLAUDE" in decision: return "Claude"
            if "GEMINI" in decision: return "Gemini"
            if "GROK" in decision: return "Grok"
            return "ChatGPT"
    except Exception as e:
        print(f"Router error: {e}")
        return "ChatGPT"


# ── CHAT ENDPOINT ────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise HTTPException(status_code=500, detail="API Key not configured on the backend.")

    # ── CHECK QUOTA ──
    token_result = consume_tokens(req.user_id, "chat")
    if "error" in token_result:
        return token_result  # Returns quota_exceeded message

    # Match the requested model
    if req.model == "Auto ✨" or req.model == "Auto":
        last_message = req.message if req.message else (req.history[-1].content if req.history else "")
        matched_key = await route_prompt(last_message, OPENROUTER_API_KEY)
        actual_model = MODEL_MAP[matched_key]
        display_key = f"{matched_key} (Auto-Routed)"
    else:
        matched_key = next((k for k in MODEL_MAP.keys() if k in req.model), "ChatGPT")
        actual_model = MODEL_MAP[matched_key]
        display_key = matched_key

    messages = []
    messages.append({
        "role": "system",
        "content": "You are Ekyra AI, a highly intelligent and professional AI assistant. Provide helpful, concise, and accurate answers. Format your output using markdown."
    })
    
    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})

    if req.message:
        messages.append({"role": "user", "content": req.message})

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ekyra.app", 
        "X-Title": "Ekyra AI",
        "Content-Type": "application/json"
    }

    payload = {
        "model": actual_model,
        "messages": messages
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            ai_text = data["choices"][0]["message"]["content"]
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
    mime_type = file.content_type
    
    if mime_type and mime_type.startswith("image/"):
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
        try:
            text_content = content.decode('utf-8')
            messages = [
                {"role": "system", "content": "Analyze the following file contents."},
                {"role": "user", "content": f"File Name: {file.filename}\n\nContents:\n{text_content}\n\nPrompt: {prompt}"}
            ]
        except Exception:
            raise HTTPException(status_code=400, detail="Only images and text files are supported for analysis currently.")

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
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "your_openrouter_api_key_here":
        raise HTTPException(status_code=500, detail="API Key not configured on the backend.")

    # ── CHECK QUOTA ──
    token_result = consume_tokens(req.user_id, "image")
    if "error" in token_result:
        return token_result

    # Use OpenRouter to call image generation models
    # GPT Image 2.0 → openai/gpt-image-1 (latest available on OpenRouter)
    # DALL-E 3 → openai/dall-e-3
    if "dall-e" in (req.model or "").lower():
        img_model = "openai/dall-e-3"
        source = "DALL-E 3"
    else:
        img_model = "openai/gpt-image-1"
        source = "GPT Image"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://ekyra.app",
        "X-Title": "Ekyra AI",
        "Content-Type": "application/json",
    }

    payload = {
        "model": img_model,
        "prompt": req.prompt,
        "n": 1,
        "size": "1024x1024",
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/images/generations",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            image_url = data["data"][0].get("url") or data["data"][0].get("b64_json", "")
            quota = get_user_quota(req.user_id)
            
            return {
                "image_url": image_url,
                "source": source,
                "quota": quota,
            }
    except httpx.HTTPStatusError as e:
        print(f"Image gen HTTP error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(status_code=500, detail="Image generation failed. The model may not be available.")
    except Exception as e:
        print(f"Image gen error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate image.")
