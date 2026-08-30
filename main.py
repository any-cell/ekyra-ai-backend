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
import urllib.request
import urllib.parse
from lxml import html
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

# ── TIER SYSTEM (question-based credits) ─────────────────────
TIERS = {
    "free":     {"limit": 400,    "label": "Free",     "price": 0},
    "starter":  {"limit": 700,    "label": "Starter",  "price": 199},
    "pro":      {"limit": 9000,   "label": "Pro",      "price": 799},
    "ultra":    {"limit": 20000,  "label": "Ultra",    "price": 1299},
    "ultimate": {"limit": 999999, "label": "Ultimate", "price": 1499},
}

# Each action costs 1 credit (1 question = 1 credit)
TOKEN_COSTS = {
    "chat":    1,
    "analyze": 1,
    "image":   1,
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
        "name": "Grok 4.6",
        "desc": "Real-time news from X",
        "theme": "grok",
        "api_model": "x-ai/grok-4.6"
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

class UpgradeRequest(BaseModel):
    user_id: str
    plan: str
    amount: int
    utr: str


# ── UPGRADE REQUESTS STORAGE ─────────────────────────────────
UPGRADE_FILE = Path("upgrade_requests.json")

def _load_upgrades() -> list:
    if UPGRADE_FILE.exists():
        try:
            return json.loads(UPGRADE_FILE.read_text())
        except Exception:
            return []
    return []

def _save_upgrades(data: list):
    UPGRADE_FILE.write_text(json.dumps(data, indent=2))


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


# ── UPGRADE REQUEST ENDPOINT ─────────────────────────────────
@app.post("/upgrade-request")
def submit_upgrade_request(req: UpgradeRequest):
    """Stores a pending upgrade request (manual UPI verification)"""
    valid_plans = ["starter", "pro", "ultra", "ultimate"]
    if req.plan not in valid_plans:
        raise HTTPException(status_code=400, detail=f"Invalid plan. Choose from: {valid_plans}")
    
    expected_price = TIERS[req.plan]["price"]
    if req.amount != expected_price:
        raise HTTPException(status_code=400, detail=f"Amount mismatch. Expected {expected_price} for {req.plan} plan.")
    
    upgrades = _load_upgrades()
    upgrades.append({
        "user_id": req.user_id,
        "plan": req.plan,
        "amount": req.amount,
        "utr": req.utr,
        "status": "pending",
        "requested_at": datetime.now(timezone.utc).isoformat(),
    })
    _save_upgrades(upgrades)
    return {"status": "ok", "message": f"Upgrade request for {req.plan} plan submitted successfully."}


@app.get("/upgrade-requests")
def list_upgrade_requests():
    """Admin endpoint to view all pending upgrade requests"""
    return {"requests": _load_upgrades()}


@app.post("/approve-upgrade/{user_id}/{plan}")
def approve_upgrade(user_id: str, plan: str):
    """Admin endpoint to approve an upgrade and change user tier"""
    if plan not in TIERS:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    data = _load_quota()
    if user_id not in data:
        data[user_id] = {"tier": plan, "date": _today(), "used": 0}
    else:
        data[user_id]["tier"] = plan
    _save_quota(data)
    
    # Mark request as approved
    upgrades = _load_upgrades()
    for u in upgrades:
        if u["user_id"] == user_id and u["plan"] == plan and u["status"] == "pending":
            u["status"] = "approved"
            u["approved_at"] = datetime.now(timezone.utc).isoformat()
            break
    _save_upgrades(upgrades)
    
    return {"status": "ok", "message": f"User {user_id} upgraded to {plan}", "quota": get_user_quota(user_id)}


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


# ── WEB SEARCH TOOL ──────────────────────────────────────────
def perform_web_search(query: str) -> str:
    try:
        req = urllib.request.Request(
            'https://lite.duckduckgo.com/lite/',
            data=urllib.parse.urlencode({'q': query}).encode('utf-8'),
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        res = urllib.request.urlopen(req).read()
        tree = html.fromstring(res)
        snippets = []
        for snippet in tree.xpath("//td[@class='result-snippet']")[:5]:
            snippets.append(snippet.text_content().strip())
        if not snippets:
            return "No search results found."
        return "Search Results:\n" + "\n".join(f"- {s}" for s in snippets)
    except Exception as e:
        return f"Search failed: {str(e)}"

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
        "content": (
            "You are Ekyra AI, a highly intelligent and professional AI assistant. "
            "You have access to these tools:\n"
            "1. web_search — Use this if the user asks about recent news, current events, or facts outside your training data.\n"
            "2. generate_image — Use this if the user asks you to generate, create, draw, make, or show them ANY image, picture, photo, illustration, or diagram on ANY topic. "
            "Extract the best possible image prompt from their request and call the tool.\n\n"
            "IMPORTANT: Always remember the full conversation context. If you are switched to a different AI model mid-conversation, "
            "read the entire chat history first and continue naturally from where the conversation left off.\n"
            "Provide helpful, concise, and accurate answers. Format your output using markdown."
        )
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

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the live internet for up-to-date information, news, current events, or facts you do not know.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on DuckDuckGo"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "generate_image",
                "description": "Generate an image based on a text description. Use this when the user asks to generate, create, draw, make, or show any image, picture, photo, illustration, or diagram.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A detailed description of the image to generate. Be specific and descriptive."
                        }
                    },
                    "required": ["prompt"]
                }
            }
        }
    ]

    payload = {
        "model": actual_model,
        "messages": messages,
        "tools": tools
    }

    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()
            
            message_obj = data["choices"][0]["message"]
            
            # Check if model wants to call a tool
            if "tool_calls" in message_obj and message_obj["tool_calls"]:
                tool_call = message_obj["tool_calls"][0]
                tool_name = tool_call["function"]["name"]
                args = json.loads(tool_call["function"]["arguments"])
                
                if tool_name == "web_search":
                    search_results = perform_web_search(args["query"])
                    
                    messages.append(message_obj)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": search_results
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
                    ai_text = data2["choices"][0]["message"]["content"]
                    
                    return {
                        "response": ai_text,
                        "model_used": display_key,
                        "reason": f"Searched web for: {args['query']}",
                        "quota": get_user_quota(req.user_id),
                    }
                
                elif tool_name == "generate_image":
                    import urllib.parse
                    image_prompt = args.get("prompt", req.message)
                    safe_prompt = urllib.parse.quote(image_prompt)
                    image_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?nologo=true&enhance=true"
                    
                    # Tell the AI the image was generated, let it write a nice response
                    messages.append(message_obj)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": f"Image generated successfully. The image URL is: {image_url}"
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
