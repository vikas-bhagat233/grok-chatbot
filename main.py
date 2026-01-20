import os
import re
import requests
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request
from typing import List
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv, dotenv_values
import base64
import matplotlib.pyplot as plt
from pathlib import Path
import sqlite3
from datetime import datetime
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io

# Load environment variables from .env (prefer .env in dev)
load_dotenv(override=True)
_DOTENV = dotenv_values()  # for diagnostics (do not log raw values)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Sanitize common mistakes: value like "GROQ_API_KEY=actualvalue" or quoted
if GROQ_API_KEY:
    val = GROQ_API_KEY.strip().strip('"').strip("'")
    if val.startswith("GROQ_API_KEY="):
        val = val.split("=", 1)[1].strip().strip('"').strip("'")
    GROQ_API_KEY = val
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY is not set; /chat and /ask will fail until configured")

app = FastAPI()

# Serve saved uploads (and any other assets you place under ./static)
_STATIC_DIR = Path(__file__).with_name("static")
(_STATIC_DIR / "uploads").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# Allow frontend (HTML file) to call this API locally
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # for dev; you can restrict later
    # Credentials + wildcard origins can cause CORS issues; auth is disabled anyway.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ok", "message": "Grok Chatbot backend is running. See /health for diagnostics."}


@app.get("/ui")
def ui():
    """Serve the frontend from the same origin as the API.

    This avoids CSP issues from editor previews/live-reload servers that may inject scripts
    and trigger errors like "Content Security Policy blocks the use of eval".
    """
    html_path = Path(__file__).with_name("frontend.html")
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="frontend.html not found")
    return FileResponse(str(html_path), media_type="text/html")

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None

class ChatResponse(BaseModel):
    reply: str

# Groq OpenAI-compatible chat endpoint
GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"  # :contentReference[oaicite:2]{index=2}
GROQ_MODEL = "llama-3.1-8b-instant"  # good fast model on Groq :contentReference[oaicite:3]{index=3}
GROQ_VISION_MODEL = "llama-3.2-11b-vision-preview" # Vision model

# Standard System Prompt
SYSTEM_PROMPT = """You are a helpful, smart assistant.
- You can generate flowcharts, diagrams, and charts using Mermaid.js. Use strict markdown code blocks labeled `mermaid`.
- For standard code, use strict markdown code blocks with the language name (e.g. `python`, `javascript`).
- Be concise and accurate."""

# Simple in-memory store for uploaded content (demo purposes)
DOC_STORE: list[dict] = []

# SQLite-backed sessions store
DB_PATH = "chat.db"

def _db_connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def _init_db():
    conn = _db_connect()
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              id TEXT PRIMARY KEY,
              name TEXT NOT NULL,
              created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              sender TEXT NOT NULL,
              text TEXT NOT NULL,
              formatted INTEGER NOT NULL DEFAULT 0,
              ts TEXT NOT NULL,
              FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
    conn.close()

def _normalize_owner_id(x_client_id: str | None) -> str:
    """Derive a stable per-user owner id.

    We prefer X-Client-Id from the frontend (stored in localStorage).
    This keeps sessions isolated per browser/user even when the frontend is hosted
    on a different origin (Netlify/Vercel) without relying on cookies.
    """
    if not x_client_id:
        return "public"
    x_client_id = x_client_id.strip()[:80]
    # Only keep safe characters so we can safely use it in LIKE prefix patterns.
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", x_client_id)
    return safe or "public"


def _escape_like(value: str) -> str:
    # Escape for SQLite LIKE with ESCAPE '\\'
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _session_prefix(owner_id: str) -> str:
    return f"{owner_id}:"


def _to_internal_session_id(owner_id: str, session_id: str | None) -> str | None:
    if not session_id:
        return None
    return f"{_session_prefix(owner_id)}{session_id}"


def _from_internal_session_id(owner_id: str, internal_id: str) -> str:
    prefix = _session_prefix(owner_id)
    return internal_id[len(prefix):] if internal_id.startswith(prefix) else internal_id


def _get_active_id(conn: sqlite3.Connection, owner_id: str) -> str | None:
    cur = conn.execute("SELECT value FROM meta WHERE key=?", (f"activeId:{owner_id}",))
    row = cur.fetchone()
    return row[0] if row else None

def _set_active_id(conn: sqlite3.Connection, owner_id: str, active_id: str | None):
    key = f"activeId:{owner_id}"
    if active_id is None:
        conn.execute("DELETE FROM meta WHERE key=?", (key,))
    else:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, active_id),
        )

def load_sessions_from_db(owner_id: str):
    conn = _db_connect()
    try:
        prefix = _session_prefix(owner_id)
        like_pattern = _escape_like(prefix) + "%"
        sessions = []
        for s in conn.execute(
            "SELECT id, name, created_at FROM sessions WHERE id LIKE ? ESCAPE '\\' ORDER BY datetime(created_at) ASC",
            (like_pattern,),
        ).fetchall():
            msgs = conn.execute(
                "SELECT sender, text, formatted FROM messages WHERE session_id=? ORDER BY id ASC",
                (s[0],),
            ).fetchall()
            sessions.append(
                {
                    "id": _from_internal_session_id(owner_id, s[0]),
                    "name": s[1],
                    "messages": [
                        {"sender": m[0], "text": m[1], "formatted": bool(m[2])} for m in msgs
                    ],
                }
            )
        active_id = _get_active_id(conn, owner_id)
        return {"sessions": sessions, "activeId": _from_internal_session_id(owner_id, active_id) if active_id else None}
    finally:
        conn.close()

def save_sessions_to_db(payload: dict, owner_id: str):
    conn = _db_connect()
    with conn:
        prefix = _session_prefix(owner_id)
        like_pattern = _escape_like(prefix) + "%"
        conn.execute("DELETE FROM messages WHERE session_id LIKE ? ESCAPE '\\'", (like_pattern,))
        conn.execute("DELETE FROM sessions WHERE id LIKE ? ESCAPE '\\'", (like_pattern,))

        for s in payload.get("sessions", []):
            sid = _to_internal_session_id(owner_id, s["id"])
            name = s.get("name", "Chat")
            conn.execute(
                "INSERT INTO sessions(id, name, created_at) VALUES(?,?,?)",
                (sid, name, datetime.utcnow().isoformat()),
            )
            for m in s.get("messages", []):
                conn.execute(
                    "INSERT INTO messages(session_id, sender, text, formatted, ts) VALUES(?,?,?,?,?)",
                    (sid, m.get("sender", "bot"), m.get("text", ""), 1 if m.get("formatted") else 0, datetime.utcnow().isoformat()),
                )

        _set_active_id(conn, owner_id, _to_internal_session_id(owner_id, payload.get("activeId")))
    conn.close()

def _get_conversation_history(owner_id: str, session_id: str | None, limit: int = 10):
    """Fetch recent messages for context."""
    if not session_id:
        return []

    internal_id = _to_internal_session_id(owner_id, session_id)
    if not internal_id:
        return []

    conn = _db_connect()
    try:
        # Get last N messages (excluding the one we just processed if any, but usually we call this before inserting the new one? 
        # Actually frontend inserts first. But we want history for the model.
        # The frontend calls /chat, getting a reply. The frontend saves to DB via /sessions usually?
        # Wait, the current logic relies on /sessions payload to save.
        # But if we rely on DB for history, we need to ensure the DB is up to date.
        # The frontend calls saveSessions() (which hits /sessions POST) BEFORE calling /chat?
        # Let's check frontend.
        # yes: s.messages.push(...); saveSessions(); ... fetch(/chat)
        # So the NEW user message IS in the DB.
        
        # We want everything ordered by ID.
        rows = conn.execute(
            "SELECT sender, text FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (internal_id, limit)
        ).fetchall()
        
        history = []
        for r in reversed(rows):
            role = "user" if r[0] == "user" else "assistant"
            content = r[1]
            history.append({"role": role, "content": content})
        return history
    except Exception:
        return []
    finally:
        conn.close()

_init_db()

# Very simple in-memory auth (demo only)
USERS = {"admin": "admin", "user": "password"}
TOKENS: dict[str, str] = {}

def require_auth(authorization: str | None):
    # Auth disabled: allow all requests
    return True

class LoginRequest(BaseModel):
    username: str
    password: str

@app.post("/login")
def login(req: LoginRequest):
    if USERS.get(req.username) != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    import secrets
    token = f"{req.username}:{secrets.token_hex(16)}"
    TOKENS[req.username] = token
    return {"token": token, "user": req.username}

# --- Sessions API (SQLite-backed persistence) ---
class SessionItem(BaseModel):
    id: str
    name: str
    messages: list[dict]

class SessionsPayload(BaseModel):
    sessions: list[SessionItem]
    activeId: str | None

@app.get("/sessions")
def get_sessions(x_client_id: str | None = Header(default=None)):
    owner_id = _normalize_owner_id(x_client_id)
    return load_sessions_from_db(owner_id)

@app.post("/sessions")
def set_sessions(payload: SessionsPayload, x_client_id: str | None = Header(default=None)):
    owner_id = _normalize_owner_id(x_client_id)
    save_sessions_to_db({
        "sessions": [s.dict() for s in payload.sessions],
        "activeId": payload.activeId,
    }, owner_id)
    return {"status": "ok"}


@app.get("/health")
def health():
    """Simple health check with Groq config status."""
    looks_valid = bool(GROQ_API_KEY) and str(GROQ_API_KEY).startswith("gsk_")
    src = "dotenv" if _DOTENV.get("GROQ_API_KEY") else "environment"
    preview = None
    if GROQ_API_KEY and len(GROQ_API_KEY) >= 12:
        preview = f"{GROQ_API_KEY[:6]}...{GROQ_API_KEY[-4:]}"
    return {
        "status": "ok",
        "groqConfigured": bool(GROQ_API_KEY),
        "groqKeyLooksValid": looks_valid,
        "groqKeySource": src,
        "groqKeyPreview": preview,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, authorization: str | None = Header(default=None), x_client_id: str | None = Header(default=None)):
    require_auth(authorization)
    """
    Take user message, send to Groq, return the model's reply.
    """
    try:
        if not GROQ_API_KEY:
            raise HTTPException(status_code=400, detail="GROQ_API_KEY is not set. Add it to a .env file or environment variable and restart the server.")

        # Build context from history
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # If session_id provided, fetch history
        owner_id = _normalize_owner_id(x_client_id)
        history = _get_conversation_history(owner_id, req.session_id, limit=20)
        
        # De-duplicate: if the last message in history is exactly the same as req.message from 'user',
        # then we shouldn't append req.message again.
        if history and history[-1]['role'] == 'user' and history[-1]['content'] == req.message:
            # The user message is already in the history (synced from frontend).
            # We can just use 'history' as is.
            messages.extend(history)
        else:
            # User message not in history (maybe sync hadn't finished or race condition),
            # or history was empty. Append explicitly.
            messages.extend(history)
            messages.append({"role": "user", "content": req.message})

        payload = {
            "model": GROQ_MODEL,
            "messages": messages,
        }

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }

        res = requests.post(GROQ_BASE_URL, json=payload, headers=headers, timeout=60)

        if res.status_code != 200:
            print("Groq API error:", res.text)
            # Map invalid key to 401 and missing key already handled above
            status = 502
            msg = "Error from Groq API"
            try:
                j = res.json()
                if isinstance(j, dict):
                    err_obj = j.get("error")
                    err_detail = None
                    err_code = None
                    if isinstance(err_obj, dict):
                        err_detail = err_obj.get("message") or err_obj.get("type")
                        err_code = err_obj.get("code")
                    else:
                        err_detail = j.get("message") or j.get("detail")
                    if err_detail:
                        msg = f"Groq API: {err_detail}"
                    if res.status_code == 401 or (err_code and "invalid_api_key" in str(err_code)) or (err_detail and "invalid api key" in str(err_detail).lower()):
                        status = 401
                        msg = "Invalid GROQ_API_KEY. Update your key in .env or environment variables."
            except Exception:
                pass
            raise HTTPException(status_code=status, detail=msg)

        data = res.json()

        # OpenAI-compatible: choices[0].message.content
        reply = data["choices"][0]["message"]["content"]

        return ChatResponse(reply=reply)

    except HTTPException:
        raise
    except Exception as e:
        print("Server error:", e)
        raise HTTPException(status_code=500, detail="Internal server error")


def _safe_import(module_name: str):
    try:
        return __import__(module_name)
    except Exception:
        return None


@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...), authorization: str | None = Header(default=None)):
    require_auth(authorization)
    """Accept multiple files, extract text/metadata, then explain them via Groq."""
    PyPDF2 = _safe_import("PyPDF2")
    openpyxl = _safe_import("openpyxl")
    pptx = _safe_import("pptx")
    docx = _safe_import("docx")

    items = []

    def clamp(s: str, limit: int = 6000) -> str:
        return s[:limit]

    for f in files:
        name = f.filename or "unnamed"
        content = await f.read()
        size_kb = round(len(content) / 1024, 2)
        lower = name.lower()

        info = {"file": name, "sizeKB": size_kb, "type": "unknown", "text": "", "notes": []}

        try:
            if lower.endswith(".pdf") and PyPDF2:
                info["type"] = "pdf"
                try:
                    reader = PyPDF2.PdfReader(BytesIO(content))
                    pages_text = []
                    for i, p in enumerate(reader.pages[:20]):  # limit to 20 pages
                        try:
                            pages_text.append(p.extract_text() or "")
                        except Exception:
                            pages_text.append("")
                    info["text"] = clamp("\n".join(pages_text))
                    info["notes"].append(f"pages={len(reader.pages)} (truncated)")
                except Exception:
                    info["notes"].append("pdf extraction failed")
            elif (lower.endswith(".xlsx") or lower.endswith(".xlsm")) and openpyxl:
                info["type"] = "excel"
                try:
                    wb = openpyxl.load_workbook(BytesIO(content), read_only=True)
                    info["notes"].append(f"sheets={wb.sheetnames}")
                    # Extract limited cell values from first sheets
                    text_chunks = []
                    for ws in wb.worksheets[:2]:
                        rows = ws.iter_rows(min_row=1, max_row=20, max_col=10, values_only=True)
                        for row in rows:
                            text_chunks.append(" ".join([str(c) for c in row if c is not None]))
                    info["text"] = clamp("\n".join(text_chunks))
                except Exception:
                    info["notes"].append("excel extraction failed")
            elif lower.endswith(".pptx") and pptx:
                info["type"] = "pptx"
                try:
                    pres = pptx.Presentation(BytesIO(content))
                    info["notes"].append(f"slides={len(pres.slides)}")
                    texts = []
                    for slide in pres.slides[:15]:
                        for shape in slide.shapes:
                            if hasattr(shape, "has_text_frame") and shape.has_text_frame:
                                texts.append(shape.text)
                    info["text"] = clamp("\n".join(texts))
                except Exception:
                    info["notes"].append("pptx extraction failed")
            elif lower.endswith(".docx") and docx:
                info["type"] = "docx"
                try:
                    Document = docx.Document  # python-docx
                    d = Document(BytesIO(content))
                    info["text"] = clamp("\n".join([p.text for p in d.paragraphs]))
                except Exception:
                    info["notes"].append("docx extraction failed")
            elif lower.endswith(".txt"):
                info["type"] = "text"
                try:
                    info["text"] = clamp(content.decode("utf-8", errors="replace"))
                except Exception:
                    info["notes"].append("txt decode failed")
            else:
                info["notes"].append("unsupported or missing parser; sending metadata only")
        except Exception:
            info["notes"].append("general extraction failure")

        items.append(info)

    # Save to in-memory store for future Q&A
    DOC_STORE.extend(items)

    # Build prompt for Groq to explain the files.
    parts = []
    for i, it in enumerate(items, 1):
        header = f"File {i}: {it['file']} (≈{it['sizeKB']} KB) type={it['type']}"
        notes = (" | " + ", ".join(it["notes"])) if it["notes"] else ""
        text_block = it["text"].strip()
        if text_block:
            parts.append(f"{header}{notes}\nContent:\n{text_block}\n---")
        else:
            parts.append(f"{header}{notes}\n(No extractable text)\n---")

    prompt = (
        "You are a helpful assistant. The user uploaded the following files. "
        "For each file, briefly explain what it contains, key topics, and any notable data. "
        "Use clear bullet points when helpful, avoid guessing beyond the provided text or metadata, "
        "and keep the explanation concise.\n\n" + "\n".join(parts)
    )

    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        res = requests.post(GROQ_BASE_URL, json=payload, headers=headers, timeout=60)
        if res.status_code != 200:
            print("Groq API error:", res.text)
            # Fall back to metadata-only summary with a helpful prefix if key invalid
            prefix = ""
            try:
                j = res.json()
                err_obj = j.get("error") if isinstance(j, dict) else None
                err_code = err_obj.get("code") if isinstance(err_obj, dict) else None
                err_msg = (err_obj.get("message") if isinstance(err_obj, dict) else None) or j.get("message") if isinstance(j, dict) else None
                if res.status_code == 401 or (err_code and "invalid_api_key" in str(err_code)) or (err_msg and "invalid api key" in str(err_msg).lower()):
                    prefix = "Groq key invalid or missing — returning metadata-only summary.\n\n"
            except Exception:
                pass
            fallback = []
            for it in items:
                line = f"File: {it['file']} (≈{it['sizeKB']} KB) type={it['type']}"
                if it["notes"]:
                    line += " | " + ", ".join(it["notes"])
                fallback.append(line)
            return {"reply": prefix + "\n".join(fallback)}


        # OpenAI-compatible choice
        data = res.json()
        reply = data["choices"][0]["message"]["content"]
        return {"reply": reply}
    except Exception as e:
        print("Explain error:", e)
        # Another fallback
        fallback = []
        for it in items:
            line = f"File: {it['file']} (≈{it['sizeKB']} KB) type={it['type']}"
            if it["notes"]:
                line += " | " + ", ".join(it["notes"])
            fallback.append(line)
        return {"reply": "\n".join(fallback)}


@app.post("/chart")
def chart_from_excel_preview(authorization: str | None = Header(default=None)):
    require_auth(authorization)
    """Generate a simple chart image (base64) from the last Excel preview in DOC_STORE."""
    # Find last excel item with some numeric row
    for it in reversed(DOC_STORE):
        if it.get("type") == "excel" and it.get("text"):
            # Very naive parse: split lines, take first line with numbers
            lines = [line for line in it["text"].split("\n") if line.strip()]
            nums = []
            labels = []
            for line in lines:
                parts = [p.strip() for p in line.split(" ") if p.strip()]
                row_nums = []
                for p in parts:
                    try:
                        val = float(p)
                        row_nums.append(val)
                    except Exception:
                        labels.append(p)
                if row_nums:
                    nums = row_nums
                    break
            if not nums:
                return {"detail": "No numeric data found in Excel preview."}
            try:
                fig, ax = plt.subplots(figsize=(4,2.5))
                ax.bar(range(len(nums)), nums, color="#3b82f6")
                ax.set_title(f"{it['file']} preview")
                ax.set_xticks(range(len(nums)))
                ax.set_xticklabels([str(i+1) for i in range(len(nums))])
                fig.tight_layout()
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                b64 = base64.b64encode(buf.read()).decode('ascii')
                return {"reply": f"Here is a chart preview for {it['file']}:\n![Chart](data:image/png;base64,{b64})"}
            except Exception as e:
                print("Chart error:", e)
                return {"detail": "Failed to generate chart."}
    return {"detail": "No Excel data available for chart."}


class ImageRequest(BaseModel):
    prompt: str
    seed: int | None = None
    width: int | None = 512
    height: int | None = 512


@app.post("/upload_image")
async def upload_image(request: Request, file: UploadFile = File(...), authorization: str | None = Header(default=None)):
    require_auth(authorization)
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")

    content_type = (file.content_type or "").lower()
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Keep extension if present; otherwise default to .png
    original_name = file.filename or "upload.png"
    ext = Path(original_name).suffix.lower()
    if ext not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
        # Fall back to a safe default based on content_type
        ext = ".png"

    safe_stem = Path(original_name).stem
    safe_stem = "".join(ch for ch in safe_stem if ch.isalnum() or ch in ("-", "_"))[:50] or "image"
    fname = f"{safe_stem}-{int(datetime.utcnow().timestamp() * 1000)}{ext}"

    out_path = _STATIC_DIR / "uploads" / fname
    out_path.write_bytes(data)

    rel_url = f"/static/uploads/{fname}"
    base = str(request.base_url).rstrip("/")
    abs_url = f"{base}{rel_url}"

    # Analyze with Groq Vision
    description = ""
    try:
        if GROQ_API_KEY:
            # Convert to JPEG for better compatibility (Groq/Llama Vision often prefers standard formats)
            # Also ensures we strip metadata and handle weird formats like WEBP/TIFF gracefully.
            try:
                img_obj = Image.open(io.BytesIO(data))
                if img_obj.mode in ("RGBA", "P"): 
                    img_obj = img_obj.convert("RGB")
                
                # Resize if too large (max 1024x1024 is usually safe and preferred for speed)
                max_dim = 1024
                if img_obj.width > max_dim or img_obj.height > max_dim:
                    img_obj.thumbnail((max_dim, max_dim))
                
                buf = io.BytesIO()
                img_obj.save(buf, format="JPEG", quality=85)
                jpeg_bytes = buf.getvalue()
                b64_img = base64.b64encode(jpeg_bytes).decode('utf-8')
                img_url = f"data:image/jpeg;base64,{b64_img}"
            except Exception as e:
                print("Image conversion error:", e)
                # Fallback to original data if resize fails
                b64_img = base64.b64encode(data).decode('utf-8')
                img_url = f"data:{content_type};base64,{b64_img}"
             
            v_payload = {
                "model": GROQ_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Analyze this image carefully. Describe the main subjects, setting, and any visible text. Be accurate and concise. Do not guess details that are not clearly visible."},
                            {"type": "image_url", "image_url": {"url": img_url}}
                        ]
                    }
                ],
                "temperature": 0.5,
                "max_completion_tokens": 1024
            }
            v_headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }
            v_res = requests.post(GROQ_BASE_URL, json=v_payload, headers=v_headers, timeout=60)
            if v_res.status_code == 200:
                v_data = v_res.json()
                description = v_data["choices"][0]["message"]["content"]
            else:
                print("Vision API error:", v_res.text)
    except Exception as e:
        print("Vision error:", e)

    # Store in DOC_STORE
    doc_text = description if description else "(Image uploaded, analysis failed or key missing)"
    DOC_STORE.append({
        "file": original_name,
        "type": "image",
        "sizeKB": round(len(data)/1024, 2),
        "text": doc_text,
        "notes": ["image analysis"]
    })

    reply_msg = f"Uploaded image: {original_name}\n![Uploaded Image]({abs_url})"
    if description:
        reply_msg += f"\n\n**Analysis:**\n{description}"

    return {"reply": reply_msg, "url": abs_url, "path": rel_url}

@app.post("/image")
def generate_image_endpoint(request: Request, req: ImageRequest, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    """
    Generate an image based on the prompt. 
    Currently a placeholder that returns a dummy image unless extended with a real API (e.g. DALL-E).
    """
    # Free demo generator: https://image.pollinations.ai/prompt/{prompt}
    # Add seed + cache-buster so different prompts don't get stuck showing the same cached image.
    import random
    import urllib.parse
    
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Keep more of the prompt (helps differentiate), but still cap to avoid gigantic URLs.
    safe_prompt = urllib.parse.quote(prompt[:200])

    seed = int(req.seed) if req.seed is not None else random.randint(0, 1_000_000_000)
    width = int(req.width or 512)
    height = int(req.height or 512)
    width = max(128, min(width, 1024))
    height = max(128, min(height, 1024))
    cache_bust = int(datetime.utcnow().timestamp() * 1000)

    remote_url = (
        f"https://image.pollinations.ai/prompt/{safe_prompt}"
        f"?nologo=true&seed={seed}&width={width}&height={height}&model=flux&v={cache_bust}"
    )

    # Try to download and serve locally to avoid third-party caching and mixed-origin issues.
    local_url = None
    try:
        r = requests.get(remote_url, timeout=60)
        ct = (r.headers.get("content-type") or "").lower()
        if r.status_code == 200 and r.content and ct.startswith("image/"):
            ext = ".png"
            if "jpeg" in ct or "jpg" in ct:
                ext = ".jpg"
            elif "webp" in ct:
                ext = ".webp"
            elif "gif" in ct:
                ext = ".gif"

            fname = f"generated-{cache_bust}-{seed}{ext}"
            out_path = _STATIC_DIR / "uploads" / fname
            out_path.write_bytes(r.content)

            rel_url = f"/static/uploads/{fname}"
            base = str(request.base_url).rstrip("/")
            local_url = f"{base}{rel_url}"
    except Exception as e:
        print("Image download failed:", e)

    final_url = local_url or remote_url
    return {
        "reply": f"Here is an image for '{prompt}':\n![Generated Image]({final_url})",
        "url": final_url,
        "remote_url": remote_url,
        "seed": seed,
    }



class AskRequest(BaseModel):
    question: str

@app.post("/ask", response_model=ChatResponse)
def ask(req: AskRequest, authorization: str | None = Header(default=None)):
    require_auth(authorization)
    """Answer questions grounded in the uploaded documents (simple demo indexing)."""
    if not DOC_STORE:
        return {"reply": "No documents indexed yet. Upload files first via the Upload button."}
    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is not set. Add it to a .env file or environment variable and restart the server.")

    # Build a compact context from stored docs
    snippets = []
    for it in DOC_STORE[-10:]:  # last 10 items
        header = f"{it['file']} (type={it['type']})"
        text = it.get("text", "").strip()
        if text:
            snippets.append(f"Source: {header}\n{text[:2000]}")
        else:
            notes = ", ".join(it.get("notes", []))
            snippets.append(f"Source: {header} | {notes}")

    prompt = (
        "Answer the user's question using ONLY the provided sources. "
        "Cite relevant sources by file name inline like [source: filename]. If insufficient information, say so.\n\n"
        + "\n\n".join(snippets)
        + f"\n\nQuestion: {req.question}"
    )

    try:
        payload = {
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        res = requests.post(GROQ_BASE_URL, json=payload, headers=headers, timeout=60)
        if res.status_code != 200:
            print("Groq API error:", res.text)
            try:
                j = res.json()
                err_obj = j.get("error") if isinstance(j, dict) else None
                err_code = err_obj.get("code") if isinstance(err_obj, dict) else None
                err_msg = (err_obj.get("message") if isinstance(err_obj, dict) else None) or j.get("message") if isinstance(j, dict) else None
            except Exception:
                err_code, err_msg = None, None
            if res.status_code == 401 or (err_code and "invalid_api_key" in str(err_code)) or (err_msg and "invalid api key" in str(err_msg).lower()):
                raise HTTPException(status_code=401, detail="Invalid GROQ_API_KEY. Update your key in .env or environment variables.")
            raise HTTPException(status_code=502, detail="Upstream model error while answering.")
        data = res.json()
        reply = data["choices"][0]["message"]["content"]
        return {"reply": reply}
    except Exception as e:
        print("Ask error:", e)
        return {"reply": "Error answering question."}
