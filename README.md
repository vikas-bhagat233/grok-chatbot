# Groq Chatbot (FastAPI + HTML)

A lightweight local chatbot UI with a FastAPI backend powered by **Groq LLMs**.

* Rich chat UI with themes, typing indicator, edit/retry, drag-and-drop files
* Upload and analyze PDF / PPTX / XLSX / DOCX / TXT
* “Ask over files” Q&A grounded on the uploaded content
* Basic chart preview generation from Excel snippets
* Local chat sessions (rename/delete) persisted in browser localStorage

## Prerequisites

* Python 3.10+
* pip

## Setup

1. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```
2. Configure your Groq API key. Either:

   * Create a `.env` file and set your key:

     ```
     GROQ_API_KEY=gsk_your_real_groq_api_key
     ```

     Note: The backend prefers `.env` values over machine/user environment variables (helps avoid stale `setx` values).
   * Or set it for the current PowerShell session:

     ```powershell
     $env:GROQ_API_KEY="gsk_your_real_groq_api_key"
     ```

## Run

### Option 1 (Recommended – Windows) - use this only 

```powershell
Set-Location 'D:\grok-chatbot'
cmd /c "run_backend.bat"
```

### Option 2 (Manual)

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Health Check

Quickly verify backend status and key detection:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
```

Expected fields:

* `status`: ok
* `groqConfigured`: whether a key is loaded
* `groqKeyLooksValid`: true if the key starts with `gsk_`
* `groqKeySource`: `dotenv` or `environment`
* `groqKeyPreview`: masked head/tail of the active key

## API

* POST `/chat`

  * Body: `{ "message": "Hello" }`
  * Returns: `{ "reply": "..." }`
* POST `/ask`

  * Body: `{ "question": "What does the doc say about X?" }`
  * Answers using only uploaded file snippets. Returns: `{ "reply": "..." }`
* POST `/analyze`

  * Multipart form-data: one or more `files`
  * Extracts text/metadata and explains contents via the model. Returns: `{ "reply": "..." }`
* GET `/health`

  * Returns backend status + key diagnostics.
* POST `/chart` (best-effort)

  * Generates a simple base64 bar chart from the last Excel preview (if available). Returns: `{ "reply": "![Chart](data:image/png;base64,...)" }`

## Frontend Tips

* Theme and color: switch themes (Light/Dark/Forest/Sunset/Ocean) and customize the bot bubble color in settings.
* Sessions: create/rename/delete chats in the left sidebar. Sessions persist in `localStorage` and survive reloads.
* Ask over files: toggle “Ask over files” to ground answers on the last uploaded snippets.
* Files: click “Upload” or drag-and-drop files into the chat area.
* Status badge: the dot next to the title shows backend + key state (green/yellow/red).

## Troubleshooting

* **Invalid `GROQ_API_KEY`:**

  * Rotate your key in the Groq console and paste the new one into `.env`.
  * Restart the server after editing `.env`.
  * Prefer `.env` for local dev; it overrides stale machine/user env values.
  * Clear in-session PowerShell variable if needed:

    ```powershell
    $env:GROQ_API_KEY=$null
    ```
  * Verify with `/health` or run:

    ```powershell
    python check_key.py
    ```
* **Backend offline (red badge):**

  ```powershell
  python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
  ```
* **Upload parsing:**

  * PDF: `PyPDF2`
  * Excel: `openpyxl`
  * PowerPoint: `python-pptx`
  * Word: `python-docx`
  * If a parser fails, the backend returns a graceful metadata summary.

## Notes

* Sessions currently persist in the browser only; the backend includes SQLite endpoints that are not used by the UI.
* CORS is permissive for local development.
* This project is intended for local use; do not expose secrets or run untrusted files.
