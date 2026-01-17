# Deployment Guide for Groq Chatbot

This guide covers how to deploy your chatbot to the web so others can access it.

## Prerequisites

- A [GitHub](https://github.com/) account (to host your code).
- A [Groq API Key](https://console.groq.com/) (you already have this).

## Option 1: Deploy on Render.com (Easiest)

Render is a cloud platform that offers a free tier for web services.

1.  **Push your code to GitHub**:
    - Create a new repository on GitHub.
    - Upload all your files (`main.py`, `frontend.html`, `requirements.txt`, `Dockerfile`, etc.) to it.

2.  **Create a Web Service on Render**:
    - Go to [Dashboard.render.com](https://dashboard.render.com/).
    - Click **New +** -> **Web Service**.
    - Connect your GitHub repository.

3.  **Configure Settings**:
    - **Name**: Give it a name (e.g., `my-grok-bot`).
    - **Runtime**: Select **Python 3**.
    - **Build Command**: `pip install -r requirements.txt`
    - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4.  **Add Environment Variables**:
    - Scroll down to "Environment Variables".
    - Add Key: `GROQ_API_KEY`
    - Add Value: `your_actual_groq_api_key_starting_with_gsk_...`

5.  **Deploy**:
    - Click **Create Web Service**.
    - Wait a few minutes. render will give you a URL (e.g., `https://my-grok-bot.onrender.com`).
    - Open that URL + `/ui` (e.g., `https://my-grok-bot.onrender.com/ui`) to use your bot!

## Option 2: Deploy using Docker

If you have a server with Docker installed (like DigitalOcean, AWS EC2, or your own computer):

1.  **Build the image**:
    ```bash
    docker build -t grok-bot .
    ```

2.  **Run the container**:
    ```bash
    docker run -d -p 8000:8000 -e GROQ_API_KEY="your_actual_key" grok-bot
    ```

3.  Access at `http://your-server-ip:8000/ui`.

## Important Note

- **The UI URL**: The frontend is served at `/ui` (e.g., `your-site.com/ui`). The root `/` just shows a "running" message.
- **Persistence**: On free cloud tiers (like Render Free), the uploaded files and SQLite database (`chat.db`) might disappear if the server restarts. For a production app, you would need an external database (like PostgreSQL) and storage (like S3) for images.
