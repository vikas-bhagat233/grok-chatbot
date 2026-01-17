# How to Push Your Code to GitHub

Follow these steps to upload your chatbot code to GitHub safely.

## Prerequisites

1.  **Git Installed**: Make sure you have Git installed (`git --version` in terminal).
2.  **GitHub Account**: You need an account at [github.com](https://github.com/).

## Step 1: Initialize Git

Open your terminal (PowerShell or Command Prompt) in the project folder (`D:\grok-chatbot`) and run:

```powershell
git init
```

## Step 2: Stage and Commit Files

This prepares your files to be saved. The `.gitignore` file we created will automatically prevent secrets (like `.env`) and heavy files (like `chat.db` or uploads) from being added.

```powershell
git add .
git commit -m "Initial commit: Grok Chatbot with voice, vision, and diagrams"
```

## Step 3: Create a Repository on GitHub

1.  Go to [Information Dashboard](https://github.com/new).
2.  **Repository name**: `grok-chatbot` (or whatever you like).
3.  **Description**: "A local AI chatbot with voice and vision features."
4.  **Public/Private**: Choose **Public** (visible to everyone) or **Private** (only you).
5.  **Initialize this repository with**: Leave all unchecked (we already have code).
6.  Click **Create repository**.

## Step 4: Link and Push

Copy the commands shown in the section **"…or push an existing repository from the command line"**. They will look like this (replace `YOUR_USERNAME` with your actual GitHub username):

```powershell
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/grok-chatbot.git
git push -u origin main
```

## Step 5: Verify

Refresh your GitHub repository page. You should see your code files (`main.py`, `frontend.html`, etc.) but **NOT** your `.env` file or `chat.db`.

---

## Updating in the Future

If you make more changes later, just run:

```powershell
git add .
git commit -m "Describe your changes"
git push
```
