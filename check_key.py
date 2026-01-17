import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key
api_key = os.getenv("GROQ_API_KEY")

print("--- Groq Key Test ---")
if api_key:
    print(f"✅ Found API Key.")
    if api_key.startswith("gsk_") and len(api_key) > 40:
        print(f"✅ Key format looks valid (starts with 'gsk_').")
        # To protect your key, we only show the first 8 and last 4 characters
        print(f"   Value: {api_key[:8]}...{api_key[-4:]}")
    else:
        print(f"❌ WARNING: Key format looks INVALID.")
        print(f"   It should start with 'gsk_'.")
        print(f"   The value found was: '{api_key}'")
else:
    print("❌ ERROR: GROQ_API_KEY not found in environment.")
    print("   Please ensure you have a .env file in the same directory with the line:")
    print("   GROQ_API_KEY=gsk_your_real_key_here")

print("---------------------")
