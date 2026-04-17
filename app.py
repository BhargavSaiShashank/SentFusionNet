import sys
import os
import uvicorn

# Tell Python where to find your Neural Brain
sys.path.append(os.path.abspath("improved_version/webapp/api"))

# Import your FastAPI app
try:
    from main import app
except ImportError:
    # Fallback for different folder structures
    sys.path.append(os.path.abspath("."))
    from improved_version.webapp.api.main import app

if __name__ == "__main__":
    # Start on the official Hugging Face port
    uvicorn.run(app, host="0.0.0.0", port=7860)
