
"""Load local configuration before importing backend services."""
from pathlib import Path

from dotenv import load_dotenv

# Explicit path works with both uvicorn and test runners, regardless of cwd.
# Environment variables supplied by the host take precedence over .env.
load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
