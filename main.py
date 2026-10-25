"""Root entrypoint for TicketLens ASGI production server on Render and Docker."""

import os
import sys
from pathlib import Path

# Automatically inject src directory into Python module search path
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Import the FastAPI application
from ticketlens.api import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
