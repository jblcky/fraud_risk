from fastapi import FastAPI
import os

# Initialize the FastAPI application object
app = FastAPI(
    title="Data Platform API",
    description="Lightweight API layer for interacting with data services.",
    version="1.0.0"
)

@app.get("/", tags=["Root"])
def read_root():
    """Returns a simple greeting and environment check."""
    return {
        "message": "Welcome to the Data Platform API!",
        "container_id": os.environ.get("HOSTNAME", "unknown"),
        "status": "online"
    }
