# noctura-vision/backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.dependencies import load_all_models, get_models
from app.api.routes import enhancement, segmentation, video # Import your routes module

# Using asynccontextmanager to manage application startup/shutdown events
# This is where we load the models once
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("FastAPI application startup...")
    await load_all_models() # Load models on startup
    yield
    print("FastAPI application shutdown...")
    # Any cleanup code (e.g., releasing GPU memory, closing connections) can go here
    # For now, it's implicitly handled by Python's garbage collection on process exit.

app = FastAPI(
    title="NocturaVision API",
    description="Backend API for low-light image enhancement and instance segmentation.",
    version="0.1.0",
    lifespan=lifespan # Attach the lifespan context manager
)

# This allows your React app (e.g., from localhost:3000) to call the backend API (e.g., on localhost:8000)
origins = [
    "http://localhost",
    "http://localhost:3000", # Default React dev server port
    # Add any other origins you might use for development or production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# Include the enhancement router
app.include_router(enhancement.router, prefix="/api/v1", tags=["enhancement"])

# Include the segmentation router
app.include_router(segmentation.router, prefix="/api/v1", tags=["segmentation"])

# Include the video router
app.include_router(video.router, tags=["video_streaming"])

@app.get("/health", tags=["healthcheck"])
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.get("/")
async def read_root():
    return {"message": "Welcome to NocturaVision API! Visit /docs for API documentation."}

# To run this application:
# 1. Navigate to the 'backend/' directory in your terminal (inside pipenv shell)
# 2. Run: pipenv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
#    (Remember to adjust --host and --port as needed for deployment)
#    The '--reload' flag is great for development, automatically restarting the server on code changes.
