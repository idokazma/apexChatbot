"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.dependencies import resources
from api.routes import admin, chat, health, tester


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("Starting Harel Insurance Chatbot API...")
    resources.initialize()
    logger.info("All resources initialized")
    yield
    logger.info("Shutting down...")
    resources.shutdown()


app = FastAPI(
    title="Harel Insurance Chatbot",
    description="AI-powered customer support chatbot for Harel Insurance",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat.router, tags=["chat"])
app.include_router(health.router, tags=["health"])
app.include_router(admin.router)
app.include_router(tester.router)

# Serve UI static files
ui_dir = Path(__file__).parent.parent / "ui"
if ui_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")

# Serve admin dashboard
admin_dir = Path(__file__).parent.parent / "ui" / "admin"
if admin_dir.exists():
    app.mount("/admin-ui", StaticFiles(directory=str(admin_dir), html=True), name="admin-ui")

# Serve tester dashboard
tester_dir = Path(__file__).parent.parent / "ui" / "tester"
if tester_dir.exists():
    app.mount("/tester-ui", StaticFiles(directory=str(tester_dir), html=True), name="tester-ui")


@app.get("/")
async def root():
    return {
        "message": "Harel Insurance Chatbot API",
        "docs": "/docs",
        "ui": "/ui",
        "admin": "/admin-ui",
        "tester": "/tester-ui",
    }
