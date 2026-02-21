"""FastAPI application factory."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from api.dependencies import resources
from api.routes import admin, chat, explorer, health, tester


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize resources on startup, clean up on shutdown."""
    logger.info("Starting Harel Insurance Chatbot API...")
    resources.initialize()
    logger.info("All resources initialized")
    logger.info(
        "\n"
        "╔══════════════════════════════════════════════════╗\n"
        "║           Harel Insurance Chatbot API            ║\n"
        "╠══════════════════════════════════════════════════╣\n"
        "║  Chat UI:    http://localhost:8000/ui            ║\n"
        "║  Admin:      http://localhost:8000/admin-ui      ║\n"
        "║  Explorer:   http://localhost:8000/explorer-ui   ║\n"
        "║  Tester:     http://localhost:8000/tester-ui     ║\n"
        "║  API Docs:   http://localhost:8000/docs          ║\n"
        "║  Health:     http://localhost:8000/health         ║\n"
        "╚══════════════════════════════════════════════════╝"
    )
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
app.include_router(explorer.router)

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

# Serve explorer dashboard
explorer_dir = Path(__file__).parent.parent / "ui" / "explorer"
if explorer_dir.exists():
    app.mount("/explorer-ui", StaticFiles(directory=str(explorer_dir), html=True), name="explorer-ui")


@app.get("/")
async def root():
    return {
        "message": "Harel Insurance Chatbot API",
        "docs": "/docs",
        "ui": "/ui",
        "admin": "/admin-ui",
        "explorer": "/explorer-ui",
        "tester": "/tester-ui",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
