from fastapi import APIRouter
from app.api.routes import auth, chatbots, files, scraping, qa

# Create main API router
api_router = APIRouter(prefix="/api")

# Include all route modules
api_router.include_router(auth.router)
api_router.include_router(chatbots.router)
api_router.include_router(files.router)
api_router.include_router(scraping.router)
api_router.include_router(qa.router)
