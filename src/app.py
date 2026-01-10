"""
Cinematic AI - Multi Model Generator
Main application entry point
"""
import os
import time
import uuid
import asyncio
import replicate
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import routers
from .pollinations import router as pollinations_router
from .apiframe import router as apiframe_router
from .shared import (
    s3_client, 
    b2_executor, 
    _sync_put_object,
    B2_FOLDER,
    cache_omnigen,
    cache_apiframe,
    refresh_cache,
    GALLERY_CACHE_TTL,
    download_and_upload_to_b2
)

# Initialize FastAPI
app = FastAPI(title="Cinematic AI - Multi Model Generator")

# Check API Keys
if not os.getenv("REPLICATE_API_TOKEN"):
    print("WARNING: REPLICATE_API_TOKEN is not set in .env")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pollinations_router)
app.include_router(apiframe_router)

# Setup directories
BASE_DIR = Path(__file__).parent.parent
GALLERY_DIR = BASE_DIR / "gallery"
UPLOADS_DIR = BASE_DIR / "uploads"
static_dir = BASE_DIR / "static"

for d in [GALLERY_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- Request Models for legacy endpoints ---
class ImagenRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"

class IdeogramRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"
    style_type: str = "None"
    magic_prompt_option: str = "Auto"


# Quality boosters
IMAGEN_QUALITY_BOOSTER = ", stunning quality, highly detailed, 8k resolution, sharp focus, professional image, cinematic lighting"
IDEOGRAM_QUALITY_BOOSTER = ", high quality, aesthetic, masterpiece, professional design"


# --- Upload Endpoint ---
@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload image directly to B2"""
    try:
        content = await file.read()
        
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        key = f"{B2_FOLDER}/{filename}"
        
        loop = asyncio.get_event_loop()
        b2_url = await asyncio.wait_for(
            loop.run_in_executor(b2_executor, _sync_put_object, content, key),
            timeout=60.0
        )
        
        if b2_url:
            cache_omnigen["data"].insert(0, {
                "url": b2_url,
                "b2_url": b2_url,
                "time": time.time()
            })
            
        print(f"Uploaded to B2: {b2_url}")
        
        return {"url": b2_url, "b2_url": b2_url}
    
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(500, str(e))


# --- Legacy Gallery Endpoint ---
@app.get("/api/gallery/{model_type}")
async def get_gallery_legacy(model_type: str):
    """Legacy gallery endpoint for compatibility"""
    if model_type == "pollinations":
        now = time.time()
        if cache_omnigen["data"] and (now - cache_omnigen["timestamp"]) < GALLERY_CACHE_TTL:
            return cache_omnigen["data"]
        return await refresh_cache("omnigen")
    elif model_type == "apiframe":
        now = time.time()
        if cache_apiframe["data"] and (now - cache_apiframe["timestamp"]) < GALLERY_CACHE_TTL:
            return cache_apiframe["data"]
        return await refresh_cache("apiframe")
    return []


# --- Replicate Endpoints (Imagen, Ideogram via Replicate) ---
@app.post("/api/generate/imagen")
async def generate_imagen(request: ImagenRequest):
    """Generate using Google Imagen 4"""
    try:
        print(f"[Imagen 4] User Prompt: {request.prompt[:50]}...")
        
        final_prompt = f"{request.prompt}{IMAGEN_QUALITY_BOOSTER}"
        
        output = replicate.run(
            "google/imagen-4",
            input={
                "prompt": final_prompt,
                "aspect_ratio": request.aspect_ratio,
                "safety_filter_level": "block_medium_and_above"
            }
        )
        
        image_url = str(output[0]) if isinstance(output, list) else str(output)
        print(f"Generated: {image_url}")
        
        b2_url = await download_and_upload_to_b2(image_url)
        
        return {"url": image_url, "b2_url": b2_url}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/ideogram")
async def generate_ideogram(request: IdeogramRequest):
    """Generate using Ideogram v3"""
    try:
        print(f"[Ideogram v3] User Prompt: {request.prompt[:50]}...")

        final_prompt = f"{request.prompt}{IDEOGRAM_QUALITY_BOOSTER}"

        output = replicate.run(
            "ideogram-ai/ideogram-v3-turbo",
            input={
                "prompt": final_prompt,
                "aspect_ratio": request.aspect_ratio,
                "style_type": request.style_type,
                "magic_prompt_option": request.magic_prompt_option,
                "resolution": "None" 
            }
        )
        
        image_url = str(output)
        print(f"Generated: {image_url}")
        
        b2_url = await download_and_upload_to_b2(image_url)
        
        return {"url": image_url, "b2_url": b2_url}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Static Files ---
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/gallery", StaticFiles(directory=str(GALLERY_DIR)), name="gallery")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")


# --- Page Routes ---
@app.get("/")
async def root():
    return FileResponse(static_dir / "index.html")

@app.get("/pollinations")
async def page_pollinations():
    return FileResponse(static_dir / "pollinations.html")

@app.get("/apiframe")
async def page_apiframe():
    return FileResponse(static_dir / "apiframe.html")

@app.get("/video")
async def page_video():
    return FileResponse(static_dir / "video.html")
