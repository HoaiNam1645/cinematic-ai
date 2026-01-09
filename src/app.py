import os
import time
import uuid
import httpx
import replicate
import boto3
from botocore.config import Config
from urllib.parse import quote
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

app = FastAPI(title="Cinematic AI - Multi Model Generator")

# Check API Keys
if not os.getenv("REPLICATE_API_TOKEN"):
    print("WARNING: REPLICATE_API_TOKEN is not set in .env")

POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "")

# Backblaze B2 Configuration
B2_ACCESS_KEY_ID = os.getenv("B2_ACCESS_KEY_ID", "")
B2_SECRET_ACCESS_KEY = os.getenv("B2_SECRET_ACCESS_KEY", "")
B2_BUCKET = os.getenv("B2_BUCKET", "cinematic-ai")
B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
B2_URL_CLOUD = os.getenv("B2_URL_CLOUD", "https://zipimgs.com/file/Lemiex-Fulfillment")
B2_FOLDER = "omniGen"  # Folder for Pollinations images

# Initialize B2 client
s3_client = None
if B2_ACCESS_KEY_ID and B2_SECRET_ACCESS_KEY:
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_ACCESS_KEY_ID,
            aws_secret_access_key=B2_SECRET_ACCESS_KEY,
            config=Config(signature_version='s3v4')
        )
        print(f"✅ B2 client initialized: {B2_BUCKET}/{B2_FOLDER}")
    except Exception as e:
        print(f"⚠️ Failed to initialize B2 client: {e}")
else:
    print("⚠️ B2 credentials not configured - images won't be uploaded to cloud")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Gallery Directories
BASE_DIR = Path(__file__).parent.parent
GALLERY_DIR = BASE_DIR / "gallery"
UPLOADS_DIR = BASE_DIR / "uploads"
IMAGEN_DIR = GALLERY_DIR / "google-imagen4"
IDEOGRAM_DIR = GALLERY_DIR / "ideogramv3"
POLLINATIONS_DIR = GALLERY_DIR / "pollinations"

for d in [IMAGEN_DIR, IDEOGRAM_DIR, POLLINATIONS_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# --- Helper Functions ---
async def upload_to_b2(file_path: Path, filename: str) -> str:
    """Upload a file to Backblaze B2 and return public URL"""
    if not s3_client:
        return ""
    
    try:
        key = f"{B2_FOLDER}/{filename}"
        
        # Upload file
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(
                f, 
                B2_BUCKET, 
                key,
                ExtraArgs={'ContentType': 'image/png'}
            )
        
        # Build public URL
        public_url = f"{B2_URL_CLOUD}/{B2_FOLDER}/{filename}"
        print(f"☁️ Uploaded to B2: {public_url}")
        return public_url
        
    except Exception as e:
        print(f"⚠️ B2 upload failed: {e}")
        return ""


async def save_image_locally(url: str, folder: Path, headers: dict = None) -> tuple[str, str]:
    """Download image URL to local folder and upload to B2. Returns (local_url, b2_url)"""
    try:
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        filepath = folder / filename
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(url, headers=headers or {})
            response.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(response.content)
                
        print(f"💾 Saved locally: {filepath}")
        model_name = folder.name
        local_url = f"/gallery/{model_name}/{filename}"
        
        # Upload to B2
        b2_url = await upload_to_b2(filepath, filename)
        
        return local_url, b2_url
        
    except Exception as e:
        print(f"⚠️ Failed to save: {e}")
        return url, ""


# --- Request Models ---
class ImagenRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"

class IdeogramRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "16:9"
    style_type: str = "None"
    magic_prompt_option: str = "Auto"

class PollinationsRequest(BaseModel):
    prompt: str
    model: str = "flux"
    width: int = 1024
    height: int = 1024

class PollinationsImg2ImgRequest(BaseModel):
    prompt: str
    image_url: str  # B2 public URL of source image
    model: str = "kontext"
    width: int = 1024
    height: int = 1024
    strength: float = 0.7

# --- Quality Assurance System Prompts ---
IMAGEN_QUALITY_BOOSTER = ", stunning quality, highly detailed, 8k resolution, sharp focus, professional image, cinematic lighting"
IDEOGRAM_QUALITY_BOOSTER = ", high quality, aesthetic, masterpiece, professional design"

POLLINATIONS_QUALITY_BOOSTER = (
    ", masterpiece, best quality, ultra detailed, 8K UHD resolution, "
    "professional photography, sharp focus, intricate details, "
    "cinematic lighting, soft natural bokeh, high dynamic range, "
    "photorealistic, award-winning, trending on artstation"
)

POLLINATIONS_API_BASE = "https://gen.pollinations.ai"


# --- Upload Endpoint ---

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to local storage and B2"""
    try:
        ext = file.filename.split('.')[-1] if '.' in file.filename else 'png'
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.{ext}"
        filepath = UPLOADS_DIR / filename
        
        content = await file.read()
        with open(filepath, "wb") as f:
            f.write(content)
        
        print(f"📤 Uploaded: {filepath}")
        
        # Also upload to B2
        b2_url = await upload_to_b2(filepath, filename)
        
        return {
            "url": f"/uploads/{filename}", 
            "filename": filename,
            "b2_url": b2_url
        }
    
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(500, str(e))


# --- Gallery Routes ---

@app.get("/api/gallery/{model_type}")
async def get_gallery(model_type: str):
    """Get list of files for a specific model"""
    valid_types = ["google-imagen4", "ideogramv3", "pollinations"]
    if model_type not in valid_types:
        raise HTTPException(400, f"Invalid model type. Must be one of: {valid_types}")
    
    target_dir = GALLERY_DIR / model_type
    if not target_dir.exists():
        return []
    
    files = []
    for f in target_dir.glob("*.png"):
        # Build B2 URL for each file
        b2_url = f"{B2_URL_CLOUD}/{B2_FOLDER}/{f.name}" if B2_URL_CLOUD else ""
        files.append({
            "url": f"/gallery/{model_type}/{f.name}",
            "b2_url": b2_url,
            "time": f.stat().st_mtime
        })
    
    files.sort(key=lambda x: x["time"], reverse=True)
    return files


# --- Image Generation Routes ---

@app.post("/api/generate/imagen")
async def generate_imagen(request: ImagenRequest):
    """Generate using Google Imagen 4"""
    try:
        print(f"🎨 [Imagen 4] User Prompt: {request.prompt[:50]}...")
        
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
        print(f"✅ Generated: {image_url}")
        
        local_url, b2_url = await save_image_locally(image_url, folder=IMAGEN_DIR)
        
        return {"url": image_url, "local_url": local_url, "b2_url": b2_url}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/ideogram")
async def generate_ideogram(request: IdeogramRequest):
    """Generate using Ideogram v3"""
    try:
        print(f"🎨 [Ideogram v3] User Prompt: {request.prompt[:50]}...")

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
        print(f"✅ Generated: {image_url}")
        
        local_url, b2_url = await save_image_locally(image_url, folder=IDEOGRAM_DIR)
        
        return {"url": image_url, "local_url": local_url, "b2_url": b2_url}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Pollinations AI (FREE) ---

@app.get("/api/pollinations/models")
async def get_pollinations_models():
    """Get available Pollinations models"""
    return {
        "text2img": ["flux", "turbo", "zimage", "seedream", "gptimage"],
        "img2img": ["kontext", "gptimage"]
    }


@app.post("/api/generate/pollinations")
async def generate_pollinations(request: PollinationsRequest):
    """Generate image using Pollinations AI (FREE)"""
    try:
        print(f"🌸 [Pollinations T2I] User Prompt: {request.prompt[:50]}...")
        print(f"   Model: {request.model}, Size: {request.width}x{request.height}")

        final_prompt = f"{request.prompt}{POLLINATIONS_QUALITY_BOOSTER}"
        encoded_prompt = quote(final_prompt)
        
        image_url = (
            f"{POLLINATIONS_API_BASE}/image/{encoded_prompt}"
            f"?model={request.model}"
            f"&width={request.width}"
            f"&height={request.height}"
            f"&seed={int(time.time())}"
            f"&enhance=true"
            f"&nologo=true"
        )
        
        if POLLINATIONS_API_KEY:
            image_url += f"&key={POLLINATIONS_API_KEY}"
        
        print(f"🔗 Pollinations URL: {image_url[:100]}...")
        
        headers = {}
        if POLLINATIONS_API_KEY:
            headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        
        local_url, b2_url = await save_image_locally(image_url, folder=POLLINATIONS_DIR, headers=headers)
        
        print(f"✅ Generated: local={local_url}, b2={b2_url}")
        
        return {"url": image_url, "local_url": local_url, "b2_url": b2_url}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate/pollinations/img2img")
async def generate_pollinations_img2img(request: PollinationsImg2ImgRequest):
    """Edit image using Pollinations AI - requires B2 public URL"""
    try:
        print(f"🖌️ [Pollinations I2I] Prompt: {request.prompt[:50]}...")
        print(f"   Source: {request.image_url[:80]}...")
        print(f"   Model: {request.model}")

        # Use the B2 URL directly (should be public)
        if not request.image_url.startswith("http"):
            raise HTTPException(400, "Image URL must be a public URL (B2)")

        edit_prompt = f"{request.prompt}, preserve original composition and style"
        encoded_prompt = quote(edit_prompt)
        encoded_source = quote(request.image_url, safe='')
        
        image_url = (
            f"{POLLINATIONS_API_BASE}/image/{encoded_prompt}"
            f"?model={request.model}"
            f"&image={encoded_source}"
            f"&seed={int(time.time())}"
            f"&nologo=true"
        )
        
        if POLLINATIONS_API_KEY:
            image_url += f"&key={POLLINATIONS_API_KEY}"
        
        print(f"🔗 Pollinations I2I URL: {image_url[:150]}...")
        
        headers = {}
        if POLLINATIONS_API_KEY:
            headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        
        local_url, b2_url = await save_image_locally(image_url, folder=POLLINATIONS_DIR, headers=headers)
        
        print(f"✅ Edited: local={local_url}, b2={b2_url}")
        
        return {"url": image_url, "local_url": local_url, "b2_url": b2_url}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Static Files Handling ---
static_dir = BASE_DIR / "static"

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/gallery", StaticFiles(directory=str(GALLERY_DIR)), name="gallery")
app.mount("/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="uploads")

@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))

@app.get("/imagen")
async def page_imagen():
    return FileResponse(str(static_dir / "imagen.html"))

@app.get("/ideogram")
async def page_ideogram():
    return FileResponse(str(static_dir / "ideogram.html"))

@app.get("/pollinations")
async def page_pollinations():
    return FileResponse(str(static_dir / "pollinations.html"))
