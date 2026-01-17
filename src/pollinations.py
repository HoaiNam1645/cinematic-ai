"""
Pollinations AI endpoints - FREE image generation
"""
import os
import time
from urllib.parse import quote
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from shared import (
    download_and_upload_to_b2,
    download_and_upload_video_to_b2,
    upload_video_to_b2,
    cache_omnigen,
    cache_video,
    refresh_cache,
    GALLERY_CACHE_TTL,
    B2_FOLDER,
    b2_executor,
    _sync_put_object
)

router = APIRouter(prefix="/api", tags=["pollinations"])

# --- Config ---
POLLINATIONS_API_KEY = os.getenv("POLLINATIONS_API_KEY", "")
POLLINATIONS_API_BASE = "https://gen.pollinations.ai"

POLLINATIONS_QUALITY_BOOSTER = (
    ", masterpiece, best quality, ultra detailed, 8K UHD resolution, "
    "professional photography, sharp focus, intricate details, "
    "cinematic lighting, soft natural bokeh, high dynamic range, "
    "photorealistic, award-winning, trending on artstation"
)

POLLINATIONS_NEGATIVE_PROMPT = (
    "blurry, low quality, bad anatomy, bad hands, distorted, "
    "watermark, text, signature, low resolution, ugly, deformed"
)

POLLINATIONS_EDIT_SUFFIX = (
    ". ISOLATED EDIT ONLY. The original image content must remain immutable except for the specific requested change. "
    "Do not add, remove, or modify any other elements, background, or style. "
    "Highest quality preservation of the source image."
)

PROMPT_OPTIMIZER_SYSTEM = """You are an expert AI image prompt engineer. Your task is to transform simple user descriptions into detailed, optimized prompts for AI image generation.

Rules:
1. Keep the core idea from the user's input
2. Add specific details about: lighting, composition, style, mood, colors
3. Include quality enhancers like: "masterpiece", "highly detailed", "8K resolution"
4. Mention technical terms like: "depth of field", "volumetric lighting", "cinematic"
5. Keep it concise but descriptive (max 200 words)
6. Output ONLY the optimized prompt, nothing else
"""


# --- Request Models ---
class PollinationsRequest(BaseModel):
    prompt: str
    model: str = "flux"
    width: int = 1024
    height: int = 1024


class PollinationsImg2ImgRequest(BaseModel):
    prompt: str
    image_url: str
    model: str = "kontext"
    width: int = 1024
    height: int = 1024
    strength: float = 0.7


class PollinationsTextRequest(BaseModel):
    prompt: str
    model: str = "openai"


class PollinationsVideoRequest(BaseModel):
    """Request model for video generation"""
    prompt: str
    model: str = "seedance"  # veo, seedance, seedance-pro
    duration: int = 4  # seconds: veo (4,6,8), seedance (2-10)
    aspect_ratio: str = "16:9"  # 16:9 or 9:16
    image_url: str = None  # Optional: for image-to-video (seedance only)
    audio: bool = False  # Enable audio (veo only)


# --- Gallery Endpoints ---
@router.get("/gallery/pollinations")
async def get_pollinations_gallery():
    global cache_omnigen
    now = time.time()
    
    if cache_omnigen["data"] and (now - cache_omnigen["timestamp"]) < GALLERY_CACHE_TTL:
        return cache_omnigen["data"]
    
    return await refresh_cache("omnigen")


@router.post("/gallery/refresh/omnigen")
async def refresh_omnigen_gallery():
    count = len(await refresh_cache("omnigen"))
    return {"status": "refreshed", "target": "omnigen", "count": count}


# --- Video Gallery ---
@router.get("/gallery/video")
async def get_video_gallery():
    """Get list of generated videos from B2"""
    global cache_video
    now = time.time()
    
    if cache_video["data"] and (now - cache_video["timestamp"]) < GALLERY_CACHE_TTL:
        print(f"[Video Gallery] Returning cached data: {len(cache_video['data'])} videos")
        return cache_video["data"]
    
    print("[Video Gallery] Cache expired, refreshing from B2...")
    return await refresh_cache("video")


@router.post("/gallery/refresh/video")
async def refresh_video_gallery():
    """Refresh video gallery from B2"""
    count = len(await refresh_cache("video"))
    return {"status": "refreshed", "target": "video", "count": count}


# --- Generation Endpoints ---
@router.get("/pollinations/models")
async def get_pollinations_models():
    """Get available Pollinations models"""
    return {
        "text2img": ["flux", "zimage", "turbo", "gptimage", "seedream", "seedream-pro", "nanobanana", "nanobanana-pro"],
        "img2img": ["kontext", "gptimage", "nanobanana", "nanobanana-pro", "seedance"],
        "text2video": ["veo", "seedance", "seedance-pro"],
        "img2video": ["seedance", "seedance-pro"],
        "text": ["openai", "openai-fast", "openai-large", "qwen-coder", "mistral"]
    }


@router.post("/generate/pollinations/text")
async def generate_pollinations_text(request: PollinationsTextRequest):
    """Optimize prompt using Pollinations Text API"""
    import httpx
    try:
        print(f"[Pollinations Text] Input: {request.prompt[:50]}...")
        print(f"   Model: {request.model}")

        encoded_prompt = quote(request.prompt)
        encoded_system = quote(PROMPT_OPTIMIZER_SYSTEM)
        
        text_url = (
            f"{POLLINATIONS_API_BASE}/text/{encoded_prompt}"
            f"?model={request.model}"
            f"&system={encoded_system}"
            f"&seed={int(time.time())}"
        )
        
        if POLLINATIONS_API_KEY:
            text_url += f"&key={POLLINATIONS_API_KEY}"
        
        headers = {}
        if POLLINATIONS_API_KEY:
            headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(text_url, headers=headers)
            response.raise_for_status()
            optimized_prompt = response.text.strip()
        
        print(f"Optimized: {optimized_prompt[:100]}...")
        
        return {"optimized_prompt": optimized_prompt}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/pollinations")
async def generate_pollinations(request: PollinationsRequest):
    """Generate image using Pollinations AI (FREE)"""
    try:
        print(f"[Pollinations T2I] User Prompt: {request.prompt[:50]}...")
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
            f"&negative={quote(POLLINATIONS_NEGATIVE_PROMPT)}"
        )
        
        if POLLINATIONS_API_KEY:
            image_url += f"&key={POLLINATIONS_API_KEY}"
        
        print(f"Pollinations URL: {image_url[:100]}...")
        
        headers = {}
        if POLLINATIONS_API_KEY:
            headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        
        b2_url = await download_and_upload_to_b2(image_url, headers=headers)
        
        final_url = b2_url if b2_url else image_url
        print(f"Generated: url={final_url[:80]}...")
        
        # Update cache
        if b2_url:
            cache_omnigen["data"].insert(0, {
                "url": b2_url,
                "b2_url": b2_url,
                "key": b2_url, 
                "folder": "omniGen",
                "time": time.time(),
                "source": "pollinations"
            })
        
        return {"url": image_url, "b2_url": final_url}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/pollinations/img2img")
async def generate_pollinations_img2img(request: PollinationsImg2ImgRequest):
    """Edit image using Pollinations AI"""
    import httpx
    try:
        print(f"[Pollinations I2I] Prompt: {request.prompt[:50]}...")
        print(f"   Source: {request.image_url[:80]}...")
        print(f"   Model: {request.model}")

        if not request.image_url.startswith("http"):
            raise HTTPException(400, "Image URL must be a public URL (B2)")

        edit_prompt = f"{request.prompt}{POLLINATIONS_EDIT_SUFFIX}"
        encoded_prompt = quote(edit_prompt)
        encoded_source = quote(request.image_url, safe='')
        
        image_url = (
            f"{POLLINATIONS_API_BASE}/image/{encoded_prompt}"
            f"?model={request.model}"
            f"&image={encoded_source}"
            f"&seed={int(time.time())}"
            f"&nologo=true"
            f"&negative={quote(POLLINATIONS_NEGATIVE_PROMPT)}"
        )
        
        if POLLINATIONS_API_KEY:
            image_url += f"&key={POLLINATIONS_API_KEY}"
        
        print(f"Pollinations I2I URL: {image_url[:150]}...")
        
        headers = {}
        if POLLINATIONS_API_KEY:
            headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        
        b2_url = await download_and_upload_to_b2(image_url, headers=headers)
        
        final_url = b2_url if b2_url else image_url
        print(f"Edited: url={final_url[:80]}...")
        
        # Update cache
        if b2_url:
            cache_omnigen["data"].insert(0, {
                "url": b2_url,
                "b2_url": b2_url,
                "key": b2_url,
                "folder": "omniGen",
                "time": time.time(),
                "source": "pollinations"
            })
        
        return {"url": image_url, "b2_url": final_url}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# --- Video Generation ---
@router.post("/generate/pollinations/video")
async def generate_pollinations_video(request: PollinationsVideoRequest):
    """Generate video using Pollinations AI (veo, seedance)"""
    import httpx
    start_time = time.time()
    
    try:
        print("=" * 60)
        print(f"[VIDEO GEN] === Starting Video Generation ===")
        print(f"[VIDEO GEN] Prompt: {request.prompt[:80]}...")
        print(f"[VIDEO GEN] Model: {request.model}")
        print(f"[VIDEO GEN] Duration: {request.duration}s")
        print(f"[VIDEO GEN] Aspect Ratio: {request.aspect_ratio}")
        print(f"[VIDEO GEN] Audio: {request.audio}")
        print(f"[VIDEO GEN] Image URL: {request.image_url[:60] if request.image_url else 'None'}...")
        print("=" * 60)
        
        encoded_prompt = quote(request.prompt)
        
        # Build video URL
        video_url = (
            f"{POLLINATIONS_API_BASE}/image/{encoded_prompt}"
            f"?model={request.model}"
            f"&duration={request.duration}"
            f"&aspectRatio={request.aspect_ratio}"
            f"&seed={int(time.time())}"
        )
        
        # Add image for image-to-video (supported by seedance, video, luma, kling, etc)
        if request.image_url:
            encoded_image = quote(request.image_url, safe='')
            video_url += f"&image={encoded_image}"
            print(f"[VIDEO GEN] Mode: IMAGE-TO-VIDEO")
            print(f"[VIDEO GEN] Source Image: {request.image_url}")
        else:
            print(f"[VIDEO GEN] Mode: TEXT-TO-VIDEO")
        
        # Add audio option (veo only)
        if request.audio and request.model == "veo":
            video_url += "&audio=true"
            print(f"[VIDEO GEN] Audio generation: ENABLED")
        
        if POLLINATIONS_API_KEY:
            video_url += f"&key={POLLINATIONS_API_KEY}"
            print(f"[VIDEO GEN] API Key: Configured")
        else:
            print(f"[VIDEO GEN] API Key: Not configured (using free tier)")
        
        print(f"[VIDEO GEN] Request URL: {video_url[:150]}...")
        
        headers = {}
        if POLLINATIONS_API_KEY:
            headers["Authorization"] = f"Bearer {POLLINATIONS_API_KEY}"
        
        # Step 1: Request video from Pollinations
        print(f"\n[STEP 1/3] Requesting video from Pollinations API...")
        print(f"[STEP 1/3] This may take 30-120 seconds depending on model and duration...")
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(video_url, headers=headers)
            
            request_duration = time.time() - start_time
            print(f"[STEP 1/3] Response received in {request_duration:.1f}s")
            print(f"[STEP 1/3] Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"[ERROR] Video generation failed!")
                print(f"[ERROR] Response: {response.text[:300]}")
                raise HTTPException(response.status_code, f"Video generation failed: {response.text[:200]}")
            
            video_data = response.content
            print(f"[STEP 1/3] ✅ Video received: {len(video_data):,} bytes ({len(video_data)/1024/1024:.2f} MB)")
        
        # Step 2: Upload to B2 (using video_data already in memory)
        print(f"\n[STEP 2/3] Uploading video to B2 (folder: video/)...")
        upload_start = time.time()
        
        b2_url = await upload_video_to_b2(video_data)  # Use data already downloaded!
        
        upload_duration = time.time() - upload_start
        
        if b2_url:
            print(f"[STEP 2/3] ✅ Upload successful in {upload_duration:.1f}s")
            print(f"[STEP 2/3] B2 URL: {b2_url}")
            final_url = b2_url
        else:
            print(f"[STEP 2/3] ⚠️ Upload failed, using Pollinations URL as fallback")
            final_url = video_url
        
        # Step 3: Update cache
        print(f"\n[STEP 3/3] Updating video cache...")
        if b2_url:
            cache_video["data"].insert(0, {
                "url": b2_url,
                "b2_url": b2_url,
                "key": b2_url,
                "folder": "video",
                "time": time.time(),
                "type": "video",
                "model": request.model,
                "duration": request.duration
            })
            print(f"[STEP 3/3] ✅ Cache updated, total videos: {len(cache_video['data'])}")
        
        total_duration = time.time() - start_time
        print(f"\n{'=' * 60}")
        print(f"[VIDEO GEN] === Video Generation Complete ===")
        print(f"[VIDEO GEN] Total time: {total_duration:.1f}s")
        print(f"[VIDEO GEN] Final URL: {final_url[:80]}...")
        print(f"{'=' * 60}\n")
        
        return {
            "url": video_url,
            "b2_url": final_url,
            "type": "video",
            "model": request.model,
            "duration": request.duration
        }

    except httpx.TimeoutException:
        elapsed = time.time() - start_time
        print(f"[ERROR] Video generation timeout after {elapsed:.1f}s (max 5 minutes)")
        raise HTTPException(504, "Video generation timed out. Try shorter duration or simpler prompt.")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] Video generation failed after {elapsed:.1f}s: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


