"""
Kling AI Video Generation Integration
API Docs: https://docs.klingai.com/
"""
import os
import time
import jwt
import httpx
import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from dotenv import load_dotenv

from shared import upload_video_to_b2, cache_video, cache_kling, refresh_cache, GALLERY_CACHE_TTL

load_dotenv()

router = APIRouter(prefix="/api")

# Kling API Configuration
KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY")
KLING_API_BASE = "https://api.klingai.com"


def generate_kling_token() -> str:
    """Generate JWT token for Kling API authentication"""
    if not KLING_ACCESS_KEY or not KLING_SECRET_KEY:
        raise ValueError("Kling API keys not configured")
    
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    
    payload = {
        "iss": KLING_ACCESS_KEY,
        "exp": int(time.time()) + 1800,  # 30 minutes
        "nbf": int(time.time()) - 5
    }
    
    token = jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256", headers=headers)
    return token


class KlingVideoRequest(BaseModel):
    prompt: str
    # Supported models: kling-v1, kling-v1-5, kling-v1-6, kling-v2-master, kling-v2-1, kling-v2-1-master, kling-v2-5-turbo, kling-v2-6
    model: str = "kling-v1-6"
    mode: str = "std"  # std (standard) or pro (professional)
    duration: str = "5"  # "5" or "10" seconds
    aspect_ratio: str = "16:9"
    image_url: Optional[str] = None  # For image-to-video
    negative_prompt: Optional[str] = None
    cfg_scale: float = 0.5


class KlingMultiImageRequest(BaseModel):
    """Request model for Multi-Image to Video (Elements)"""
    prompt: str
    image_urls: List[str]  # List of image URLs (up to 4)
    model: str = "kling-v1-6"  # Only kling-v1-6 is supported
    mode: str = "std"  # std or pro
    duration: str = "5"  # "5" or "10"
    aspect_ratio: str = "16:9"  # 16:9, 9:16, 1:1
    negative_prompt: Optional[str] = None


# --- Kling Gallery Endpoints ---
@router.get("/gallery/kling")
async def get_kling_gallery():
    """Get list of Kling-generated videos from B2"""
    global cache_kling
    now = time.time()
    
    if cache_kling["data"] and (now - cache_kling["timestamp"]) < GALLERY_CACHE_TTL:
        print(f"[Kling Gallery] Returning cached data: {len(cache_kling['data'])} videos")
        return cache_kling["data"]
    
    print("[Kling Gallery] Cache expired, refreshing from B2...")
    return await refresh_cache("kling")


@router.post("/gallery/refresh/kling")
async def refresh_kling_gallery():
    """Refresh Kling video gallery from B2"""
    count = len(await refresh_cache("kling"))
    return {"status": "refreshed", "target": "kling", "count": count}


@router.post("/generate/kling/video")
async def generate_kling_video(request: KlingVideoRequest):
    """Generate video using Kling AI API"""
    
    if not KLING_ACCESS_KEY or not KLING_SECRET_KEY:
        raise HTTPException(400, "Kling AI API keys not configured. Add KLING_ACCESS_KEY and KLING_SECRET_KEY to .env")
    
    start_time = time.time()
    
    try:
        print("=" * 60)
        print(f"[KLING] === Starting Kling Video Generation ===")
        print(f"[KLING] Prompt: {request.prompt[:80]}...")
        print(f"[KLING] Model: {request.model}")
        print(f"[KLING] Mode: {request.mode}")
        print(f"[KLING] Duration: {request.duration}s")
        print(f"[KLING] Aspect Ratio: {request.aspect_ratio}")
        print(f"[KLING] Image URL: {request.image_url[:60] if request.image_url else 'None'}...")
        print("=" * 60)
        
        # Generate JWT token
        token = generate_kling_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        # Determine if text-to-video or image-to-video
        is_v2 = request.model.startswith("kling-v2")
        is_master = "master" in request.model  # v2-master, v2-1-master
        is_v26 = request.model == "kling-v2-6"
        
        # For master models, mode should be "master" instead of std/pro
        effective_mode = "master" if is_master else request.mode
        
        if request.image_url:
            endpoint = f"{KLING_API_BASE}/v1/videos/image2video"
            payload = {
                "model_name": request.model,
                "mode": effective_mode,
                "duration": request.duration,
                "image": request.image_url,
                "prompt": request.prompt
            }
            print(f"[KLING] Mode: IMAGE-TO-VIDEO")
        else:
            endpoint = f"{KLING_API_BASE}/v1/videos/text2video"
            payload = {
                "model_name": request.model,
                "mode": effective_mode,
                "duration": request.duration,
                "aspect_ratio": request.aspect_ratio,
                "prompt": request.prompt
            }
            print(f"[KLING] Mode: TEXT-TO-VIDEO")
        
        # cfg_scale only supported for v1.x models
        if not is_v2 and request.cfg_scale is not None:
            payload["cfg_scale"] = request.cfg_scale
        
        # negative_prompt not supported for v2.5 models
        if request.negative_prompt and "v2-5" not in request.model:
            payload["negative_prompt"] = request.negative_prompt
        
        # sound parameter only for v2.6+
        if is_v26:
            payload["sound"] = "off"  # Default off, can be extended to support "on"
        
        # Step 1: Submit task
        print(f"\n[STEP 1/3] Submitting task to Kling API...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            
            if response.status_code != 200:
                print(f"[ERROR] Kling API error: {response.text}")
                raise HTTPException(response.status_code, f"Kling API error: {response.text}")
            
            result = response.json()
            
            if result.get("code") != 0:
                raise HTTPException(400, f"Kling error: {result.get('message')}")
            
            task_id = result["data"]["task_id"]
            print(f"[STEP 1/3] ✅ Task created: {task_id}")
        
        # Step 2: Poll for completion
        print(f"\n[STEP 2/3] Waiting for video generation...")
        video_url = None
        max_attempts = 120  # 10 minutes max
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Check every 5 seconds
                
                query_url = f"{KLING_API_BASE}/v1/videos/{'image2video' if request.image_url else 'text2video'}/{task_id}"
                response = await client.get(query_url, headers=headers)
                
                if response.status_code != 200:
                    continue
                
                result = response.json()
                status = result.get("data", {}).get("task_status")
                
                if status == "succeed":
                    videos = result.get("data", {}).get("task_result", {}).get("videos", [])
                    if videos:
                        video_url = videos[0].get("url")
                        print(f"[STEP 2/3] ✅ Video ready after {attempt * 5}s")
                        break
                elif status == "failed":
                    error_msg = result.get("data", {}).get("task_status_msg", "Unknown error")
                    raise HTTPException(500, f"Kling generation failed: {error_msg}")
                else:
                    if attempt % 6 == 0:  # Log every 30s
                        print(f"[STEP 2/3] Status: {status}... ({attempt * 5}s)")
        
        if not video_url:
            raise HTTPException(500, "Video generation timed out")
        
        # Step 3: Download and upload to B2
        print(f"\n[STEP 3/3] Downloading and uploading to B2...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            video_response = await client.get(video_url)
            if video_response.status_code == 200:
                video_data = video_response.content
                print(f"[STEP 3/3] Downloaded {len(video_data):,} bytes")
                
                b2_url = await upload_video_to_b2(video_data, folder="kling_video")
                
                if b2_url:
                    print(f"[STEP 3/3] ✅ Uploaded to B2: {b2_url}")
                    
                    # Update kling cache
                    cache_kling["data"].insert(0, {
                        "url": b2_url,
                        "b2_url": b2_url,
                        "time": time.time(),
                        "source": "kling"
                    })
                    
                    total_time = time.time() - start_time
                    print(f"\n[KLING] ✅ Complete in {total_time:.1f}s")
                    
                    return {
                        "url": b2_url,
                        "b2_url": b2_url,
                        "source": "kling",
                        "task_id": task_id
                    }
        
        # Fallback to original URL
        return {
            "url": video_url,
            "b2_url": video_url,
            "source": "kling",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Kling error: {str(e)}")
        raise HTTPException(500, str(e))


@router.post("/generate/kling/multi-image")
async def generate_kling_multi_image_video(request: KlingMultiImageRequest):
    """Generate video from multiple images using Kling AI API (Elements)"""
    try:
        # Validate image count
        if len(request.image_urls) < 1 or len(request.image_urls) > 4:
            raise HTTPException(400, "Must provide 1-4 images")
        
        print("=" * 60)
        print(f"[KLING] === Multi-Image to Video Generation ===")
        print(f"[KLING] Prompt: {request.prompt[:80]}...")
        print(f"[KLING] Images: {len(request.image_urls)} images")
        print(f"[KLING] Mode: {request.mode}")
        print(f"[KLING] Duration: {request.duration}s")
        print("=" * 60)
        
        # Generate JWT token
        token = generate_kling_token()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
        
        # Build image_list for API
        image_list = [{"image": url} for url in request.image_urls]
        
        endpoint = f"{KLING_API_BASE}/v1/videos/multi-image2video"
        payload = {
            "model_name": "kling-v1-6",  # Only v1.6 supported
            "mode": request.mode,
            "duration": request.duration,
            "aspect_ratio": request.aspect_ratio,
            "prompt": request.prompt,
            "image_list": image_list
        }
        
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt
        
        # Step 1: Submit task
        print(f"\n[STEP 1/3] Submitting multi-image task to Kling API...")
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(endpoint, json=payload, headers=headers)
            
            if response.status_code != 200:
                print(f"[ERROR] Kling API error: {response.text}")
                raise HTTPException(response.status_code, f"Kling API error: {response.text}")
            
            result = response.json()
            
            if result.get("code") != 0:
                raise HTTPException(400, f"Kling error: {result.get('message')}")
            
            task_id = result.get("data", {}).get("task_id")
            print(f"[KLING] Task submitted: {task_id}")
        
        # Step 2: Poll for completion
        print(f"\n[STEP 2/3] Polling for completion...")
        query_endpoint = f"{KLING_API_BASE}/v1/videos/multi-image2video/{task_id}"
        max_attempts = 120
        poll_interval = 5
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(max_attempts):
                await asyncio.sleep(poll_interval)
                
                response = await client.get(query_endpoint, headers=headers)
                if response.status_code != 200:
                    continue
                
                result = response.json()
                task_status = result.get("data", {}).get("task_status")
                
                if task_status == "succeed":
                    videos = result.get("data", {}).get("task_result", {}).get("videos", [])
                    if videos:
                        video_url = videos[0].get("url")
                        print(f"[KLING] ✅ Video ready: {video_url[:60]}...")
                        break
                elif task_status == "failed":
                    error_msg = result.get("data", {}).get("task_status_msg", "Unknown error")
                    raise HTTPException(400, f"Video generation failed: {error_msg}")
                
                print(f"[KLING] Status: {task_status} (attempt {attempt + 1}/{max_attempts})")
            else:
                raise HTTPException(408, "Video generation timed out")
        
        # Step 3: Upload to B2
        print(f"\n[STEP 3/3] Uploading to B2 Storage...")
        async with httpx.AsyncClient(timeout=120.0) as client:
            video_response = await client.get(video_url)
            if video_response.status_code == 200:
                video_data = video_response.content
                print(f"[STEP 3/3] Downloaded {len(video_data):,} bytes")
                b2_url = await upload_video_to_b2(video_data, folder="kling_video")
                if b2_url:
                    print(f"[KLING] ✅ Multi-image video complete!")
                    
                    # Update kling cache
                    cache_kling["data"].insert(0, {
                        "url": b2_url,
                        "b2_url": b2_url,
                        "time": time.time(),
                        "source": "kling-multi"
                    })
                    
                    return {
                        "url": b2_url,
                        "b2_url": b2_url,
                        "source": "kling-multi",
                        "task_id": task_id
                    }
        
        return {
            "url": video_url,
            "b2_url": video_url,
            "source": "kling-multi",
            "task_id": task_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Kling multi-image error: {str(e)}")
        raise HTTPException(500, str(e))


@router.get("/kling/status")
async def check_kling_status():
    """Check if Kling API is configured"""
    return {
        "configured": bool(KLING_ACCESS_KEY and KLING_SECRET_KEY),
        "has_access_key": bool(KLING_ACCESS_KEY),
        "has_secret_key": bool(KLING_SECRET_KEY)
    }

