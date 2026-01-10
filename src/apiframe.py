"""
APIFrame Pro endpoints - Ideogram, Flux, Nano Banana
"""
import os
import time
import asyncio
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .shared import (
    download_and_upload_to_b2, 
    cache_apiframe, 
    refresh_cache,
    GALLERY_CACHE_TTL
)

router = APIRouter(prefix="/api", tags=["apiframe"])


# --- Request Model ---
class APIFrameRequest(BaseModel):
    prompt: str
    model: str = "ideogram"  # ideogram, flux, nano-banana
    source_image: Optional[str] = None


# --- Gallery Endpoints ---
@router.get("/gallery/apiframe")
async def get_apiframe_gallery():
    global cache_apiframe
    now = time.time()
    
    if cache_apiframe["data"] and (now - cache_apiframe["timestamp"]) < GALLERY_CACHE_TTL:
        return cache_apiframe["data"]
    
    return await refresh_cache("apiframe")


@router.post("/gallery/refresh/apiframe")
async def refresh_apiframe_gallery():
    count = len(await refresh_cache("apiframe"))
    return {"status": "refreshed", "target": "apiframe", "count": count}


# --- Generation Endpoint ---
@router.post("/generate/apiframe")
async def generate_apiframe(request: APIFrameRequest):
    """Generate image using APIFrame (Ideogram/Flux/Nano)"""
    api_key = os.getenv("APIFRAME_API_KEY")
    if not api_key:
        raise HTTPException(500, "APIFRAME_API_KEY not configured")
        
    headers = {"Content-Type": "application/json", "Authorization": api_key}
    
    # 1. Select Endpoint & Logic
    is_sync = False
    
    if request.model == "flux":
        submit_url = "https://api.apiframe.pro/flux-imagine"
        payload = {
            "prompt": request.prompt,
            "model": "flux-pro"
        }
    elif request.model == "nano-banana":
        submit_url = "https://api.apiframe.pro/nano-banana"
        images_payload = [request.source_image] if request.source_image else []
        payload = {
            "prompt": request.prompt,
            "images": images_payload,
            "aspect_ratio": "match_input_image"
        }
        is_sync = True
    else:
        # Default to Ideogram
        submit_url = "https://api.apiframe.pro/ideogram-imagine"
        payload = {
            "prompt": request.prompt,
            "aspect_ratio": "ASPECT_1_1"
        }
    
    print(f"[APIFrame] Generating with {request.model}: {request.prompt[:50]}...")
    print(f"[APIFrame] Payload: {payload}")

    # 2. Submit Task
    task_id = None
    final_image_url = None

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(submit_url, json=payload, headers=headers)
            if resp.status_code != 200:
                print(f"APIFrame Error ({request.model}): {resp.text}")
            resp.raise_for_status()
            data = resp.json()
            
            if is_sync:
                if "image_urls" in data and data["image_urls"]:
                    final_image_url = data["image_urls"][0]
                    print(f"   Success (Sync)! URL: {final_image_url[:60]}...")
                else:
                    raise HTTPException(500, f"No images returned from {request.model}")
            else:
                task_id = data.get("task_id")
                print(f"   Task ID: {task_id}")
            
    except Exception as e:
        print(f"APIFrame Submit Error: {e}")
        detail = str(e)
        if 'resp' in locals() and hasattr(resp, 'text'):
            detail += f" | Body: {resp.text}"
        raise HTTPException(500, f"Submit failed: {detail}")

    # 3. Polling Loop (Only if not sync)
    if task_id:
        fetch_url = "https://api.apiframe.pro/fetch"
        async with httpx.AsyncClient(timeout=120.0) as client:
            for i in range(20): 
                await asyncio.sleep(3)
                try:
                    resp = await client.post(fetch_url, json={"task_id": task_id}, headers=headers)
                    
                    if resp.status_code != 200:
                         print(f"Polling non-200: {resp.status_code}")
                         continue
                         
                    state = resp.json()
                    if not state:
                        print("Polling received empty JSON")
                        continue

                    status = state.get("status")
                    print(f"   Status: {status}")
                    
                    if status in ["finished", "completed", "succeeded"]:
                        if "image_urls" in state and isinstance(state["image_urls"], list) and len(state["image_urls"]) > 0:
                            final_image_url = state["image_urls"][0]
                        elif "image_url" in state and state["image_url"]:
                            final_image_url = state["image_url"]
                        elif "output" in state and state["output"]:
                             out = state["output"]
                             final_image_url = out[0] if isinstance(out, list) else out
                        
                        if final_image_url:
                            print(f"   Success (Polled)! URL: {final_image_url[:60]}...")
                            break
                        else:
                            print(f"   Finished but no URL found: {state}")
                            
                    elif status == "failed":
                        print(f"   Task failed: {state}")
                        raise HTTPException(500, f"Generation failed: {state.get('error', 'Unknown')}")
                        
                except Exception as e:
                    print(f"Polling error: {e}")
                    continue
    
    if not final_image_url:
        raise HTTPException(504, "Generation timed out")

    # 4. Upload to B2 (apiFrame folder)
    try:
        b2_url = await download_and_upload_to_b2(final_image_url, subfolder="apiFrame")
        final_url = b2_url if b2_url else final_image_url
        
        # Update Cache
        if b2_url:
            cache_apiframe["data"].insert(0, {
                "url": b2_url,
                "b2_url": b2_url,
                "key": b2_url,
                "folder": "apiFrame",
                "time": time.time(),
                "source": "apiFrame"
            })
            
        return {"url": final_image_url, "b2_url": final_url}
        
    except Exception as e:
        print(f"Upload/Cache Error: {e}")
        return {"url": final_image_url, "b2_url": final_image_url}
