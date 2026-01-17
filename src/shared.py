"""
Shared utilities, caches, and B2 functions for Cinematic AI
"""
import os
import time
import uuid
import httpx
import asyncio
import boto3
from botocore.config import Config
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Backblaze B2 Configuration
B2_ACCESS_KEY_ID = os.getenv("B2_ACCESS_KEY_ID", "")
B2_SECRET_ACCESS_KEY = os.getenv("B2_SECRET_ACCESS_KEY", "")
B2_BUCKET = os.getenv("B2_BUCKET", "cinematic-ai")
B2_ENDPOINT = os.getenv("B2_ENDPOINT", "https://s3.us-east-005.backblazeb2.com")
B2_URL_CLOUD = os.getenv("B2_URL_CLOUD", "https://zipimgs.com/file/Lemiex-Fulfillment")
B2_FOLDER = "omniGen"  # Default folder for Pollinations

# Initialize B2 client
s3_client = None
if B2_ACCESS_KEY_ID and B2_SECRET_ACCESS_KEY:
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=B2_ENDPOINT,
            aws_access_key_id=B2_ACCESS_KEY_ID,
            aws_secret_access_key=B2_SECRET_ACCESS_KEY,
            config=Config(
                signature_version='s3v4', 
                connect_timeout=10, 
                read_timeout=30,
                retries={'max_attempts': 2}
            )
        )
        print(f"B2 client initialized: {B2_BUCKET}/{B2_FOLDER}")
    except Exception as e:
        print(f"Failed to initialize B2 client: {e}")
else:
    print("B2 credentials not configured - images won't be uploaded to cloud")

# Thread pool for B2 uploads
b2_executor = ThreadPoolExecutor(max_workers=5)

# Gallery caches (separate for each service)
cache_omnigen = {"data": [], "timestamp": 0}
cache_apiframe = {"data": [], "timestamp": 0}
cache_video = {"data": [], "timestamp": 0}
cache_kling = {"data": [], "timestamp": 0}  # Kling videos
GALLERY_CACHE_TTL = 300  # 5 minutes


def _sync_put_object(image_data: bytes, key: str, content_type: str = 'image/png') -> str:
    """Synchronous B2 put - runs in thread pool"""
    try:
        print(f"Start uploading {len(image_data)} bytes to {key}...")
        start_t = time.time()
        s3_client.put_object(
            Bucket=B2_BUCKET,
            Key=key,
            Body=image_data,
            ContentType=content_type
        )
        duration = time.time() - start_t
        base_url = os.getenv("B2_URL_CLOUD")
        b2_url = f"{base_url}/{key}"
        print(f"Uploaded to B2 in {duration:.2f}s: {b2_url}")
        return b2_url
    except Exception as e:
        print(f"B2 upload exception: {type(e).__name__}: {e}")
        return ""


async def upload_video_to_b2(video_data: bytes, folder: str = "video") -> str:
    """Upload video data directly to B2 (no re-download). Folder auto-created if not exists."""
    try:
        if not video_data:
            print("[B2 Video] No video data provided")
            return ""
        
        if not s3_client:
            print("[B2 Video] B2 not configured")
            return ""
        
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4"
        key = f"{folder}/{filename}"
        
        print(f"[B2 Video] Uploading {len(video_data):,} bytes to {key}...")
        
        loop = asyncio.get_event_loop()
        
        try:
            b2_url = await asyncio.wait_for(
                loop.run_in_executor(b2_executor, _sync_put_object, video_data, key, 'video/mp4'),
                timeout=180.0  # 3 minutes for large videos
            )
            return b2_url
        except asyncio.TimeoutError:
            print("[B2 Video] Upload timeout after 180s")
            return ""
        
    except Exception as e:
        print(f"[B2 Video] Upload failed: {e}")
        return ""


async def download_and_upload_video_to_b2(url: str, headers: dict = None) -> str:
    """Download video from URL and upload to B2 in 'video' folder"""
    try:
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.mp4"
        
        # Download video
        video_data = None
        async with httpx.AsyncClient(timeout=300.0) as client:
            for attempt in range(3):
                response = await client.get(url, headers=headers or {})
                
                if response.status_code == 429:
                    if attempt < 2:
                        wait_time = 5 * (attempt + 1)
                        print(f"Got 429, retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print("Max retries exceeded.")
                        return ""
                
                response.raise_for_status()
                video_data = response.content
                break
        
        if not video_data:
            return ""
        
        print(f"Downloaded video: {len(video_data)} bytes")
        
        if not s3_client:
            print("B2 not configured")
            return ""
        
        key = f"video/{filename}"
        loop = asyncio.get_event_loop()
        
        try:
            b2_url = await asyncio.wait_for(
                loop.run_in_executor(b2_executor, _sync_put_object, video_data, key, 'video/mp4'),
                timeout=120.0  # Longer timeout for video
            )
            return b2_url
        except asyncio.TimeoutError:
            print("B2 video upload timeout")
            return ""
        
    except Exception as e:
        print(f"Failed to download/upload video: {e}")
        return ""


async def download_and_upload_to_b2(url: str, subfolder: str = B2_FOLDER, headers: dict = None) -> str:
    """Download image from URL and upload directly to B2. Returns B2 public URL"""
    try:
        filename = f"{int(time.time())}_{uuid.uuid4().hex[:6]}.png"
        
        # Download to memory with retry logic for 429
        image_data = None
        async with httpx.AsyncClient(timeout=120.0) as client:
            for attempt in range(4):
                response = await client.get(url, headers=headers or {})
                
                if response.status_code == 429:
                    if attempt < 3:
                        wait_time = 3 * (attempt + 1)
                        print(f"Got 429 Too Many Requests, retrying in {wait_time}s... ({attempt+1}/3)")
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        print("Max retries exceeded for 429.")
                        return ""
                
                response.raise_for_status()
                image_data = response.content
                break
        
        if not image_data:
            return ""
        
        print(f"Downloaded image: {len(image_data)} bytes")
        
        # Upload to B2 in thread pool
        if not s3_client:
            print("B2 not configured")
            return ""
        
        key = f"{subfolder}/{filename}"
        loop = asyncio.get_event_loop()
        
        try:
            b2_url = await asyncio.wait_for(
                loop.run_in_executor(b2_executor, _sync_put_object, image_data, key),
                timeout=60.0
            )
            return b2_url
        except asyncio.TimeoutError:
            print("B2 upload timeout after 60s")
            return ""
        
    except Exception as e:
        print(f"Failed to download/upload: {e}")
        return ""


def _sync_list_b2_objects(prefix: str = "omniGen") -> list:
    """Synchronous B2 list with prefix"""
    try:
        print(f"--- [B2 List] Start listing for prefix: '{prefix}/' ---")
        response = s3_client.list_objects_v2(
            Bucket=B2_BUCKET,
            Prefix=f"{prefix}/",
            MaxKeys=500 
        )
        
        contents = response.get('Contents', [])
        print(f"--- [B2 List] Found {len(contents)} objects in '{prefix}' ---")
        
        base_url = os.getenv("B2_URL_CLOUD")
        files = []
        for obj in contents:
            key = obj['Key']
            filename = key.split('/')[-1]
            if filename.endswith('.png'):
                b2_url = f"{base_url}/{key}"
                files.append({
                    "url": b2_url,
                    "b2_url": b2_url,
                    "key": key,
                    "folder": prefix,
                    "time": obj['LastModified'].timestamp()
                })
        
        files.sort(key=lambda x: x["time"], reverse=True)
        return files
    except Exception as e:
        print(f"Failed to list B2 ({prefix}): {e}")
        return []


def _sync_list_b2_videos(prefix: str = "video") -> list:
    """List video files from B2"""
    try:
        print(f"--- [B2 Video List] Start listing for prefix: '{prefix}/' ---")
        response = s3_client.list_objects_v2(
            Bucket=B2_BUCKET,
            Prefix=f"{prefix}/",
            MaxKeys=200
        )
        
        contents = response.get('Contents', [])
        print(f"--- [B2 Video List] Found {len(contents)} videos in '{prefix}' ---")
        
        base_url = os.getenv("B2_URL_CLOUD")
        files = []
        for obj in contents:
            key = obj['Key']
            filename = key.split('/')[-1]
            if filename.endswith('.mp4'):
                b2_url = f"{base_url}/{key}"
                files.append({
                    "url": b2_url,
                    "b2_url": b2_url,
                    "key": key,
                    "folder": prefix,
                    "time": obj['LastModified'].timestamp(),
                    "type": "video"
                })
        
        files.sort(key=lambda x: x["time"], reverse=True)
        return files
    except Exception as e:
        print(f"Failed to list B2 videos ({prefix}): {e}")
        return []


async def refresh_cache(target: str = "omnigen") -> list:
    """Refresh specific gallery cache"""
    global cache_omnigen, cache_apiframe, cache_video, cache_kling
    loop = asyncio.get_running_loop()
    
    if target == "video":
        files = await loop.run_in_executor(b2_executor, _sync_list_b2_videos, "video")
        cache_video["data"] = files
        cache_video["timestamp"] = time.time()
        return cache_video["data"]
    
    if target == "kling":
        files = await loop.run_in_executor(b2_executor, _sync_list_b2_videos, "kling_video")
        cache_kling["data"] = files
        cache_kling["timestamp"] = time.time()
        return cache_kling["data"]
    
    prefix = "omniGen" if target == "omnigen" else "apiFrame"
    files = await loop.run_in_executor(b2_executor, _sync_list_b2_objects, prefix)
    
    if target == "omnigen":
        cache_omnigen["data"] = files
        cache_omnigen["timestamp"] = time.time()
        return cache_omnigen["data"]
    else:
        cache_apiframe["data"] = files
        cache_apiframe["timestamp"] = time.time()
        return cache_apiframe["data"]


