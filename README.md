# Cinematic AI Video Platform

AI-powered video generation from text prompts and images. Create short cinematic videos with smooth animations, contextual sound effects, and professional transitions.

## Features

- 🎬 **Text-to-Video**: Generate complete video scenes from text descriptions
- 🖼️ **Image Animation**: Convert static images into smooth animated clips
- 🔊 **Smart Audio**: Automatic SFX insertion and background music handling
- 🎥 **Scene Composition**: Stitch multiple scenes with professional transitions
- 🛡️ **Content Safety**: Multi-layer moderation for family-friendly output
- 📱 **Social Ready**: Export MP4 optimized for social platforms

## Quick Start

### Prerequisites

- Docker & Docker Compose
- NVIDIA GPU with CUDA 12.1+ (for local model inference)
- 16GB+ VRAM recommended for SDXL + SVD

### Development Setup

```bash
# Clone the repository
git clone https://github.com/your-org/cinematic-ai-video.git
cd cinematic-ai-video

# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Check logs
docker-compose logs -f api

# API available at http://localhost:8000
# API docs at http://localhost:8000/api/docs
```

### API Usage

```python
import requests

# Create a video project
response = requests.post(
    "http://localhost:8000/api/v1/projects",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "title": "My First Video",
        "scenes": [
            {
                "scene_number": 1,
                "prompt": "A serene forest at sunrise, golden light filtering through trees",
                "duration": 5.0,
                "style_preset": "cinematic",
                "sound_effects": [
                    {"type": "nature", "description": "birds chirping"}
                ]
            },
            {
                "scene_number": 2,
                "prompt": "A deer walking through the forest, peaceful morning",
                "duration": 5.0,
                "transition": "crossfade"
            }
        ]
    }
)

project = response.json()
print(f"Project ID: {project['id']}")

# Check progress
status = requests.get(
    f"http://localhost:8000/api/v1/projects/{project['id']}/progress",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)
print(f"Progress: {status.json()['progress_percent']}%")
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Client    │────▶│  FastAPI    │────▶│   Redis     │
│   (Web/API) │     │   Server    │     │   Queue     │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌──────────────────────────┴──────┐
                    ▼                                 ▼
             ┌─────────────┐                  ┌─────────────┐
             │ GPU Worker  │                  │ CPU Worker  │
             │ (SDXL, SVD) │                  │ (Audio,FFmpeg)│
             └─────────────┘                  └─────────────┘
                    │                                 │
                    └──────────────┬──────────────────┘
                                   ▼
                            ┌─────────────┐
                            │   S3/MinIO  │
                            │   Storage   │
                            └─────────────┘
```

## Project Structure

```
cinematic-ai-video/
├── src/
│   ├── api/                 # FastAPI application
│   │   ├── main.py          # App entry point
│   │   ├── routes/          # API endpoints
│   │   ├── deps.py          # Dependencies
│   │   └── middleware.py    # Middleware
│   ├── config/
│   │   └── settings.py      # Configuration
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   ├── services/            # Business logic
│   │   ├── safety.py        # Content moderation
│   │   ├── image_generator.py
│   │   ├── video_animator.py
│   │   ├── audio_processor.py
│   │   ├── video_composer.py
│   │   └── storage.py
│   ├── workers/             # Celery tasks
│   │   ├── celery_app.py
│   │   └── tasks.py
│   └── database/
│       └── connection.py
├── docs/
│   └── ARCHITECTURE.md
├── config/
├── tests/
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Configuration

Key environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `DB_POSTGRES_URL` | PostgreSQL connection | `postgresql+asyncpg://...` |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `AI_DEVICE` | PyTorch device | `cuda` |
| `AI_IMAGE_MODEL` | SDXL model ID | `stabilityai/stable-diffusion-xl-base-1.0` |
| `SAFETY_ENABLED` | Enable content moderation | `true` |
| `VIDEO_DEFAULT_FPS` | Output frame rate | `24` |

## API Reference

### Projects

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/projects` | Create new video project |
| GET | `/api/v1/projects` | List user's projects |
| GET | `/api/v1/projects/{id}` | Get project details |
| GET | `/api/v1/projects/{id}/progress` | Get generation progress |
| POST | `/api/v1/projects/{id}/cancel` | Cancel generation |
| POST | `/api/v1/projects/{id}/retry` | Retry failed project |
| DELETE | `/api/v1/projects/{id}` | Delete project |

### Assets

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/assets/upload` | Upload image/audio |
| GET | `/api/v1/assets/{id}` | Get asset download URL |

## Development

```bash
# Install dependencies locally
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API server
uvicorn src.api.main:app --reload

# Run Celery worker
celery -A src.workers.celery_app:celery_app worker --loglevel=info

# Run tests
pytest tests/ -v --cov=src
```

## License

MIT License - see LICENSE file for details.
