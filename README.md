# Marker serve

A FastAPI server for document processing using the marker-pdf library for production environments, inspired by the [official Marker server script](https://github.com/VikParuchuri/marker/blob/b8a4c5d8769ed40f83d0ac9b452e85532ac7cd47/marker/scripts/server.py).

## Requirements

- Docker with NVIDIA GPU support
- NVIDIA drivers and CUDA installed on the host
- Docker Compose v2+

## Running the Container

The project includes a `compose.yml` file for easy deployment:

```bash
docker compose up --detach
```

## Environment Variables

The following environment variables can be configured:

- `HOST`: Host address to bind to (default: 0.0.0.0)
- `PORT`: Port to listen on (default: 8000)
- `WORKERS`: Number of worker processes (default: 1)
- `WORKER_CONNECTIONS`: Maximum number of connections for each worker (default: 1000)
- `TIMEOUT`: Worker timeout in seconds (default: 30)
- `KEEP_ALIVE`: Keep-alive timeout in seconds (default: 75)
- `GRACEFUL_TIMEOUT`: Graceful timeout in seconds (default: 75)
- `LOG_LEVEL`: Logging level (default: info)
- `TORCH_DEVICE`: Device to run marker on (default: cuda)
- `GUNICORN_CMD_ARGS`: Additional gunicorn command arguments (default: "")
- `API_KEY`: Secret key for API authentication. If not set, authentication is disabled.

Example in compose.yml:

```yaml
services:
  marker:
    # other configuration...
    environment:
      - TORCH_DEVICE=cuda
      - WORKERS=2
      - LOG_LEVEL=debug
      - TIMEOUT=60
      - API_KEY=your_secret_api_key
```

## Project Structure

```
.
├── server/            # Main FastAPI application code
│   ├── main.py        # Application entry point
│   ├── schemas.py     # Pydantic data models
│   └── ...
├── scripts/           # Utility scripts
│   └── startup.sh     # Container startup script
└── ...
```

## API Endpoints

### GET /health

Health check endpoint.

### POST /marker/upload

Process a document.

**Parameters:**
- `file`: The PDF file to process (required)
- `page_range`: Page range to convert (e.g. "0,5-10,20")
- `force_ocr`: Force OCR on all pages (default: false)
- `paginate_output`: Whether to paginate the output (default: false)
- `output_format`: Output format - "markdown", "json", or "html" (default: "markdown")
- `use_llm`: Use LLM to improve conversion accuracy (default: false)

**Example:**

```bash
curl -X POST "http://localhost:8000/parse" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "output_format=markdown" \
  -F "force_ocr=false"
```

**Authentication:**

If API_KEY is set, all requests must include the Bearer token:

```bash
curl -X POST "http://localhost:8000/parse" \
  -H "accept: application/json" \
  -H "Authorization: Bearer your_secret_api_key" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```