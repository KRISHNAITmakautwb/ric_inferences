# RIC Inference Pipeline
## Router → Inference → Cache
**Mac Mini M3 (Gemma 3 12B) + OmniRoute Cloud**

---

## Project Structure

```
ric_inference/
├── config.yaml          # ← Edit this first! (IPs, model names)
├── requirements.txt
├── server.py            # FastAPI REST server
├── benchmark.py         # Benchmark local vs cloud vs RIC
├── client_example.py    # Quick test example
├── ric/
│   ├── pipeline.py      # Main orchestrator (Router→Inference→Cache)
│   ├── router.py        # Query routing logic
│   ├── inference.py     # Ollama (local) + OmniRoute (cloud) adapters
│   ├── cache.py         # SQLite async cache with TTL + LRU
│   └── utils.py         # Config loader, logger, helpers
├── slides/
│   └── ric_presentation.tex   # LaTeX Beamer slides for PI
├── logs/                # Runtime logs + benchmark output
└── cache_db/            # SQLite cache database
```

---

## Quick Start

### 1. On Mac Mini — Install Ollama + Gemma 3

```bash
# Install Ollama (on the Mac Mini)
brew install ollama

# Pull Gemma 3 12B (needs ~8GB RAM, perfect for M3)
ollama pull gemma3:12b

# Start server
ollama serve
```

### 2. Edit `config.yaml`

```yaml
local:
  host: "192.168.1.XXX"    # ← Mac Mini's IP address
  model: "gemma3:12b"

cloud:
  base_url: "http://YOUR_OMNI_ROUTE_IP/v1"
  api_key:  ""          # keep empty, see below
```

Put the API key in an environment variable, never in `config.yaml`:

```bash
export OMNIROUTE_API_KEY="your-key"
# optional: export OMNIROUTE_BASE_URL="http://YOUR_OMNI_ROUTE_IP/v1"
```

### 3. Install Python Dependencies

```bash
cd /home/crypto_nn/ric_inference
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run

```bash
# Start the RIC API server
python server.py
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)

# Quick test
python client_example.py

# Full benchmark
python benchmark.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/infer` | Run query through RIC pipeline |
| `GET`  | `/health` | Check local + cloud connectivity |
| `GET`  | `/stats` | Cache hit rate + routing stats |
| `DELETE` | `/cache/clear` | Clear all cached responses |

**Example:**
```bash
curl -X POST http://localhost:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?"}'
```

**Response:**
```json
{
  "destination": "local",
  "latency_ms": 840.2,
  "total_tokens": 187,
  "from_cache": false,
  "routing_reason": "short query (4 words ≤ 60 threshold)"
}
```

---

## How Routing Works

```
Query received
    │
    ├─ In cache?          → Return instantly (0 tokens, ~2ms)
    │
    ├─ Cloud keyword?     → Cloud (OmniRoute)
    │   (analyze, debug, write code, research...)
    │
    ├─ Local keyword?     → Local (Mac Mini Gemma)
    │   (translate, define, summarize briefly...)
    │
    └─ Word count ≤ 60?   → Local
       Word count > 60?   → Cloud
```

---

## Expected Token Savings

| Query Type | Without RIC | With RIC | Saving |
|------------|------------|----------|--------|
| Simple queries | Cloud tokens | **Local (free)** | **100%** |
| Repeated queries | Cloud tokens | **Cache (free)** | **100%** |
| Complex queries | Cloud tokens | Cloud tokens | 0% |
| **Overall** | 100% | **~20-40%** | **60-80%** |

---

## Build Slides PDF

```bash
cd slides/
pdflatex ric_presentation.tex
pdflatex ric_presentation.tex   # Run twice for TOC
```

---

## Configuration Reference

See `config.yaml` for all options including:
- `router.mode`: `heuristic` / `always_local` / `always_cloud`
- `router.complexity_threshold`: word count cutoff (default: 60)
- `cache.ttl_hours`: how long to keep cached responses
- `local.model`: change Gemma variant (`gemma3:4b`, `gemma3:12b`, `gemma2:9b`)
