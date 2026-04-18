# 🤖 LocalScript

> Local AI coding agent — one container, zero cloud. Runs on GPU or CPU with a single command.

Built for **Hackathon 2026** · Python · Docker · Ollama · Qwen2.5-Coder · Make

---

## Getting started

> [!IMPORTANT]
> The first launch pulls Docker layers and downloads the model — this may take several minutes.

```bash
docker compose build allinone && make run
```

`make run` auto-detects GPU availability. If Docker can expose a GPU, it uses it — otherwise it falls back to CPU automatically.

---

## Modes

LocalScript runs in two modes:

- **REPL** — interactive session, ask questions one by one
- **One-shot** — single prompt, prints result and exits

### REPL

```bash
# CPU (any host with Docker)
docker compose run --rm allinone

# GPU (Linux · NVIDIA drivers · nvidia-container-toolkit)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm allinone
```

### One-shot

```bash
# CPU
docker compose run --rm allinone --pretty -p "Your prompt here"

# GPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml run --rm allinone --pretty -p "Your prompt here"
```

---

## Make shortcuts

The `make` targets probe for GPU support and choose the right Compose file automatically.

```bash
make run                          # REPL, auto GPU/CPU
make once PROMPT='Your prompt'    # one-shot, auto GPU/CPU
```

Explicit targets if you want to force one or the other:

```bash
make run-gpu
make run-cpu
make once-gpu PROMPT='...'
make once-cpu  PROMPT='...'
```

---

## Configuration

Pass environment variables to override defaults:

```bash
OLLAMA_MODEL=qwen2.5-coder:7b \
NUM_CTX=4096 \
NUM_PREDICT=256 \
docker compose run --rm allinone
```

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5-coder:7b` | Any Ollama-compatible model tag |
| `NUM_CTX` | `4096` | Context window size (tokens) |
| `NUM_PREDICT` | `256` | Max tokens to generate |
| `OLLAMA_VERBOSE_LOGS` | unset | Set to `1` to print Ollama logs to terminal |

---

## Notes

- Model weights are cached in Docker volume `ollama_data` — subsequent launches are fast.
- On low-memory hosts, use a smaller model: `OLLAMA_MODEL=qwen2.5-coder:3b`.
- Ollama logs are suppressed in REPL output by default; written to `/tmp/ollama.log` inside the container.
- GPU inference works on **Linux** with an NVIDIA GPU and `nvidia-container-toolkit`. **Docker Desktop on macOS** does not expose NVIDIA GPUs to Linux containers — use CPU mode there, or run on a Linux host.

---

## Requirements

- Docker 24+ with Docker Compose
- *(GPU mode)* Linux · NVIDIA drivers · [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- ~5 GB disk for model weights
- 8 GB RAM minimum (16 GB recommended for 7b model)

---

## License

MIT © Hackathon 2026
