# Architecture

## Overview

```
┌─────────────────────────────────────────────────────┐
│                  Hermes Agent                        │
│                                                      │
│  ┌──────────────┐    ┌──────────────┐                │
│  │ vision_      │    │ browser_     │                │
│  │ analyze      │    │ vision       │                │
│  └──────┬───────┘    └──────┬───────┘                │
│         │                  │                         │
│         └────────┬─────────┘                         │
│                  ▼                                   │
│         call_llm(task="vision")                      │
│                  │                                   │
│                  ▼                                   │
│         resolve_vision_provider_client()              │
│                  │                                   │
│                  ▼                                   │
│         _resolve_task_provider_model("vision")        │
│                  │                                   │
│         ┌────────┴────────┐                          │
│         │  config.yaml    │                          │
│         │  auxiliary.     │                          │
│         │  vision.*       │                          │
│         └────────┬────────┘                          │
│                  │                                   │
│                  ▼                                   │
│         OpenAI-compatible HTTP client                 │
│         base_url="http://127.0.0.1:11788/v1"          │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│           llama-server (standalone process)          │
│                                                      │
│  Port:       11788                                   │
│  Binary:     llama-server.exe                        │
│  Backend:    CUDA 12/13 (or Metal, or CPU)           │
│  Flash attn: yes                                     │
│  GPU layers: all (-ngl -1)                           │
│  Context:    4096                                    │
│                                                      │
│  ┌──────────────────┐  ┌────────────────────────┐   │
│  │ SmolVLM-Instruct │  │ mmproj (SigLip ViT)    │   │
│  │ Q4_K_M (1.04 GB) │  │ Q8_0 (0.55 GB)         │   │
│  └──────────────────┘  └────────────────────────┘   │
└────────────────────┬──────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│              NVIDIA GPU (RTX 40 series)              │
│                                                      │
│  ~1.1 GB VRAM used                                   │
│  ~15 GB VRAM remaining for main model                 │
│  GPU acceleration via --flash-attn          │
└─────────────────────────────────────────────────────┘
```

## Process Lifecycle

```
┌──────────────┐
│  Hermes cron │  runs every 10 minutes
│  watchdog    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│ blip-watchdog.sh                                   │
│                                                       │
│  ┌─ Health check: curl 127.0.0.1:11788/v1/models     │
│  │                                                    │
│  ├── Healthy → touch activity timestamp → silent exit │
│  │                                                    │
│  ├── Down → blip.sh start → exit with log  │
│  │                                                    │
│  ├── Unresponsive → kill + restart → exit with log   │
│  │                                                    │
│  └── Idle >30 min → blip.sh stop → exit    │
│         (frees GPU VRAM for main model)               │
└──────────────────────────────────────────────────────┘
```

## Request Flow (detailed)

1. User uploads image or takes screenshot
2. Hermes agent calls `vision_analyze` or `browser_vision`
3. Both tools eventually call `call_llm(task="vision", ...)` from `agent/auxiliary_client.py`
4. `call_llm` checks `if task == "vision":` → calls `resolve_vision_provider_client()`
5. That calls `_resolve_task_provider_model("vision")` which reads `auxiliary.vision` from `config.yaml`
6. Config returns `provider=openai, model=SmolVLM-Instruct, base_url=http://127.0.0.1:11788/v1`
7. An OpenAI-compatible HTTP client is created pointing at the local server
8. Image is base64-encoded and sent as `data:image/png;base64,...` via the chat completions API
9. llama-server processes the image through the mmproj vision encoder
10. SmolVLM generates a text response describing the image

## Why This Design

- **Standalone process** - doesn't compete with your main model for GPU scheduling
- **No LM Studio dependency** - can run even if LM Studio is serving a different model
- **30-min idle kill** - frees ~1.1 GB VRAM when you're not using vision
- **Cron watchdog** - no need to remember to start the server
- **One port** - 11788 is unlikely to conflict with anything
- **OpenAI-compatible API** - Hermes sends standard chat completions requests
