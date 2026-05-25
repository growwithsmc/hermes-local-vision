# What Works / What Doesn't

## ✅ Working

| Scenario | Status | Notes |
|----------|--------|-------|
| `vision_analyze` with local image file | ✅ | Base64-encoded data URL via OpenAI-compatible API |
| `vision_analyze` with remote URL | ✅ | llama.cpp downloads and processes the image |
| `browser_vision` screenshot analysis | ✅ | Routes through `call_llm(task="vision")` → aux pipeline |
| Multi-image input | ✅ | Multiple images in a single turn |
| Text-only prompts alongside images | ✅ | Mixed content arrays |
| Chat template rendering | ✅ | SmolVLM's built-in chat template (im_start/im_end format) |
| Server idle shutdown (30 min) | ✅ | Watchdog kills process, frees GPU VRAM |
| Auto-restart on crash | ✅ | Watchdog detects down server, restarts within 10s |
| Standalone process | ✅ | Port 11788, independent of LM Studio or other servers |
| CUDA GPU offload | ✅ | `-ngl -1` offloads all layers to GPU |
| Flash attention | ✅ | `--flash-attn on` (GPU acceleration) |

## ⚠️ Partial / Untested

| Scenario | Status | Notes |
|----------|--------|-------|
| Linux / macOS | ⚠️ Untested | Scripts are bash, should work with `llama-server` binary |
| CPU-only inference | ⚠️ Untested | Remove `-ngl -1`, expect ~5-10s per image |
| AMD GPUs (ROCm) | ⚠️ Untested | Needs custom llama.cpp build with ROCm |
| Apple Silicon (Metal) | ⚠️ Untested | Needs `-ngl -1` and Metal-enabled llama.cpp build |
| SmolVLM-256M / 500M | ⚠️ Untested | Smaller models should work, may need different `--image-min-tokens` |
| SmolVLM2 models | ⚠️ Untested | Different GGUF repo, may need separate mmproj |

## ❌ Known Issues

| Issue | Root Cause | Workaround |
|-------|-----------|------------|
| "Invalid token" error with SmolVLM GGUF | `<global-img>` token missing in vocabulary | Apply the included patch to llama.cpp's `tools/mtmd/mtmd.cpp` |
| `file:///` URLs not working | `--media-path` not set on server | Use base64 data URLs instead (Hermes does this automatically) |
| Server fails with "no CPU backend" | CUDA-only build without CPU fallback | Use the dynamic-load build from `build/bin/llama-server.exe` (11K stub + DLLs) |
| Env vars stale after config change | Config bridge reads at session start | Session restart (`/reset`) picks up changes |
| VRAM not freed on Hermes exit | Server is a separate process | Watchdog idle-kills after 30 min, or manual `blip.sh stop` |

## Model Compatibility

The patch applies specifically to `ggml-org/SmolVLM-Instruct-GGUF` (the 2.2B
SmolVLM model). Other Idefics3-based models may also be affected if they
lack the `<global-img>` token in their vocabulary.

Tested models:
- ✅ `ggml-org/SmolVLM-Instruct-GGUF` (Q4_K_M, Q8_0)
- ✅ Built from local conversion of `HuggingFaceTB/SmolVLM-Instruct`
- ❓ `ggml-org/SmolVLM-256M-Instruct-GGUF` (likely needs same patch)
- ❓ `ggml-org/SmolVLM2-2.2B-Instruct-GGUF` (likely needs same patch)
