# Known Issues

## 1. `<global-img>` Token Missing in SmolVLM GGUF

**Status:** Fixed via patch  
**Files affected:** `patches/llama-cpp-mtmd-idefics3-token-fix.patch`

The official SmolVLM GGUF from `ggml-org/SmolVLM-Instruct-GGUF` lacks the
`<global-img>` special token in its vocabulary. llama.cpp's Idefics3 projector
code (in `tools/mtmd/mtmd.cpp`) expects this token and returns `LLAMA_TOKEN_NULL`
when it's missing, causing "Invalid token" errors on all image requests.

## 2. CUDA-Only Build Requires Dynamic Loading

**Status:** Behavior, not bug  
**Workaround:** Use `BUILD_SHARED_LIBS=ON` (default with Ninja)

When llama.cpp is built with only CUDA support (no CPU backend), models that
try to split tensors between CPU and GPU will fail with:
`make_cpu_buft_list: no CPU backend found`

The `BUILD_SHARED_LIBS=ON` build includes CPU backend variants as separate
DLLs that are loaded dynamically when needed.

## 3. `file://` URLs Require `--media-path`

**Status:** By design in llama.cpp  
**Workaround:** Not needed - Hermes sends images as base64 data URLs

llama.cpp blocks `file://` URLs unless `--media-path` is specified. Base64
data URLs (which Hermes uses natively) work without this flag.

## 4. Config Changes Require Session Restart

**Status:** Behavior, not bug  
**Workaround:** `/reset` in Hermes or start a new session

The Hermes CLI config bridge reads `config.yaml` at module import time. Changes
made while a session is running aren't picked up until the next session start.

## 5. Env Vars Stale After Config Change

**Status:** Behavior, not bug  
**Workaround:** Session restart

Same root cause as #4 - the env var bridge and config reader both snapshot at
session start. Changing `auxiliary.vision.*` in `config.yaml` requires a
`/reset` to take effect.

## 6. Large Image Processing Time

**Status:** Expected  
**Mitigation:** Use `--image-min-tokens 64 --image-max-tokens 4096`

SmolVLM at Q4_K_M processes images in ~100-600ms depending on GPU.
Larger images take longer as they're split into more 384px tiles.

## 7. VRAM Usage Reporting

**Status:** Cosmetic  
**Explanation:** SmolVLM Q4_K_M uses ~1.1 GB VRAM. `nvidia-smi` may report
slightly more due to CUDA context overhead.

## 8. Windows Path Resolution in Git-Bash

**Status:** Environment-specific  
**Workaround:** The script handles `cygpath` conversion automatically

When running the server script in git-bash (MSYS2), paths are converted from
MSYS-style (`/c/Users/...`) to Windows-native (`C:\Users\...`) for the
`llama-server.exe` binary.
