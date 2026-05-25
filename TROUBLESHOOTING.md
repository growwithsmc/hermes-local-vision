# Troubleshooting

## "Invalid token" 400 error on vision requests

**Cause:** The SmolVLM GGUF doesn't contain the `<global-img>` special token
that llama.cpp's Idefics3 multimodal projector expects. `lookup_token()` returns
`LLAMA_TOKEN_NULL`, which poisons the entire image processing chain.

**Fix:** Apply the patch in `patches/llama-cpp-mtmd-idefics3-token-fix.patch`
to `llama.cpp/tools/mtmd/mtmd.cpp`, then rebuild.

```bash
cd ~/llama.cpp
git apply /path/to/hermes-smolvlm-addon/patches/llama-cpp-mtmd-idefics3-token-fix.patch
cmake --build build --config Release -j --target llama-server
```

**Verify:** The patched line should read:
```cpp
tok_ov_img_start = {lookup_token("<fake_token_around_image>")};
```

## Server starts but vision requests time out

1. Check if the server is actually running: `bash scripts/blip.sh status`
2. Check that the model loaded correctly: `curl http://127.0.0.1:11788/v1/models`
3. Ensure CUDA is available: `nvidia-smi` - the server should show GPU memory usage
4. Check if the mmproj file matches the model (wrong mmproj = silent failure)

## "Failed to download image" error

The server can't reach the image URL. This is common on Windows with certain
network configurations. Fix: use base64-encoded image data URLs instead
(the `vision_analyze` tool does this automatically for local files).

## Server dies immediately on startup

Run the server in the foreground to see the error:

```bash
/path/to/llama-server.exe \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --port 11788 -c 4096 --flash-attn on -ngl -1
```

Common errors:
- `no CPU backend found` - CUDA-only build, use the dynamic-load build
- `failed to load model` - corrupted GGUF or wrong mmproj
- `CUDA error` - driver version mismatch or insufficient VRAM

## vision_analyze works but browser_vision doesn't

If `vision_analyze` succeeds but `browser_vision` fails:

1. Check the main model supports vision (DeepSeek does not)
2. Verify `AUXILIARY_VISION_MODEL` env var is set: `echo $AUXILIARY_VISION_MODEL`
3. Check `config.yaml` has the correct `auxiliary.vision` section
4. After changing config, do `/reset` in Hermes

Both tools route through `call_llm(task="vision")` from `agent/auxiliary_client.py`,
which reads the config at call time. A stale session is the most common cause.

## Env vars show old values after config change

The Hermes CLI config bridge runs at module import time. Changes to
`config.yaml` mid-session aren't picked up. Either:
- Start a new Hermes session: `/reset`
- Or restart the terminal: exit and re-run `hermes`

## GPU VRAM not freed after stopping

The llama-server keeps VRAM allocated until the process exits. Stop it
explicitly: `bash scripts/blip.sh stop`. The watchdog will
auto-shut it after 30 min of inactivity.

## Cannot start on port 11788 (already in use)

```bash
# Find what's on the port
netstat -ano | grep 11788
# Kill it
taskkill /F /PID <PID>
# Or use a different port by editing the script
```

## Getting Help

If you encounter an issue not covered here:

1. Check open/closed issues: https://github.com/growwithsmc/hermes-local-vision/issues
2. Open a new issue with:
   - The exact command you ran
   - The full error output
   - Your OS and GPU model
   - What you expected to happen
3. Include the output of: `bash ~/.hermes/scripts/blip.sh status`
