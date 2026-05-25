# Patches for llama.cpp

These patches fix compatibility issues between llama.cpp's multimodal pipeline
and specific GGUF model files.

## Why Patches Are Needed

llama.cpp's multimodal support (`libmtmd`) uses specific special tokens to
mark image boundaries in the token stream. These tokens are looked up in the
model's vocabulary at initialization time. If a token doesn't exist in the
vocabulary, `lookup_token()` returns `LLAMA_TOKEN_NULL`, which causes
"Invalid token" 400 errors when processing images.

## Patch List

### `llama-cpp-mtmd-idefics3-token-fix.patch`

**Affects:** `tools/mtmd/mtmd.cpp`  
**Models:** SmolVLM-Instruct (all sizes, all quants) from
`ggml-org/SmolVLM-Instruct-GGUF`

**Root cause:** The Idefics3 projector code tries to look up `<global-img>`
as a fallback token. The SmolVLM GGUF doesn't include this token in its
vocabulary, causing the entire image processing chain to produce null tokens.

**Fix:** Remove `<global-img>` from the fallback chain. The remaining token
`<fake_token_around_image>` (token ID 49152) exists in the vocabulary and is
sufficient for correct image boundary marking.

**To apply:**
```bash
cd /path/to/llama.cpp
git apply /path/to/hermes-smolvlm-addon/patches/llama-cpp-mtmd-idefics3-token-fix.patch

# Rebuild:
cmake --build build --config Release -j --target llama-server
```

**Upstream status:** Not yet submitted.

## How to Create a Patch

```bash
cd ~/llama.cpp
# Make your edits, then:
git diff tools/mtmd/mtmd.cpp > patches/my-fix.patch
```
