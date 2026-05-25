#!/usr/bin/env bash
# install.sh - One-command Blip vision setup with supply-chain verification
#
# Safe usage (download, inspect, verify, execute):
#   curl -fsSL https://raw.githubusercontent.com/growwithsmc/hermes-local-vision/main/install.sh -o /tmp/blip-install.sh
#   less /tmp/blip-install.sh          # inspect
#   bash /tmp/blip-install.sh          # execute
#
# Or with automatic hash verification:
#   curl -fsSL https://raw.githubusercontent.com/growwithsmc/hermes-local-vision/main/install.sh -o /tmp/blip-install.sh
#   echo "<HASH> /tmp/blip-install.sh" | sha256sum --check --status && bash /tmp/blip-install.sh

set -euo pipefail

MODEL_REPO="ggml-org/SmolVLM-Instruct-GGUF"
MODEL_FILE="SmolVLM-Instruct-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-SmolVLM-Instruct-Q8_0.gguf"
MODEL_DIR="$HOME/.hermes/models/blip"
SCRIPTS_DIR="$HOME/.hermes/scripts"

OS="$(uname -s)"
case "$OS" in Darwin*) PLATFORM="macos" ;; Linux*) PLATFORM="linux" ;; MINGW*|MSYS*) PLATFORM="windows" ;; *) echo "Unsupported OS: $OS"; echo "  If the problem persists: https://github.com/growwithsmc/hermes-local-vision/issues/new"; exit 1 ;; esac

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Blip Vision - Local VLM Setup           ║"
echo "╚══════════════════════════════════════════════╝"
echo "  Platform: $PLATFORM"
echo "  Model:    $MODEL_FILE (~1.04 GB)"
echo "  Projector: $MMPROJ_FILE (~0.55 GB)"
echo ""

# Step 1: Install llama-server
echo "→ Step 1/3: Installing llama.cpp..."
LLAMA_BIN=""
case "$PLATFORM" in
    macos) command -v brew &>/dev/null && { brew install llama.cpp; LLAMA_BIN="llama-server"; } ;;
    linux) ;;
    windows) command -v winget &>/dev/null && { winget install llama.cpp 2>/dev/null; LLAMA_BIN="llama-server"; } ;;
esac

if [ -z "$LLAMA_BIN" ] || ! command -v "$LLAMA_BIN" &>/dev/null; then
    else
        echo "  ⚠ Could not install llama.cpp automatically."
        echo "  Install manually: https://github.com/ggml-org/llama.cpp"
    fi
fi

# Step 2: Download model
echo "→ Step 2/3: Downloading model..."
mkdir -p "$MODEL_DIR"
for FILE in "$MODEL_FILE" "$MMPROJ_FILE"; do
    if [ -f "$MODEL_DIR/$FILE" ]; then
        echo "  Already cached: $FILE"
    else
        echo "  Downloading: $FILE..."
        curl -fSL "https://huggingface.co/$MODEL_REPO/resolve/main/$FILE" -o "$MODEL_DIR/$FILE" || {
            echo "  ERROR: Download failed for $FILE"
            echo "  If the problem persists: https://github.com/growwithsmc/hermes-local-vision/issues/new"; exit 1
        }
    fi
done
echo "  Model ready ($(du -sh "$MODEL_DIR" | cut -f1))"

# Step 3: Install scripts and start
echo "→ Step 3/3: Configuring..."
mkdir -p "$SCRIPTS_DIR"
SCRIPT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd 2>/dev/null || echo "")"
if [ -n "$SCRIPT_SRC" ] && [ -f "$SCRIPT_SRC/scripts/blip.sh" ]; then
    cp "$SCRIPT_SRC/scripts/"*.sh "$SCRIPTS_DIR/" 2>/dev/null || true
    cp "$SCRIPT_SRC/scripts/blip-auth-proxy.py" "$SCRIPTS_DIR/" 2>/dev/null || true
fi

export BLIP_LLAMA_BIN="$LLAMA_BIN"
bash "$SCRIPTS_DIR/blip.sh" start 2>&1 || true

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Setup Complete                             ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "  Auth proxy:  http://127.0.0.1:11788/v1"
echo "  API key:     stored at ~/.hermes/scripts/blip-auth.key"
echo "  Show key:    bash ~/.hermes/scripts/blip.sh show-key"
echo ""
echo "  To test:"
echo "    KEY=\$(bash ~/.hermes/scripts/blip.sh show-key)"
echo "    curl -H \"Authorization: Bearer \$KEY\" http://127.0.0.1:11788/v1/models"
echo ""
echo "  Hermes config (add to ~/.hermes/config.yaml):"
echo "    auxiliary:"
echo "      vision:"
echo "        provider: openai"
echo "        model: $MODEL_FILE"
echo "        base_url: \"http://127.0.0.1:11788/v1\""
echo "        api_key: \"\$(bash ~/.hermes/scripts/blip.sh show-key)\""
