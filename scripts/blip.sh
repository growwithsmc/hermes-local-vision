#!/usr/bin/env bash
# blip.sh - Blip: Local Vision for Hermes
# Usage: blip.sh [setup|start|stop|status|health|restart|show-key]
#
# Zero hardcoded values. EVERY configurable has an env var with a default.
# The defaults are sensible for a fresh install. Override any/all via env.
#
# For Hermes profiles that override $HOME, pass HOME=/real/home before blip.sh
# so paths resolve correctly. All paths below use BLIP_*DIR / BLIP_*FILE vars
# instead of bare $HOME references, so you can override them directly.
set -euo pipefail

# ── Config (ALL from env, ALL overridable) ────────────────────────────────
BLIP_PORT="${BLIP_PORT:-11788}"
BLIP_HOST="${BLIP_HOST:-127.0.0.1}"
BLIP_LLAMA_BIN="${BLIP_LLAMA_BIN:-}"
BLIP_SCRIPTS_DIR="${BLIP_SCRIPTS_DIR:-$HOME/.hermes/scripts}"
BLIP_MODEL_DIR="${BLIP_MODEL_DIR:-$HOME/.hermes/models/blip}"
BLIP_MODEL_FILE="${BLIP_MODEL_FILE:-SmolVLM-Instruct-Q4_K_M.gguf}"
BLIP_MMPROJ_FILE="${BLIP_MMPROJ_FILE:-mmproj-SmolVLM-Instruct-Q8_0.gguf}"
BLIP_PID_FILE="${BLIP_PID_FILE:-$BLIP_SCRIPTS_DIR/blip-server.pid}"
BLIP_AUTH_KEY_FILE="${BLIP_AUTH_KEY_FILE:-$BLIP_SCRIPTS_DIR/blip-auth.key}"
BLIP_ACTIVITY_FILE="${BLIP_ACTIVITY_FILE:-$BLIP_SCRIPTS_DIR/blip-last-used}"
BLIP_IDLE_TIMEOUT="${BLIP_IDLE_TIMEOUT:-90}"
BLIP_CONTEXT_SIZE="${BLIP_CONTEXT_SIZE:-8192}"
BLIP_IMAGE_MIN_TOKENS="${BLIP_IMAGE_MIN_TOKENS:--1}"
BLIP_IMAGE_MAX_TOKENS="${BLIP_IMAGE_MAX_TOKENS:--1}"
BLIP_NGL="${BLIP_NGL:--1}"
BLIP_THREADS="${BLIP_THREADS:--1}"
BLIP_FLASH_ATTN="${BLIP_FLASH_ATTN:-on}"
BLIP_MAX_IMAGES_PER_SEQUENCE="${BLIP_MAX_IMAGES_PER_SEQUENCE:-3}"
BLIP_TIMEOUT="${BLIP_TIMEOUT:-180}"
BLIP_MAX_IMAGE_SIZE="${BLIP_MAX_IMAGE_SIZE:-1048576}"
BLIP_MAX_IMAGE_DIMENSION="${BLIP_MAX_IMAGE_DIMENSION:-2048}"
BLIP_IMAGE_QUALITY="${BLIP_IMAGE_QUALITY:-85}"
BLIP_HERMES_CONFIG="${BLIP_HERMES_CONFIG:-$HOME/.hermes/config.yaml}"
BLIP_HF_REPO="${BLIP_HF_REPO:-ggml-org/SmolVLM-Instruct-GGUF}"

MODEL_PATH="$BLIP_MODEL_DIR/$BLIP_MODEL_FILE"
MMPROJ_PATH="$BLIP_MODEL_DIR/$BLIP_MMPROJ_FILE"
AUTH_PORT="$BLIP_PORT"
LLAMA_PORT=$((BLIP_PORT + 1))

# Auto-detect llama-server if not set
if [ -z "$BLIP_LLAMA_BIN" ]; then
    for c in "llama-server" "/usr/local/bin/llama-server" "/usr/bin/llama-server"; do
        command -v "$c" &>/dev/null && { BLIP_LLAMA_BIN="$c"; break; } || true
    done
fi

# ── Helpers ──────────────────────────────────────────────────────────────────
health_check() { curl -sf -H "Authorization: Bearer $(cat "$BLIP_AUTH_KEY_FILE" 2>/dev/null || true)" "http://$BLIP_HOST:$AUTH_PORT/v1/models" > /dev/null 2>&1; }
llama_health() { curl -sf "http://$BLIP_HOST:$LLAMA_PORT/v1/models" > /dev/null 2>&1; }
get_pid() { [ -f "$BLIP_PID_FILE" ] && cat "$BLIP_PID_FILE" || echo ""; }
is_running() { local p; p=$(get_pid); [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

port_free() {
    if command -v netstat &>/dev/null; then
        netstat -ano 2>/dev/null | grep -q ":$1 " && return 1 || return 0
    fi
    return 0
}

find_free_port() {
    local base=$1
    for attempt in $(seq 0 20); do
        local candidate=$((base + attempt))
        port_free "$candidate" && { echo "$candidate"; return 0; }
    done
    echo "$base"
    return 1
}

safe_path() {
    case "$1" in *[\;\&\|\$\`\(\)\{\}]*) return 1;; *) return 0;; esac
}

verify_files() {
    local m=0
    [ ! -f "$MODEL_PATH" ] && echo "ERROR: Model not found at $MODEL_PATH" && m=1
    [ ! -f "$MMPROJ_PATH" ] && echo "ERROR: MMProj not found at $MMPROJ_PATH" && m=1
    [ -z "$BLIP_LLAMA_BIN" ] || [ ! -f "$BLIP_LLAMA_BIN" ] && \
        echo "ERROR: llama-server not found. Set BLIP_LLAMA_BIN or build from https://github.com/ggml-org/llama.cpp" && m=1
    safe_path "$MODEL_PATH" || { echo "ERROR: Model path unsafe"; m=1; }
    safe_path "$MMPROJ_PATH" || { echo "ERROR: MMProj path unsafe"; m=1; }
    return $m
}

wait_for() {
    local name=$1 port=$2 pid=$3 timeout=${4:-30} auth_key=$5
    echo -n "  Waiting for $name..."
    for i in $(seq 1 "$timeout"); do
        if [ -n "$auth_key" ]; then
            curl -sf -H "Authorization: Bearer $auth_key" \r
                "http://$BLIP_HOST:$port/v1/models" > /dev/null 2>&1 && { echo " READY"; return 0; }
        else
            curl -sf "http://$BLIP_HOST:$port/v1/models" > /dev/null 2>&1 && { echo " READY"; return 0; }
        fi
        [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null && { echo " FAILED"; return 1; }
        echo -n "."; sleep 1
    done
    echo " TIMEOUT"; return 1
}

generate_key() {
    local file=$1
    mkdir -p "$(dirname "$file")"
    python3 -c "
import secrets, os
key = 'blip_' + secrets.token_hex(24)
with open('$file', 'w') as f: f.write(key + '\n')
os.chmod('$file', 0o600)
print(key)
" 2>/dev/null || { echo "ERROR: Failed to generate key" >&2; return 1; }
}

# ── Commands ─────────────────────────────────────────────────────────────────
cmd_setup() {
    echo "=== Blip Setup ==="
    echo ""

    if [ -z "$BLIP_LLAMA_BIN" ] || [ ! -f "$BLIP_LLAMA_BIN" ]; then
        echo "1/4. Building llama.cpp from source..."
        local tmp
        tmp=$(mktemp -d)
        git clone --depth 1 https://github.com/ggml-org/llama.cpp "$tmp/llama.cpp" 2>&1 | tail -1
        cd "$tmp/llama.cpp"
        cmake -B build -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -1
        cmake --build build --config Release -j --target llama-server 2>&1 | tail -1
        BLIP_LLAMA_BIN="$tmp/llama.cpp/build/bin/llama-server"
        [ ! -f "$BLIP_LLAMA_BIN" ] && BLIP_LLAMA_BIN="$tmp/llama.cpp/build/bin/Release/llama-server.exe"
        echo "  Built: $BLIP_LLAMA_BIN"
        echo "  Set BLIP_LLAMA_BIN=$BLIP_LLAMA_BIN in your env to reuse."
    else
        echo "1/4. llama-server: $BLIP_LLAMA_BIN"
    fi

    echo "2/4. Downloading model..."
    mkdir -p "$BLIP_MODEL_DIR"
    for f in "$BLIP_MODEL_FILE" "$BLIP_MMPROJ_FILE"; do
        [ -f "$BLIP_MODEL_DIR/$f" ] && echo "  Already cached: $f" && continue
        echo "  Downloading: $f..."
        curl -fSL "https://huggingface.co/$BLIP_HF_REPO/resolve/main/$f" \
            -o "$BLIP_MODEL_DIR/$f" || { echo "  FAILED"; exit 1; }
    done
    echo "  Done ($(du -sh "$BLIP_MODEL_DIR" | cut -f1))"

    echo "3/4. Generating auth key..."
    generate_key "$BLIP_AUTH_KEY_FILE"
    local key
    key=$(cat "$BLIP_AUTH_KEY_FILE")
    echo "  Key: ${key:0:16}..."

    echo "4/4. Starting server..."
    cmd_start || echo "  Server may already be running."

    echo ""
    echo "=== Blip Ready ==="
    echo "  Port:       $AUTH_PORT"
    echo "  API Key:    ${key:0:16}..."
    echo "  Config for Hermes:"
    echo "    auxiliary:"
    echo "      vision:"
    echo "        provider: openai"
    echo "        model: $BLIP_MODEL_FILE"
    echo "        base_url: \"http://$BLIP_HOST:$AUTH_PORT/v1\""
    echo "        api_key: \"$key\""
    echo ""
    echo "  To remember settings across sessions:"
    echo "    export BLIP_PORT=$AUTH_PORT"
    echo "    export BLIP_AUTH_KEY_FILE=\"$BLIP_AUTH_KEY_FILE\""
}

cmd_start() {
    is_running && echo "Already running on port $AUTH_PORT (PID $(get_pid))" && return 0

    local found_port
    found_port=$(find_free_port "$BLIP_PORT") || true
    AUTH_PORT="$found_port"
    LLAMA_PORT=$((AUTH_PORT + 1))

    verify_files || {
        echo "  Run 'blip.sh setup' first, or see https://github.com/growwithsmc/hermes-local-vision"
        return 1
    }

    mkdir -p "$BLIP_SCRIPTS_DIR"

    [ ! -f "$BLIP_AUTH_KEY_FILE" ] && generate_key "$BLIP_AUTH_KEY_FILE"
    local AUTH_KEY
    AUTH_KEY=$(cat "$BLIP_AUTH_KEY_FILE" 2>/dev/null || echo "")

    echo "Starting Blip on port $AUTH_PORT..."
    echo "  Model: $BLIP_MODEL_FILE"

    # Resolve Windows paths
    local LLAMA="$BLIP_LLAMA_BIN" MODEL="$MODEL_PATH" MMPROJ="$MMPROJ_PATH"
    if command -v cygpath &>/dev/null; then
        LLAMA=$(cygpath -w "$BLIP_LLAMA_BIN" 2>/dev/null || echo "$BLIP_LLAMA_BIN")
        MODEL=$(cygpath -w "$MODEL_PATH" 2>/dev/null || echo "$MODEL_PATH")
        MMPROJ=$(cygpath -w "$MMPROJ_PATH" 2>/dev/null || echo "$MMPROJ_PATH")
    fi

    # Start llama-server (internal port, localhost only)
    "$LLAMA" -m "$MODEL" --mmproj "$MMPROJ" \
        --port "$LLAMA_PORT" --host "$BLIP_HOST" \
        -c "$BLIP_CONTEXT_SIZE" --flash-attn "$BLIP_FLASH_ATTN" \
        -ngl "$BLIP_NGL" --threads "$BLIP_THREADS" --no-webui \
        --image-min-tokens "$BLIP_IMAGE_MIN_TOKENS" \
        --image-max-tokens "$BLIP_IMAGE_MAX_TOKENS" \
        > /dev/null 2>&1 &

    wait_for "llama-server" "$LLAMA_PORT" "$!" 30 || return 1

    # Start auth proxy (public port) - pass ALL BLIP_* vars that the proxy needs
    BLIP_PORT="$AUTH_PORT" \
    BLIP_HOST="$BLIP_HOST" \
    BLIP_MODEL_DIR="$BLIP_MODEL_DIR" \
    BLIP_MODEL_FILE="$BLIP_MODEL_FILE" \
    BLIP_MMPROJ_FILE="$BLIP_MMPROJ_FILE" \
    BLIP_LLAMA_BIN="$BLIP_LLAMA_BIN" \
    BLIP_CONTEXT_SIZE="$BLIP_CONTEXT_SIZE" \
    BLIP_IMAGE_MIN_TOKENS="$BLIP_IMAGE_MIN_TOKENS" \
    BLIP_IMAGE_MAX_TOKENS="$BLIP_IMAGE_MAX_TOKENS" \
    BLIP_NGL="$BLIP_NGL" \
    BLIP_THREADS="$BLIP_THREADS" \
    BLIP_FLASH_ATTN="$BLIP_FLASH_ATTN" \
    BLIP_AUTH_KEY_FILE="$BLIP_AUTH_KEY_FILE" \
    BLIP_PID_FILE="$BLIP_PID_FILE" \
    BLIP_SCRIPTS_DIR="$BLIP_SCRIPTS_DIR" \
    BLIP_MAX_IMAGES_PER_SEQUENCE="${BLIP_MAX_IMAGES_PER_SEQUENCE:-3}" \
    BLIP_TIMEOUT="${BLIP_TIMEOUT:-180}" \
    BLIP_MAX_IMAGE_SIZE="${BLIP_MAX_IMAGE_SIZE:-1048576}" \
    BLIP_MAX_IMAGE_DIMENSION="${BLIP_MAX_IMAGE_DIMENSION:-2048}" \
    BLIP_IMAGE_QUALITY="${BLIP_IMAGE_QUALITY:-85}" \
    python3 "$BLIP_SCRIPTS_DIR/blip-auth-proxy.py" --no-llama \
        > /dev/null 2>&1 &

    local AUTH_PID=$!
    echo "$AUTH_PID" > "$BLIP_PID_FILE"
    wait_for "auth proxy" "$AUTH_PORT" "$AUTH_PID" 15 "$AUTH_KEY"

    echo ""
    echo "  Ready on port $AUTH_PORT."
    echo "  API Key: ${AUTH_KEY:0:16}..."
    echo "  Config:"
    echo "    base_url: \"http://$BLIP_HOST:$AUTH_PORT/v1\""
    echo "    api_key: \"$AUTH_KEY\""
}

cmd_stop() {
    local pid; pid=$(get_pid)
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && { echo "Stopping auth proxy (PID $pid)..."; kill "$pid" 2>/dev/null || true; }

    local lp
    if command -v tasklist &>/dev/null; then
        lp=$(tasklist /fi "imagename eq llama-server.exe" /fo csv /nh 2>/dev/null | awk -F'"' '{print $2}' || true)
    else
        lp=$(ps -o pid= -C llama-server 2>/dev/null || true)
    fi
    [ -n "$lp" ] && { echo "Stopping llama-server (PID $lp)..."; kill "$lp" 2>/dev/null || true; }

    rm -f "$BLIP_PID_FILE"
    echo "Stopped"
}

cmd_status() {
    echo "Blip - Local Vision for Hermes"
    echo "  Port:       $AUTH_PORT"
    echo "  Host:       $BLIP_HOST"
    echo "  Auth proxy: $(is_running && echo "RUNNING (PID $(get_pid))" || echo STOPPED)"
    echo "  llama-srv:  $(llama_health && echo RUNNING || echo STOPPED)"
    echo "  Health:     $(health_check && echo "OK" || echo DOWN)"
    [ -f "$BLIP_AUTH_KEY_FILE" ] && echo "  API Key:    $(head -c 16 < "$BLIP_AUTH_KEY_FILE")..."
}

cmd_health() { health_check && echo "OK" && return 0; echo "DOWN"; return 1; }
cmd_show_key() { [ -f "$BLIP_AUTH_KEY_FILE" ] && cat "$BLIP_AUTH_KEY_FILE" || { echo "No key. Start server first." >&2; return 1; }; }
cmd_restart() { cmd_stop; sleep 1; cmd_start; }

# ── Dispatch ─────────────────────────────────────────────────────────────────
case "${1:-status}" in
    setup)     cmd_setup ;;
    start)     cmd_start ;;
    stop)      cmd_stop ;;
    status)    cmd_status ;;
    health)    cmd_health ;;
    restart)   cmd_restart ;;
    show-key)  cmd_show_key ;;
    *)
        echo "Blip - Local Vision for Hermes"
        echo ""
        echo "Usage: blip.sh <command>"
        echo ""
        echo "Commands:"
        echo "  setup      Full install: build llama.cpp, download model, start server"
        echo "  start      Start the vision server"
        echo "  stop       Stop the vision server"
        echo "  status     Show server status"
        echo "  health     Quick health check (OK/DOWN)"
        echo "  restart    Stop then start"
        echo "  show-key   Print the API key"
        echo ""
        echo "Env vars (all optional - nothing is hardcoded):"
        echo "  BLIP_PORT=11788            Base port (falls forward)"
        echo "  BLIP_HOST=127.0.0.1        Bind address"
        echo "  BLIP_MODEL_DIR             Model directory"
        echo "  BLIP_MODEL_FILE            Model GGUF filename"
        echo "  BLIP_MMPROJ_FILE           MMProj GGUF filename"
        echo "  BLIP_LLAMA_BIN             Path to llama-server binary"
        echo "  BLIP_SCRIPTS_DIR           Scripts directory"
        echo "  BLIP_PID_FILE              PID file path"
        echo "  BLIP_AUTH_KEY_FILE         Auth key file path"
        echo "  BLIP_ACTIVITY_FILE         Activity timestamp path"
        echo "  BLIP_IDLE_TIMEOUT=90       Minutes before idle shutdown (0 = indefinite)"
        echo "  BLIP_CONTEXT_SIZE=8192     Context window (tokens)"
        echo "  BLIP_IMAGE_MIN_TOKENS=-1   Min image tokens (-1 = auto)"
        echo "  BLIP_IMAGE_MAX_TOKENS=-1   Max image tokens (-1 = auto)"
        echo "  BLIP_NGL=-1                GPU layers (-1 = all)"
        echo "  BLIP_THREADS=-1            CPU threads (-1 = auto)"
        echo "  BLIP_FLASH_ATTN=on         Flash attention (on/off)"
        echo "  BLIP_MAX_IMAGES_PER_SEQUENCE=3  Max images before sequential split"
        echo "  BLIP_TIMEOUT=180           Upstream request timeout (seconds)"
        echo "  BLIP_MAX_IMAGE_SIZE=1048576  Max image bytes before compression"
        echo "  BLIP_MAX_IMAGE_DIMENSION=2048 Max image px (preserves aspect ratio)"
        echo "  BLIP_IMAGE_QUALITY=85      JPEG quality for compressed images"
        echo "  BLIP_HF_REPO               HF repo for model download"
        echo "  BLIP_HERMES_CONFIG         Hermes config to update"
        echo ""
        echo "Examples:"
        echo "  BLIP_PORT=11800 blip.sh start"
        echo "  blip.sh setup"
        exit 1
        ;;
esac
