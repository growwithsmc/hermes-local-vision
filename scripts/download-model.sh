#!/usr/bin/env bash
# download-model.sh - Download SmolVLM GGUF + mmproj with SHA256 verification
# Usage: download-model.sh [target-dir]

set -euo pipefail

REPO="${BLIP_HF_REPO:-ggml-org/SmolVLM-Instruct-GGUF}"
MODEL_FILE="${BLIP_HF_MODEL:-SmolVLM-Instruct-Q4_K_M.gguf}"
MMPROJ_FILE="${BLIP_HF_MMPROJ:-mmproj-SmolVLM-Instruct-Q8_0.gguf}"
TARGET="${1:-${BLIP_MODEL_DIR:-$HOME/.hermes/models/blip}}"
mkdir -p "$TARGET"

# SHA256 checksums - verify integrity before loading
# SmolVLM-Instruct Q4_K_M:   dc80966bd84789de64115f07888939c03abb1714d431c477dfb405517a554af5
# mmproj SmolVLM-Instruct:    (not computed - skipped due to placeholder)
# Qwen2.5-VL-7B Q4_K_M:      9258bf05b12686d097ff3b6b18d968ab393649780aa2b3cd67fec43d50554392
# mmproj Qwen2.5-VL-7B:      2ddb555391bae966e412deab9e07b58afa18bcc06930ba0f1c78a3695ab9e506
declare -A CHECKSUMS
CHECKSUMS["SmolVLM-Instruct-Q4_K_M.gguf"]="dc80966bd84789de64115f07888939c03abb1714d431c477dfb405517a554af5"
CHECKSUMS["mmproj-SmolVLM-Instruct-Q8_0.gguf"]="0000000000000000000000000000000000000000000000000000000000000000"
CHECKSUMS["Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"]="9258bf05b12686d097ff3b6b18d968ab393649780aa2b3cd67fec43d50554392"
CHECKSUMS["mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"]="2ddb555391bae966e412deab9e07b58afa18bcc06930ba0f1c78a3695ab9e506"

echo "Downloading SmolVLM-Instruct to: $TARGET"
echo "  Main model:  ~1.04 GB (Q4_K_M)"
echo "  Projector:   ~0.55 GB (Q8_0)"
echo ""

verify_sha256() {
    local file=$1
    local expected=$2
    if [ "$expected" = "0000000000000000000000000000000000000000000000000000000000000000" ]; then
        echo "  ⚠ SHA256 check skipped (placeholder hash)"
        return 0
    fi
    if command -v sha256sum &>/dev/null; then
        local actual
        actual=$(sha256sum "$file" | cut -d' ' -f1)
        if [ "$actual" != "$expected" ]; then
            echo "  ✗ SHA256 MISMATCH for $(basename "$file")"
            echo "    Expected: $expected"
            echo "    Actual:   $actual"
            echo "    File may be corrupted or tampered with. Aborting."
            rm -f "$file"
            return 1
        fi
        echo "  ✓ SHA256 verified"
    elif command -v shasum &>/dev/null; then
        local actual
        actual=$(shasum -a 256 "$file" | cut -d' ' -f1)
        if [ "$actual" != "$expected" ]; then
            echo "  ✗ SHA256 MISMATCH for $(basename "$file")"
            rm -f "$file"
            return 1
        fi
        echo "  ✓ SHA256 verified"
    else
        echo "  ⚠ No sha256sum available, skipping verification"
    fi
    return 0
}

download_file() {
    local repo=$1
    local file=$2
    local dest="$TARGET/$file"

    if [ -f "$dest" ]; then
        echo "  Already cached: $file"
        verify_sha256 "$dest" "${CHECKSUMS[$file]:-}" || return 1
        return 0
    fi

    echo "  Downloading: $file..."
    local url="https://huggingface.co/$repo/resolve/main/$file"

    if command -v curl &>/dev/null; then
        curl -fSL "$url" -o "$dest" || return 1
    elif command -v wget &>/dev/null; then
        wget -O "$dest" "$url" || return 1
    else
        echo "    ERROR: Neither curl nor wget found"
        return 1
    fi

    verify_sha256 "$dest" "${CHECKSUMS[$file]:-}" || return 1
    return 0
}

download_file "$REPO" "$MODEL_FILE" || exit 1
download_file "$REPO" "$MMPROJ_FILE" || exit 1

echo ""
echo "Download complete:"
ls -lh "$TARGET"
