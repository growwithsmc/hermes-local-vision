"""
Blip Vision Plugin for Hermes Agent
=====================================
Local vision server for Hermes. Auto-detects GPU, downloads binaries + model,
configures Hermes, and starts the server. All from within a conversation.

Usage from chat:
  "Set up local vision for me" → Hermes calls blip_setup()
  "/blip status" → Check if running
  "/blip stop" → Stop the server

Architecture:
  register(ctx) - called by plugin loader. Registers /blip command.
  blip_setup()  - full auto-setup. Callable by Hermes agent or user.
  _auto_install() - the actual work. Idempotent, safe to re-run.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("hermes.plugins.blip_vision")

# ── Paths (all from BLIP_* env vars - single source of truth) ─────────────
_PLUGIN_DIR = Path(__file__).parent.resolve()
_USER_HOME = Path.home()
_HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(_USER_HOME / ".hermes")))
_SCRIPTS_DIR = Path(os.environ.get("BLIP_SCRIPTS_DIR", str(_USER_HOME / ".hermes" / "scripts")))
_MODELS_DIR = Path(os.environ.get("BLIP_MODEL_DIR", str(_USER_HOME / ".hermes" / "models" / "blip")))
_BIN_DIR = Path(os.environ.get("BLIP_BIN_DIR", str(_USER_HOME / ".hermes" / "bin")))
_HF_REPO = os.environ.get("BLIP_HF_REPO", "ggml-org/SmolVLM-Instruct-GGUF")
_HF_MODEL = os.environ.get("BLIP_HF_MODEL", "SmolVLM-Instruct-Q4_K_M.gguf")
_HF_MMPROJ = os.environ.get("BLIP_HF_MMPROJ", "mmproj-SmolVLM-Instruct-Q8_0.gguf")
_INSTALLED_MARKER = _SCRIPTS_DIR / ".blip-installed"

# ── Model picker ──────────────────────────────────────────────────────────
_MODELS = [
    # (min_vram_mb, name, hf_repo, model_file, mmproj_file)
    (0,     "SmolVLM 2B",     "ggml-org/SmolVLM-Instruct-GGUF",
     "SmolVLM-Instruct-Q4_K_M.gguf", "mmproj-SmolVLM-Instruct-Q8_0.gguf"),
    (4096,  "Gemma 3 4B",     "ggml-org/gemma-3-4b-it-GGUF",
     "gemma-3-4b-it-Q4_K_M.gguf", "mmproj-gemma-3-4b-it-Q8_0.gguf"),
    (8192,  "Qwen2.5-VL-7B",  "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF",
     "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"),
]

_LLAMA_RELEASES = "https://github.com/ggml-org/llama.cpp/releases"


# ═══════════════════════════════════════════════════════════════════════════
# Core setup logic - callable by Hermes agent or /blip setup
# ═══════════════════════════════════════════════════════════════════════════

def _detect_gpu() -> dict:
    """Detect GPU and return info dict. Returns {vram_mb: int, name: str}."""
    info = {"vram_mb": 0, "name": "unknown", "cuda": False}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(", ")
            info["name"] = parts[0]
            info["vram_mb"] = int(parts[1].split()[0])
            info["cuda"] = True
    except Exception:
        pass
    return info


def pick_model(vram_mb: int) -> dict:
    """Pick the best model for available VRAM. Returns model config dict.
    
    If BLIP_HF_REPO is set (user explicitly chose a model), use that.
    Otherwise auto-pick based on VRAM from the built-in model list.
    """
    # User explicitly chose via env vars
    repo = os.environ.get("BLIP_HF_REPO", "")
    model_file = os.environ.get("BLIP_HF_MODEL", "")
    mmproj_file = os.environ.get("BLIP_HF_MMPROJ", "")
    if repo and model_file and mmproj_file:
        name = repo.split("/")[-1].replace("-GGUF", "").replace("-Instruct", "")
        return {"name": name, "hf_repo": repo,
                "model_file": model_file, "mmproj_file": mmproj_file}

    # Auto-pick based on VRAM
    chosen = _MODELS[0]
    for min_vram, name, repo, mf, mpf in _MODELS:
        if vram_mb >= min_vram:
            chosen = (min_vram, name, repo, mf, mpf)
    return {
        "name": chosen[1], "hf_repo": chosen[2],
        "model_file": chosen[3], "mmproj_file": chosen[4],
    }


def _find_prebuilt_llama() -> Path | None:
    """Search for an existing llama-server binary.
    
    Checks BLIP_LLAMA_BIN env var first, then common locations.
    """
    # 1. Env var override
    env_bin = os.environ.get("BLIP_LLAMA_BIN", "")
    if env_bin:
        p = Path(env_bin)
        if p.exists():
            return p
    # 2. PATH
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    # 3. LM Studio backends
    for p in [
        _USER_HOME / ".cache" / "lm-studio" / "extensions" / "backends",
        _USER_HOME / ".cache" / "lm-studio" / "backends",
    ]:
        if p.exists():
            for b in sorted(p.iterdir(), reverse=True):
                exe = b / "llama-server.exe"
                if exe.exists():
                    return exe
    # Unsloth build
    for p in [
        _USER_HOME / ".unsloth" / "llama.cpp" / "build" / "bin",
        _USER_HOME / "llama.cpp" / "build" / "bin",
    ]:
        exe = p / "llama-server.exe"
        if exe.exists():
            return exe
    return None


def _download_llama_server(target_dir: Path) -> dict:
    """Download pre-built llama-server. Returns {path, status, message}."""
    # Check for existing first
    existing = _find_prebuilt_llama()
    if existing:
        return {"path": str(existing), "status": "found", "message": f"Using existing at {existing}"}

    import platform
    system = platform.system().lower()
    machine = platform.machine().lower()

    try:
        gpu = _detect_gpu()
        cuda_tag = "cu12"
    except Exception:
        cuda_tag = "cu12"

    urls = {
        "windows": f"{_LLAMA_RELEASES}/download/latest/llama-bins-win-{cuda_tag}-amd64.zip",
        "linux": f"{_LLAMA_RELEASES}/download/latest/llama-bins-ubuntu-x64.zip",
    }

    url = urls.get(system)
    if not url:
        return {"path": "", "status": "error", "message": f"Unsupported OS: {system}"}

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "llama.zip"

    try:
        logger.info("blip: downloading llama-server (%s)...", url.split("/")[-1])
        urllib.request.urlretrieve(url, zip_path)
        import zipfile
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(target_dir)
        zip_path.unlink()

        exe = next(target_dir.rglob("llama-server*"), None)
        if exe:
            exe.chmod(0o755)
            return {"path": str(exe), "status": "downloaded",
                    "message": f"Downloaded to {exe}"}
        return {"path": "", "status": "error", "message": "Binary not found in archive"}
    except Exception as e:
        return {"path": "", "status": "error", "message": str(e)}


def _download_model(model: dict, dest: Path) -> dict:
    """Download model GGUF files. Returns {status, message, paths}."""
    files = [model["model_file"], model["mmproj_file"]]
    results = {}
    base_url = f"https://huggingface.co/{model['hf_repo']}/resolve/main"

    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        path = dest / f
        if path.exists():
            results[f] = {"path": str(path), "status": "cached"}
            continue
        url = f"{base_url}/{f}"
        logger.info("blip: downloading %s (~%d MB)...", f,
                    5000 if "7B" in f else 1000)
        try:
            urllib.request.urlretrieve(url, path)
            results[f] = {"path": str(path), "status": "downloaded"}
        except Exception as e:
            results[f] = {"path": "", "status": "error", "message": str(e)}

    errors = [r.get("message", "") for r in results.values()
              if r.get("status") == "error"]
    if errors:
        return {"status": "error", "message": "; ".join(errors), "files": results}
    return {"status": "ok", "message": "All model files ready", "files": results}


def _configure_hermes(port: int, api_key: str, model_file: str) -> dict:
    """Add auxiliary.vision section to Hermes config.yaml. Uses YAML parsing
    so model:/base_url:/api_key: replacements are scoped to auxiliary.vision
    only - never touch other config sections."""
    import yaml

    config_path = _HERMES_HOME / "config.yaml"
    if not config_path.exists():
        config_path = _USER_HOME / ".hermes" / "config.yaml"
    if not config_path.exists():
        return {"status": "error", "message": "No config.yaml found"}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Ensure structure exists
        if "auxiliary" not in cfg:
            cfg["auxiliary"] = {}
        if "vision" not in cfg["auxiliary"]:
            cfg["auxiliary"]["vision"] = {}

        # Set vision config - only touches auxiliary.vision.*
        cfg["auxiliary"]["vision"]["provider"] = "openai"
        cfg["auxiliary"]["vision"]["model"] = model_file
        cfg["auxiliary"]["vision"]["base_url"] = f"http://127.0.0.1:{port}/v1"
        cfg["auxiliary"]["vision"]["api_key"] = api_key
        cfg["auxiliary"]["vision"]["timeout"] = 120

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

        return {"status": "ok", "message": f"Config updated at {config_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def _start_server(port: int, llama_bin: str, model_dir: str,
                  model_file: str, mmproj_file: str, api_key: str) -> dict:
    """Start the BLIP vision server. Returns {status, message}."""
    script = _SCRIPTS_DIR / "blip.sh"

    # Copy scripts if needed
    src = _PLUGIN_DIR / "scripts"
    if src.exists() and not script.exists():
        _SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.sh"):
            shutil.copy2(f, _SCRIPTS_DIR / f.name)
            (_SCRIPTS_DIR / f.name).chmod(0o755)
        for f in src.glob("*.py"):
            shutil.copy2(f, _SCRIPTS_DIR / f.name)

    if not script.exists():
        return {"status": "error", "message": "blip.sh not found"}

    env = os.environ.copy()
    env["HOME"] = str(_USER_HOME)
    env["BLIP_PORT"] = str(port)
    env["BLIP_LLAMA_BIN"] = llama_bin
    env["BLIP_MODEL_DIR"] = model_dir
    env["BLIP_MODEL_FILE"] = model_file
    env["BLIP_MMPROJ_FILE"] = mmproj_file
    env["BLIP_IDLE_TIMEOUT"] = os.environ.get("BLIP_IDLE_TIMEOUT", "90")
    env["BLIP_SCRIPTS_DIR"] = str(_SCRIPTS_DIR)
    env["BLIP_AUTH_KEY_FILE"] = str(_SCRIPTS_DIR / "blip-auth.key")
    env["BLIP_MAX_IMAGE_SIZE"] = "1048576"
    env["BLIP_MAX_IMAGE_DIMENSION"] = "2048"

    try:
        subprocess.Popen(
            ["bash", str(script), "start"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(4)

        # Verify it started
        import urllib.request as req
        try:
            r = req.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            return {"status": "ok", "message": f"Server running on port {port}"}
        except Exception:
            return {"status": "warning",
                    "message": "Server may still be starting. Run /blip status to check."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# Public API - callable by Hermes agent
# ═══════════════════════════════════════════════════════════════════════════

def blip_setup(port: int = None) -> str:
    """Full auto-setup. Callable by Hermes agent or /blip setup.

    Port defaults to BLIP_PORT env var (or 11788 if unset).
    All paths read from BLIP_* env vars, falling back to defaults.

    Returns a human-readable summary of what happened.
    Can be called from within a conversation - Hermes reads the result.
    Safe to re-run (skips already-downloaded files).
    """
    if port is None:
        port = int(os.environ.get("BLIP_PORT", "11788"))
    steps = []

    # 1. Detect GPU
    gpu = _detect_gpu()
    if gpu["vram_mb"] > 0:
        steps.append(f"✅ Detected {gpu['name']} ({gpu['vram_mb']} MB VRAM)")
    else:
        steps.append("⚠️  No NVIDIA GPU detected - will use CPU (slow)")

    # 2. Pick model
    model = pick_model(gpu["vram_mb"])
    steps.append(f"📦 Selected model: {model['name']}")

    # 3. Get/Download llama-server
    llama = _download_llama_server(_BIN_DIR)
    if llama["status"] == "found":
        steps.append(f"✅ Found llama-server: {llama['path']}")
    elif llama["status"] == "downloaded":
        steps.append(f"✅ Downloaded llama-server")
    else:
        steps.append(f"❌ {llama['message']}")

    # 4. Download model
    model_result = _download_model(model, _MODELS_DIR)
    if model_result["status"] == "ok":
        steps.append("✅ Model files ready")
    else:
        steps.append(f"❌ Model download: {model_result['message']}")

    # 5. Generate auth key
    key_file = _SCRIPTS_DIR / "blip-auth.key"
    key_file.parent.mkdir(parents=True, exist_ok=True)
    if not key_file.exists():
        key = "blip_" + secrets.token_hex(24)
        key_file.write_text(key + "\n")
        key_file.chmod(0o600)
    else:
        key = key_file.read_text().strip()
    steps.append(f"🔑 Auth key ready")

    # 6. Update Hermes config
    if llama["path"] or model_result["status"] == "ok":
        cfg_result = _configure_hermes(port, key, model["model_file"])
        steps.append(f"📝 {cfg_result['message']}")

    # 7. Start server
    if llama["path"] and model_result["status"] == "ok":
        start_result = _start_server(
            port, llama["path"], str(_MODELS_DIR),
            model["model_file"], model["mmproj_file"], key,
        )
        steps.append(start_result["message"])

        # Install vision-context sub-plugin
        sub_plugin = _PLUGIN_DIR / "plugins" / "vision-context"
        if sub_plugin.exists():
            target = _USER_HOME / ".hermes" / "plugins" / "vision-context"
            target.mkdir(parents=True, exist_ok=True)
            for f in sub_plugin.glob("*"):
                if f.is_file():
                    shutil.copy2(f, target / f.name)
            steps.append("✅ Vision context plugin installed")

    # Mark as installed
    _INSTALLED_MARKER.write_text(
        f"installed:{time.time()}\nmodel:{model['name']}\nport:{port}\n"
    )

    summary = "\n".join(steps)
    logger.info("blip-setup complete:\n%s", summary)
    return f"**BLIP Vision Setup Complete**\n\n{summary}\n\nUse /blip status to check or just send me an image to test!"


# ═══════════════════════════════════════════════════════════════════════════
# Plugin lifecycle
# ═══════════════════════════════════════════════════════════════════════════

def register(ctx) -> None:
    """Called by Hermes plugin loader. Registers commands + optional auto-setup."""

    # Copy scripts if needed
    src = _PLUGIN_DIR / "scripts"
    if src.exists():
        _SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*.sh"):
            shutil.copy2(f, _SCRIPTS_DIR / f.name)
            (_SCRIPTS_DIR / f.name).chmod(0o755)
        for f in src.glob("*.py"):
            shutil.copy2(f, _SCRIPTS_DIR / f.name)

    # Register /blip command
    @ctx.command("blip")
    def cmd_blip(args: str) -> str:
        """Manage BLIP local vision server.

        Usage:
          /blip setup       - auto-download + configure + start (full setup)
          /blip start       - start the server
          /blip stop        - stop the server
          /blip status      - check if running
          /blip restart     - restart
          /blip show-key    - show API key

        From chat: just ask "set up local vision" and Hermes handles it.
        """
        script = _SCRIPTS_DIR / "blip.sh"
        parts = args.strip().split()
        cmd = parts[0] if parts else "status"

        if cmd == "setup":
            return blip_setup()

        if cmd in ("start", "stop", "status", "restart", "show-key", "key"):
            if not script.exists():
                return "BLIP scripts not found. Try: /blip setup"
            try:
                sub_cmd = "show-key" if cmd == "key" else cmd
                result = subprocess.run(
                    ["bash", str(script), sub_cmd],
                    capture_output=True, text=True, timeout=60,
                )
                output = result.stdout or result.stderr or ""
                return f"BLIP {cmd}:\n{output.strip()}"
            except subprocess.TimeoutExpired:
                return "Command timed out. The model may still be loading."
            except Exception as e:
                return f"Error: {e}"

        return (
            "Usage: /blip [setup|start|stop|status|restart|show-key|key]\n"
            "Or just tell me: 'set up local vision' and I'll run /blip setup for you."
        )

    logger.info("blip-vision: registered (/blip setup|start|stop|status|restart|key)")


def uninstall(ctx) -> None:
    """Clean up on plugin removal."""
    script = _SCRIPTS_DIR / "blip.sh"
    if script.exists():
        try:
            subprocess.run(["bash", str(script), "stop"], timeout=10)
        except Exception:
            pass
    _INSTALLED_MARKER.unlink(missing_ok=True)
    logger.info("blip-vision: uninstalled")
