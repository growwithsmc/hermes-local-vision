#!/usr/bin/env bash
# blip-watchdog.sh -- Cron watchdog for Blip vision server
# Designed for no_agent=true cron jobs. Silent unless events occur.
#
# Every path and config value comes from env vars (same vars as blip.sh).
# Nothing hardcoded. Works with any Hermes profile.

BLIP_PORT="${BLIP_PORT:-11788}"
BLIP_HOST="${BLIP_HOST:-127.0.0.1}"
BLIP_IDLE_TIMEOUT="${BLIP_IDLE_TIMEOUT:-90}"
BLIP_SCRIPTS_DIR="${BLIP_SCRIPTS_DIR:-$HOME/.hermes/scripts}"
BLIP_SCRIPT="${BLIP_SCRIPT:-$BLIP_SCRIPTS_DIR/blip.sh}"
BLIP_AUTH_KEY_FILE="${BLIP_AUTH_KEY_FILE:-$BLIP_SCRIPTS_DIR/blip-auth.key}"
BLIP_ACTIVITY_FILE="${BLIP_ACTIVITY_FILE:-$BLIP_SCRIPTS_DIR/blip-last-used}"

# Read auth key
BLIP_AUTH_KEY="${BLIP_AUTH_KEY:-$(cat "$BLIP_AUTH_KEY_FILE" 2>/dev/null)}"

# Health check -- updates activity timestamp on success
IS_UP=false
curl -sf -H "Authorization: Bearer $BLIP_AUTH_KEY" \
  "http://$BLIP_HOST:$BLIP_PORT/v1/health" > /dev/null 2>&1 && { IS_UP=true; touch "$BLIP_ACTIVITY_FILE"; }

# Idle shutdown
if $IS_UP && [ -f "$BLIP_ACTIVITY_FILE" ]; then
    NOW=$(date +%s)
    LAST=$(date -r "$BLIP_ACTIVITY_FILE" +%s 2>/dev/null || echo "$NOW")
    [ $((NOW - LAST)) -gt $((BLIP_IDLE_TIMEOUT * 60)) ] && {
        echo "Blip idle >${BLIP_IDLE_TIMEOUT}m. Shutting down."
        BLIP_PORT="$BLIP_PORT" BLIP_HOST="$BLIP_HOST" \
        BLIP_SCRIPTS_DIR="$BLIP_SCRIPTS_DIR" \
        bash "$BLIP_SCRIPT" stop
        rm -f "$BLIP_ACTIVITY_FILE"; exit 0
    }
fi

# Restart if down
$IS_UP && exit 0
echo "Blip down. Starting..."
BLIP_PORT="$BLIP_PORT" BLIP_HOST="$BLIP_HOST" \
BLIP_SCRIPTS_DIR="$BLIP_SCRIPTS_DIR" \
bash "$BLIP_SCRIPT" start && touch "$BLIP_ACTIVITY_FILE"
