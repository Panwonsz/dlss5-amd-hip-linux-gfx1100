#!/bin/bash
# Start the DLSS-NR model server on the host, then launch the game.
#
# Use it as a Steam launch option:
#
#   /path/to/dlss5-nr-run.sh %command%
#
# This replaces the whole preload arrangement. The model no longer runs inside the game, so there is
# no LD_PRELOAD, no ROCm bundled into the Steam container, no libamd_comgr copy beside the game and no
# 601 MB of weights in the game folder. The daemon is an ordinary host process using /opt/rocm
# directly; the game reaches it on loopback, which crosses into the container because pressure-vessel
# shares the network namespace.
#
# The daemon outlives the game on purpose. Loading the weights costs real seconds, and a game that
# crashes or gets restarted should not pay them again -- the next launch reconnects to a daemon that
# is already warm.
#
# Environment it honours:
#   DLSS5_WEIGHTS     the weights folder (the one with manifest.json); found automatically if unset
#   DLSS5_NR_PORT     loopback port, default 47820
#   DLSS5_NR_STEPS    how much of the exchange the game records, 0..3, default 3
#   DLSS5_NR_DAEMON   the daemon binary, if it is not beside this script
set -uo pipefail

PORT="${DLSS5_NR_PORT:-47820}"
STATE="${XDG_CACHE_HOME:-$HOME/.cache}/dlss5-hip"
LOG="$STATE/daemon.log"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$STATE"

say() { echo "dlss5-nr: $*" >&2; }

listening() { (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; }

# --- the daemon binary -------------------------------------------------------------------------
DAEMON="${DLSS5_NR_DAEMON:-}"

if [[ -z "$DAEMON" ]]; then
    for candidate in "$HERE/dlss5-nr-daemon" "$HERE/../linux/dlss5-nr-daemon" "$STATE/dlss5-nr-daemon"; do
        [[ -x "$candidate" ]] && { DAEMON="$candidate"; break; }
    done
fi

# --- the model library -------------------------------------------------------------------------
LIBRARY="${DLSS5_NR_LIBRARY:-}"

if [[ -z "$LIBRARY" ]]; then
    for candidate in "$HERE/../hip/libdlss5_hip.so" "$HERE/libdlss5_hip.so" "$STATE/libdlss5_hip.so"; do
        [[ -f "$candidate" ]] && { LIBRARY="$(cd "$(dirname "$candidate")" && pwd)/$(basename "$candidate")"; break; }
    done
fi

# --- the weights -------------------------------------------------------------------------------
# The folder holding manifest.json, not the DLL it came out of. Searching the cache is worth it
# because the path has a schema and a hash in it that nobody remembers.
WEIGHTS="${DLSS5_WEIGHTS:-}"

if [[ -z "$WEIGHTS" && -d "$STATE/weights" ]]; then
    found=$(find "$STATE/weights" -maxdepth 3 -name manifest.json -print -quit 2>/dev/null)
    [[ -n "$found" ]] && WEIGHTS=$(dirname "$found")
fi

# --- start it if it is not already up ------------------------------------------------------------
if listening; then
    say "daemon already running on 127.0.0.1:$PORT"
else
    if [[ -z "$DAEMON" || ! -x "$DAEMON" ]]; then
        say "no daemon binary found -- build it with 'make -C hip daemon' (looked beside $HERE)"
    elif [[ -z "$WEIGHTS" || ! -f "$WEIGHTS/manifest.json" ]]; then
        say "no weights found -- set DLSS5_WEIGHTS to the folder containing manifest.json"
        WEIGHTS=""
    else
        say "starting the daemon: $DAEMON --weights $WEIGHTS"
        : > "$LOG"
        setsid "$DAEMON" --weights "$WEIGHTS" ${LIBRARY:+--library "$LIBRARY"} --port "$PORT" \
            >> "$LOG" 2>&1 < /dev/null &
        disown 2>/dev/null

        # Loading 148 MB of weights and bringing up ROCm takes a while, and the game must not start
        # before the daemon can answer -- Ensure() only tries to connect once.
        for _ in $(seq 1 120); do
            listening && break
            sleep 1
        done

        if listening; then
            say "daemon ready (log: $LOG)"
        else
            say "the daemon did not come up in two minutes -- see $LOG"
            tail -n 5 "$LOG" >&2 2>/dev/null
        fi
    fi
fi

# --- hand over to the game -----------------------------------------------------------------------
# Only two variables cross into the container now, and neither of them is a path.
export DLSS5_NR_BACKEND=hip
export DLSS5_NR_PORT="$PORT"
[[ -n "${DLSS5_NR_STEPS:-}" ]] && export DLSS5_NR_STEPS

exec "$@"
