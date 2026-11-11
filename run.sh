#!/usr/bin/env sh
# Serve the site locally and open it in the default browser (macOS / Linux).
cd "$(dirname "$0")" || exit 1
PORT="${PORT:-8000}"

if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else
  echo "Python 3 was not found. Open index.html directly, or install Python 3 and run this again."
  exit 1
fi

URL="http://localhost:$PORT/"
echo "Starting How AI Learns to See on $URL  (Ctrl+C to stop)"
( sleep 1; if command -v open >/dev/null 2>&1; then open "$URL"; elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"; fi ) >/dev/null 2>&1 &
exec "$PY" -m http.server "$PORT" --bind 127.0.0.1
