#!/bin/sh
# Run Stillpoint on the Lapis device screen.
# Needs the workspace numpy and the system Gio typelib on the path.
export PYTHONPATH="$LAPIS_WORKSPACE_NIX_PROFILE/lib/python3.13/site-packages:/usr/lib/python3/dist-packages:$PYTHONPATH"
exec python3 "$(dirname "$0")/weather.py" "$@"
