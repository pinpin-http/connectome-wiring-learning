#!/usr/bin/env bash
# Vast.ai container services (supervisor; no systemd, no Docker). Idempotent; root; never touches the driver.
#   deployment/vast_services.sh install|status|remove    (remove leaves runs and data untouched)
# smb-monitor    deployment/monitor.py --loop 60: artifacts/monitor/{state.json,resources.jsonl,events.jsonl,dashboard.html}
# smb-dashboard  static server for artifacts/monitor on 127.0.0.1:17100, published by the Vast Caddy edge on
#                port 10100 with the instance's token auth (/etc/portal.yaml); tokenless only via an SSH tunnel
# smb-backup     deployment/backup_remote.sh --loop 1800, once the rclone remote "smbbackup" is configured
#                (deployment/README.md, "Backups"); until then it records "not configured"
# Services survive SSH disconnection and instance stop/start, not recycle or destroy. They start no runs.
set -euo pipefail
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPTS=/opt/supervisor-scripts
CONF=/etc/supervisor/conf.d
PORT_EXT="${SMB_DASHBOARD_PORT:-10100}"      # one of the instance's free "normal" open ports
PORT_INT=17100
LABEL="SMB dashboard"
[ -d "$SCRIPTS/utils" ] && command -v supervisorctl >/dev/null || { echo "not a Vast base image (no supervisor utils)" >&2; exit 1; }
yamlpy() { for p in /venv/main/bin/python python3; do "$p" -c 'import yaml' 2>/dev/null && { echo "$p"; return; }; done; echo "no python with yaml" >&2; exit 1; }

write_conf() {   # name
  cat > "$CONF/$1.conf" <<EOF
[program:$1]
environment=PROC_NAME="%(program_name)s"
command=$SCRIPTS/$1.sh
autostart=true
autorestart=true
startsecs=5
stopasgroup=true
killasgroup=true
stopsignal=TERM
stopwaitsecs=20
stdout_logfile=/dev/stdout
redirect_stderr=true
stdout_events_enabled=true
stdout_logfile_maxbytes=0
stdout_logfile_backups=0
EOF
}

install() {
  cat > "$SCRIPTS/smb-monitor.sh" <<EOF
#!/bin/bash
utils=$SCRIPTS/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
cd "$PROJECT"
exec python3 deployment/monitor.py --loop 60 2>&1
EOF
  cat > "$SCRIPTS/smb-dashboard.sh" <<EOF
#!/bin/bash
utils=$SCRIPTS/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
. "\${utils}/exit_portal.sh" "$LABEL"
mkdir -p "$PROJECT/artifacts/monitor"
cd "$PROJECT/artifacts/monitor"
exec python3 -m http.server $PORT_INT --bind 127.0.0.1 2>&1
EOF
  cat > "$SCRIPTS/smb-backup.sh" <<EOF
#!/bin/bash
utils=$SCRIPTS/utils
. "\${utils}/logging.sh"
. "\${utils}/environment.sh"
cd "$PROJECT"
exec bash deployment/backup_remote.sh --loop 1800 2>&1
EOF
  chmod 755 "$SCRIPTS"/smb-{monitor,dashboard,backup}.sh
  for s in smb-monitor smb-dashboard smb-backup; do write_conf "$s"; done
  cp -n /etc/portal.yaml /etc/portal.yaml.before-smb 2>/dev/null || true
  "$(yamlpy)" - "$LABEL" "$PORT_EXT" "$PORT_INT" <<'PY'
import os, sys, tempfile, yaml
label, ext, internal = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
path = "/etc/portal.yaml"
d = yaml.safe_load(open(path)) or {}
apps = d.setdefault("applications", {})
used = {a.get("external_port") for k, a in apps.items() if k != label}
if ext in used:
    sys.exit(f"external port {ext} already used in {path}")
if not os.environ.get(f"VAST_TCP_PORT_{ext}"):
    sys.exit(f"VAST_TCP_PORT_{ext} is not set: {ext} is not an open port of this instance")
apps[label] = {"hostname": "localhost", "external_port": ext, "internal_port": internal, "open_path": "/dashboard.html",
               "name": label}
fd, tmp = tempfile.mkstemp(dir="/etc", prefix=".portal.")
with os.fdopen(fd, "w") as f:
    yaml.safe_dump(d, f, sort_keys=False)
os.chmod(tmp, 0o644)
os.replace(tmp, path)
print(f"{path}: '{label}' -> external {ext} (public {os.environ[f'VAST_TCP_PORT_{ext}']}) -> 127.0.0.1:{internal}")
PY
  rm -f "/tmp/supervisor-skip/smb-dashboard"
  supervisorctl reread
  supervisorctl update
  supervisorctl restart smb-monitor smb-dashboard smb-backup >/dev/null || true
  supervisorctl restart caddy
  sleep 3
  status
}

status() {
  supervisorctl status smb-monitor smb-dashboard smb-backup caddy || true
  echo "local check: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT_INT/dashboard.html) (200 = served)"
  echo "edge without token: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT_EXT/dashboard.html) (401/403 or a redirect = protected)"
  echo "public address: http://${PUBLIC_IPADDR:-?}:$(printenv "VAST_TCP_PORT_$PORT_EXT" || echo '?')/dashboard.html (token required)"
  curl -s http://localhost:11111/get-all-quick-tunnels 2>/dev/null | python3 -c '
import json, sys
try:
    for t in json.load(sys.stdin):
        if t.get("targetUrl", "").endswith(":'"$PORT_EXT"'"):
            print("HTTPS quick tunnel (token required):", t["tunnelUrl"] + "/dashboard.html")
except Exception:
    pass' || true
}

remove() {
  supervisorctl stop smb-monitor smb-dashboard smb-backup smb-watchdog 2>/dev/null || true
  rm -f "$CONF"/smb-{monitor,dashboard,backup,watchdog}.conf "$SCRIPTS"/smb-{monitor,dashboard,backup,watchdog}.sh
  "$(yamlpy)" - "$LABEL" <<'PY'
import os, sys, tempfile, yaml
path = "/etc/portal.yaml"
d = yaml.safe_load(open(path)) or {}
d.get("applications", {}).pop(sys.argv[1], None)
fd, tmp = tempfile.mkstemp(dir="/etc", prefix=".portal.")
with os.fdopen(fd, "w") as f:
    yaml.safe_dump(d, f, sort_keys=False)
os.chmod(tmp, 0o644)
os.replace(tmp, path)
PY
  supervisorctl reread && supervisorctl update && supervisorctl restart caddy
}

case "${1:-status}" in
  install) install ;;
  status) status ;;
  remove) remove ;;
  *) echo "usage: $0 install|status|remove" >&2; exit 2 ;;
esac
