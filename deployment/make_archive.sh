#!/bin/bash
# Final archive: one zip of the repository with its git history, results and reports, uploaded with
# the French-language final report files to the rclone remote of backup_remote.sh.
#   deployment/make_archive.sh            # build the zip and upload it
#   deployment/make_archive.sh --no-upload
# Left out (with how to restore each, in the zip's ARCHIVE_CONTENTS.md): .venv/ (`make setup`),
# artifacts/cache/ (rebuilt from seeds), *.pt (regenerable by their runs), the ROM (never distributed),
# artifacts/monitor/server-repo.bundle (duplicate of .git). No credentials live in the repository.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=$(basename "$PWD")
REMOTE="${BACKUP_REMOTE:-smbbackup:SMB_Flywire_Speedrun_backups}"
STAMP=$(date -u +%Y-%m-%d_%H%MUTC)
OUTDIR="$(dirname "$PWD")/archives"
ZIP="$OUTDIR/${NAME}_final_${STAMP}.zip"
mkdir -p "$OUTDIR"

cat > ARCHIVE_CONTENTS.md <<EOF
# Contents of this archive

Built $(date -u +%Y-%m-%dT%H:%M:%SZ) on Vast instance ${CONTAINER_ID:-?} by \`deployment/make_archive.sh\`, at git
commit $(git rev-parse --short HEAD) ($(git log -1 --format=%cI)).

**Latest mission (2026-10-20), ACTIVITY_MATCH_V1:** \`docs/activity_match/FINAL_REPORT_FR.md\` (French), its
frozen protocol \`docs/activity_match/PROTOCOL_FROZEN.md\` and all its data under \`artifacts/activity_match/\`.
Previous mission (2026-10-19), RETINO_ALIGN_V1: \`docs/retino_align/FINAL_REPORT_FR.md\`, data under
\`artifacts/retino_align/\`.

**Start here:** \`README.md\`, then \`ARCHITECTURE.md\` and \`docs/REPRODUCIBILITY.md\`.

Included: the whole repository with its git history (\`.git\`), the connectome data (\`data/connectome\`), every
run's outputs and logs (\`artifacts/runs\`, \`artifacts/deploy_runs\`, \`artifacts/monitor\`) and figures.

Left out:

| what | size | how to get it back |
|---|---|---|
| \`.venv/\` (Python environment) | 8.6 GB | \`make setup\` (see README.md) |
| \`artifacts/cache/\` (null realisations, caches) | 8.2 GB | rebuilt from their seeds on first use |
| \`*.pt\` (checkpoints, network states) | 3.5 GB | rerun the run that wrote it; most are in the Drive backup folder |
| \`roms/\`, \`*.nes\` (game ROM) | - | never distributed: use your own legal copy |
| \`artifacts/monitor/server-repo.bundle\` | 22 MB | duplicate of \`.git\` |
EOF

rm -f "$ZIP"
( cd .. && zip -r -q -y "$ZIP" "$NAME" \
    -x "$NAME/.venv/*" "$NAME/artifacts/cache/*" "$NAME/roms/*" "*.nes" "*.pt" \
       "*/__pycache__/*" "$NAME/.pytest_cache/*" "$NAME/.ruff_cache/*" "$NAME/artifacts/monitor/server-repo.bundle" )
rm -f ARCHIVE_CONTENTS.md
zip -T -q "$ZIP" && echo "zip ok: $ZIP ($(du -h "$ZIP" | cut -f1), $(unzip -Z1 "$ZIP" | wc -l) entries)"
if unzip -Z1 "$ZIP" | grep -qiE '\.nes$|\.pt$|/\.venv/'; then echo "ERROR: excluded files found in the zip" >&2; exit 1; fi

[ "${1:-}" = "--no-upload" ] && exit 0
rclone copy "$ZIP" "$REMOTE/" --log-level NOTICE
local_md5=$(md5sum "$ZIP" | cut -d' ' -f1)
remote_md5=$(rclone md5sum "$REMOTE/$(basename "$ZIP")" | cut -d' ' -f1)
[ "$local_md5" = "$remote_md5" ] && echo "upload verified: $REMOTE/$(basename "$ZIP") (md5 $local_md5)" \
  || { echo "ERROR: md5 mismatch (local $local_md5, remote ${remote_md5:-missing})" >&2; exit 1; }
