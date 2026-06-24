#!/bin/bash
# Nightly git backup. Mirrors the Spring 2026 backup script.
# git add -A && commit && push, logged to .backup.log (gitignored). No-op safe.
set -u
REPO="/Users/vsreenivas/Documents/Claude/Projects/Teaching"
LOG="$REPO/.backup.log"
DATE_ISO=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
DATE_TAG=$(date +"%Y-%m-%d")
log() { printf '%s\t%s\t%s\n' "$DATE_ISO" "$1" "$2" >> "$LOG"; }
if [ ! -d "$REPO/.git" ]; then log "REPO_MISSING" "no .git at $REPO"; exit 1; fi
cd "$REPO" || { log "REPO_MISSING" "cd failed"; exit 1; }
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
git add -A 2>>"$LOG"
if git diff --cached --quiet; then log "NOOP" "no changes"; exit 0; fi
if ! git commit -m "nightly backup $DATE_TAG" >> "$LOG" 2>&1; then log "COMMIT_FAIL" "commit non-zero"; exit 1; fi
if ! git push origin main >> "$LOG" 2>&1; then log "PUSH_FAIL" "push non-zero"; exit 1; fi
log "OK" "nightly backup $DATE_TAG pushed"
exit 0
