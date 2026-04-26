#!/usr/bin/env bash
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ Claude Code — PreToolUse Safety Hook                                       │
# │ Blocks dangerous commands even when --dangerously-skip-permissions is on.  │
# │ Exit 0 = allow | Exit 2 = block (reason printed to stderr)                │
# └─────────────────────────────────────────────────────────────────────────────┘

set -euo pipefail

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
FILE_PATH=$(echo "$INPUT" | jq -r '.tool_input.file_path // empty')

block() {
  echo "BLOCKED: $1" >&2
  exit 2
}

# ── Bash command guards ──────────────────────────────────────────────────────

if [ "$TOOL" = "Bash" ] && [ -n "$COMMAND" ]; then

  # Destructive file operations
  echo "$COMMAND" | grep -qE 'rm\s+-(rf|fr|r\s+-f|f\s+-r)\b' \
    && block "Recursive force delete (rm -rf)"

  echo "$COMMAND" | grep -qE '(shred|wipefs|mkfs)\b' \
    && block "Disk/file destruction command"

  # Sudo
  echo "$COMMAND" | grep -qE '(^|\||\;|&&)\s*sudo\b' \
    && block "sudo is not allowed"

  # Git — force push
  echo "$COMMAND" | grep -qE 'git\s+push\s+.*(-f|--force)' \
    && block "Force push"

  # Git — destructive reset / clean
  echo "$COMMAND" | grep -qE 'git\s+reset\s+--hard' \
    && block "git reset --hard"
  echo "$COMMAND" | grep -qE 'git\s+clean\s+-f' \
    && block "git clean -f"

  # Git — direct push to main/master
  echo "$COMMAND" | grep -qE 'git\s+push\s+\S+\s+(main|master)\b' \
    && block "Direct push to main/master"

  # Dangerous permissions
  echo "$COMMAND" | grep -qE 'chmod\s+(777|666|a\+w)\b' \
    && block "Overly permissive chmod"
  echo "$COMMAND" | grep -qE 'chown\s+root\b' \
    && block "chown to root"

  # Database destructive ops
  echo "$COMMAND" | grep -qiE '(drop\s+(table|database|schema)|truncate\s+table)' \
    && block "Destructive database operation"
  echo "$COMMAND" | grep -qiE 'delete\s+from\s+\S+\s*;' \
    && block "Unscoped DELETE (missing WHERE clause)"

  # System killers
  echo "$COMMAND" | grep -qE '(killall|pkill\s+-9|kill\s+-9\s+-1|shutdown|reboot|init\s+[06])' \
    && block "System/process killer"

  # ── fsvlm-specific guards ──────────────────────────────────────────────

  # Never delete user's training data
  echo "$COMMAND" | grep -qE 'rm\s.*\.(jpg|jpeg|png|bmp|tiff|webp)\b' \
    && block "Deleting image files is not allowed"

  # Never delete trained adapters
  echo "$COMMAND" | grep -qE 'rm\s.*\.fsvlm/adapters' \
    && block "Deleting trained adapters is not allowed"

  # Never delete downloaded models
  echo "$COMMAND" | grep -qE 'rm\s.*\.fsvlm/models' \
    && block "Deleting downloaded models is not allowed"

fi

# ── File write guards ────────────────────────────────────────────────────────

if [ "$TOOL" = "Edit" ] || [ "$TOOL" = "Write" ]; then
  echo "$FILE_PATH" | grep -qiE '(\.env(\..*)?$|credentials|secrets|\.pem$|\.key$|id_rsa)' \
    && block "Writing to sensitive file: $FILE_PATH"
fi

exit 0
