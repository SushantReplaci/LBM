#!/usr/bin/env bash
# =============================================================================
# run_training.sh — LBM Removal Training Runner
#
# Features:
#   • Sends Slack notifications on start, success, and failure
#   • Shuts down the instance on success OR failure
#   • Logs all output to a timestamped log file
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# CONFIGURATION — edit these before running
# ---------------------------------------------------------------------------

# Slack Incoming Webhook URL
SLACK_WEBHOOK_URL=""

# Path to the config YAML  (passed to main_from_config)
CONFIG_YAML="/workspace/LBM/examples/training/config/removal.yaml"

# Working directory (repo root so relative imports resolve)
WORKDIR="/workspace/LBM"

# Python interpreter / environment
PYTHON="${PYTHON:-python}"

# Log directory & file
LOG_DIR="${WORKDIR}/logs/run_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"

# Hostname / instance identifier shown in Slack messages
INSTANCE_ID="$(hostname)"

# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

# Send a Slack message via the webhook.
# Usage: slack_notify <emoji> <title> <body>
slack_notify() {
    local emoji="$1"
    local title="$2"
    local body="$3"

    local payload
    payload=$(cat <<EOF
{
    "text": "${emoji} *${title}*",
    "blocks": [
        {
            "type": "header",
            "text": { "type": "plain_text", "text": "${emoji} ${title}", "emoji": true }
        },
        {
            "type": "section",
            "fields": [
                { "type": "mrkdwn", "text": "*Instance:*\n\`${INSTANCE_ID}\`" },
                { "type": "mrkdwn", "text": "*Time:*\n$(date '+%Y-%m-%d %H:%M:%S UTC')" }
            ]
        },
        {
            "type": "section",
            "text": { "type": "mrkdwn", "text": "${body}" }
        }
    ]
}
EOF
)

    curl -s -X POST \
        -H "Content-Type: application/json" \
        --data "$payload" \
        "$SLACK_WEBHOOK_URL" || true   # never fail the script because of Slack
}

# Shut down the machine gracefully.
shutdown_instance() {
    echo "[$(date '+%H:%M:%S')] Initiating instance shutdown..."
    # Works on most Linux cloud instances; comment out if not desired.
    sudo shutdown -h now || poweroff || true
}

# ---------------------------------------------------------------------------
# TRAP — runs whenever the script exits (success, error, or signal)
# ---------------------------------------------------------------------------

EXIT_CODE=0   # updated by on_exit

on_exit() {
    EXIT_CODE=$?

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[$(date '+%H:%M:%S')] Training finished successfully."

        slack_notify \
            ":white_check_mark:" \
            "Training Completed Successfully" \
            "*Config:* \`${CONFIG_YAML}\`\n*Log:* \`${LOG_FILE}\`\n\nThe instance will shut down now."

    else
        local last_lines
        last_lines=$(tail -n 30 "$LOG_FILE" 2>/dev/null | sed 's/"/\\"/g; s/`/\\`/g' || echo "(log unavailable)")

        echo "[$(date '+%H:%M:%S')] Training FAILED with exit code ${EXIT_CODE}."

        slack_notify \
            ":x:" \
            "Training FAILED (exit code ${EXIT_CODE})" \
            "*Config:* \`${CONFIG_YAML}\`\n*Log:* \`${LOG_FILE}\`\n\n*Last 30 log lines:*\n\`\`\`\n${last_lines}\n\`\`\`\n\nThe instance will shut down now."
    fi

    shutdown_instance
}

trap on_exit EXIT

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

echo "[$(date '+%H:%M:%S')] Starting LBM Removal training..." | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Config : ${CONFIG_YAML}"           | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Log    : ${LOG_FILE}"              | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Instance: ${INSTANCE_ID}"          | tee -a "$LOG_FILE"

# Verify the config file exists before even notifying Slack
if [[ ! -f "$CONFIG_YAML" ]]; then
    echo "[ERROR] Config file not found: ${CONFIG_YAML}" | tee -a "$LOG_FILE"
    exit 1
fi

# Notify Slack that training is starting
slack_notify \
    ":rocket:" \
    "LBM Removal Training Started" \
    "*Config:* \`${CONFIG_YAML}\`\n*Log:* \`${LOG_FILE}\`\n\nTraining has begun — you'll be notified when it finishes."

echo "[$(date '+%H:%M:%S')] Slack start notification sent." | tee -a "$LOG_FILE"
echo "-----------------------------------------------------------" | tee -a "$LOG_FILE"

# Run training — pipe stdout+stderr to both the terminal and the log file
cd "$WORKDIR"
"$PYTHON" examples/training/train_lbm_removal.py \
    --path_config "$CONFIG_YAML" \
    2>&1 | tee -a "$LOG_FILE"

# If set -e didn't catch a non-zero exit from the pipe, check manually.
# (pipefail is set so the exit code of the python command propagates.)
echo "-----------------------------------------------------------" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Python process exited."              | tee -a "$LOG_FILE"

# EXIT trap fires here automatically →
