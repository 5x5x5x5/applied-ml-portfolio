#!/usr/bin/env bash
################################################################################
# RegRecord - Daily Compliance Check Script
#
# Runs the daily compliance status check, escalation, and alerting.
# Designed for Control-M scheduling in a UNIX environment.
#
# Usage:
#   compliance_check.sh --precheck    # Verify prerequisites
#   compliance_check.sh --run         # Run compliance check + escalation
#   compliance_check.sh --alerts      # Generate and send alerts
#   compliance_check.sh --full        # Run all steps
#
# Environment Variables:
#   REGRECORD_HOME      - Application home directory
#   REGRECORD_DB_URL    - Database connection URL
#   BATCH_DATE          - Processing date (default: today)
#   ALERT_THRESHOLD_DAYS - Days before due to trigger alert (default: 7)
#   LOG_DIR             - Log directory
#   REPORT_DIR          - Report output directory
################################################################################

set -euo pipefail
IFS=$'\n\t'

# --- Configuration -----------------------------------------------------------

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly REGRECORD_HOME="${REGRECORD_HOME:-/opt/regrecord}"
readonly LOG_DIR="${LOG_DIR:-/var/log/regrecord}"
readonly REPORT_DIR="${REPORT_DIR:-${REGRECORD_HOME}/reports}"
readonly BATCH_DATE="${BATCH_DATE:-$(date +%Y%m%d)}"
readonly ALERT_THRESHOLD="${ALERT_THRESHOLD_DAYS:-7}"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/compliance_${BATCH_DATE}.log"
readonly LOCK_FILE="/tmp/regrecord_compliance.lock"
readonly API_URL="${REGRECORD_API_URL:-http://localhost:8000}"
readonly PYTHON_BIN="${PYTHON_BIN:-python3}"

# Exit codes
readonly E_SUCCESS=0
readonly E_PREREQ_FAIL=1
readonly E_DB_FAIL=2
readonly E_COMPLIANCE_FAIL=3
readonly E_ALERT_FAIL=4
readonly E_LOCK_FAIL=5

# --- Logging Functions --------------------------------------------------------

log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] INFO  ${SCRIPT_NAME}: $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

log_warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] WARN  ${SCRIPT_NAME}: $1"
    echo "$msg" | tee -a "$LOG_FILE" >&2
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] ERROR ${SCRIPT_NAME}: $1"
    echo "$msg" | tee -a "$LOG_FILE" >&2
}

log_separator() {
    echo "================================================================" | tee -a "$LOG_FILE"
}

# --- Lock Management ----------------------------------------------------------

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid
        pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Another instance is running (PID: $pid). Exiting."
            exit $E_LOCK_FAIL
        else
            log_warn "Removing stale lock file for PID $pid"
            rm -f "$LOCK_FILE"
        fi
    fi
    echo $$ > "$LOCK_FILE"
    trap 'release_lock' EXIT
}

release_lock() {
    rm -f "$LOCK_FILE"
}

# --- Pre-check ----------------------------------------------------------------

run_precheck() {
    log_separator
    log_info "Starting pre-check for batch date: $BATCH_DATE"

    # 1. Check required directories
    for dir in "$LOG_DIR" "$REPORT_DIR"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir" 2>/dev/null || {
                log_error "Cannot create directory: $dir"
                return $E_PREREQ_FAIL
            }
        fi
    done
    log_info "Directories OK"

    # 2. Check Python environment
    if ! command -v "$PYTHON_BIN" &>/dev/null; then
        log_error "Python not found at: $PYTHON_BIN"
        return $E_PREREQ_FAIL
    fi
    local py_version
    py_version=$("$PYTHON_BIN" --version 2>&1)
    log_info "Python: $py_version"

    # 3. Check API health
    local health_response
    health_response=$(curl -s -o /dev/null -w "%{http_code}" \
        "${API_URL}/health" 2>/dev/null || echo "000")

    if [ "$health_response" != "200" ]; then
        log_error "API health check failed (HTTP $health_response)"
        return $E_DB_FAIL
    fi
    log_info "API health check: OK (HTTP $health_response)"

    # 4. Check disk space (require at least 1GB free)
    local free_space_kb
    free_space_kb=$(df -k "$LOG_DIR" | awk 'NR==2 {print $4}')
    if [ "$free_space_kb" -lt 1048576 ]; then
        log_warn "Low disk space: $(( free_space_kb / 1024 )) MB free"
    else
        log_info "Disk space: $(( free_space_kb / 1024 )) MB free"
    fi

    log_info "Pre-check completed successfully"
    return $E_SUCCESS
}

# --- Compliance Check ---------------------------------------------------------

run_compliance_check() {
    log_separator
    log_info "Starting compliance check for batch date: $BATCH_DATE"

    local start_time
    start_time=$(date +%s)

    # Step 1: Get current compliance summary
    log_info "Step 1: Fetching compliance summary..."
    local summary
    summary=$(curl -s -X GET "${API_URL}/api/v1/compliance/summary" \
        -H "X-User-Role: admin" \
        -H "X-User-Name: system_compliance_check" \
        2>/dev/null)

    if [ -z "$summary" ]; then
        log_error "Failed to fetch compliance summary"
        return $E_COMPLIANCE_FAIL
    fi

    log_info "Compliance Summary:"
    echo "$summary" | "$PYTHON_BIN" -m json.tool 2>/dev/null | tee -a "$LOG_FILE"

    # Step 2: Escalate overdue items
    log_info "Step 2: Escalating overdue compliance items..."
    local escalation_result
    escalation_result=$(curl -s -X POST "${API_URL}/api/v1/compliance/escalate" \
        -H "X-User-Role: admin" \
        -H "X-User-Name: system_escalation" \
        2>/dev/null)

    local escalated_count
    escalated_count=$(echo "$escalation_result" | "$PYTHON_BIN" -c \
        "import sys,json; print(json.load(sys.stdin).get('escalated_count',0))" 2>/dev/null || echo "0")
    log_info "Escalated $escalated_count compliance records"

    # Step 3: Get upcoming deadlines
    log_info "Step 3: Checking upcoming deadlines (${ALERT_THRESHOLD} days)..."
    local deadlines
    deadlines=$(curl -s -X GET \
        "${API_URL}/api/v1/compliance/deadlines?days_ahead=${ALERT_THRESHOLD}" \
        -H "X-User-Role: admin" \
        -H "X-User-Name: system_compliance_check" \
        2>/dev/null)

    local deadline_count
    deadline_count=$(echo "$deadlines" | "$PYTHON_BIN" -c \
        "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    log_info "Found $deadline_count upcoming deadlines within ${ALERT_THRESHOLD} days"

    # Step 4: Get gap analysis
    log_info "Step 4: Running gap analysis..."
    local gaps
    gaps=$(curl -s -X GET "${API_URL}/api/v1/compliance/gaps" \
        -H "X-User-Role: admin" \
        -H "X-User-Name: system_compliance_check" \
        2>/dev/null)

    local gap_count
    gap_count=$(echo "$gaps" | "$PYTHON_BIN" -c \
        "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    log_info "Compliance gaps found: $gap_count"

    # Step 5: Generate report
    log_info "Step 5: Generating compliance report..."
    local report_file="${REPORT_DIR}/compliance_${BATCH_DATE}.json"
    cat > "$report_file" <<REPORT_EOF
{
    "report_date": "$(date -Iseconds)",
    "batch_date": "$BATCH_DATE",
    "summary": $summary,
    "escalated_count": $escalated_count,
    "upcoming_deadlines": $deadlines,
    "gap_analysis": $gaps,
    "alert_threshold_days": $ALERT_THRESHOLD
}
REPORT_EOF

    log_info "Report saved to: $report_file"

    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    log_info "Compliance check completed in ${duration}s"

    return $E_SUCCESS
}

# --- Alerts -------------------------------------------------------------------

run_alerts() {
    log_separator
    log_info "Starting alert generation for batch date: $BATCH_DATE"

    local report_file="${REPORT_DIR}/compliance_${BATCH_DATE}.json"
    if [ ! -f "$report_file" ]; then
        log_error "Report file not found: $report_file. Run --run first."
        return $E_ALERT_FAIL
    fi

    # Parse critical items from report
    local critical_count
    critical_count=$("$PYTHON_BIN" -c "
import json, sys
with open('$report_file') as f:
    data = json.load(f)
deadlines = data.get('upcoming_deadlines', [])
critical = [d for d in deadlines if d.get('urgency') in ('CRITICAL', 'OVERDUE')]
print(len(critical))
" 2>/dev/null || echo "0")

    log_info "Critical items requiring immediate attention: $critical_count"

    # Generate alert summary
    local alert_file="${REPORT_DIR}/compliance_alerts_${BATCH_DATE}.txt"
    "$PYTHON_BIN" -c "
import json, sys
from datetime import datetime

with open('$report_file') as f:
    data = json.load(f)

summary = data.get('summary', {})
deadlines = data.get('upcoming_deadlines', [])
gaps = data.get('gap_analysis', [])

print('=' * 70)
print('REGRECORD DAILY COMPLIANCE ALERT REPORT')
print(f'Date: {datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")}')
print(f'Batch: ${BATCH_DATE}')
print('=' * 70)
print()
print('COMPLIANCE SUMMARY')
print('-' * 40)
print(f'  Total Active:   {summary.get(\"total\", 0)}')
print(f'  Pending:        {summary.get(\"pending\", 0)}')
print(f'  In Progress:    {summary.get(\"in_progress\", 0)}')
print(f'  Overdue:        {summary.get(\"overdue\", 0)}')
print(f'  Escalated:      {summary.get(\"escalated\", 0)}')
print(f'  Completed:      {summary.get(\"completed\", 0)}')
risk = summary.get('avg_risk_score')
print(f'  Avg Risk Score: {risk if risk else \"N/A\"}')
print()

critical = [d for d in deadlines if d.get('urgency') in ('CRITICAL', 'OVERDUE')]
if critical:
    print('CRITICAL ITEMS (Require Immediate Action)')
    print('-' * 70)
    for item in critical:
        print(f'  [{item[\"urgency\"]}] {item[\"drug_name\"]} - {item[\"requirement_type\"]}')
        print(f'    Due: {item[\"due_date\"]}  |  Responsible: {item[\"responsible_party\"]}')
        print(f'    Risk Score: {item.get(\"risk_score\", \"N/A\")}')
        print()

if gaps:
    print('COMPLIANCE GAPS')
    print('-' * 70)
    for gap in gaps[:10]:
        print(f'  [{gap[\"severity\"]}] {gap[\"drug_name\"]} - {gap[\"requirement_type\"]}')
        print(f'    Days Overdue: {gap[\"days_overdue\"]}  |  Escalation: {gap[\"escalation_level\"]}')
        print()

print('=' * 70)
print('END OF REPORT')
" > "$alert_file" 2>/dev/null

    log_info "Alert report generated: $alert_file"

    # In production, this would send emails via sendmail/mailx
    if [ "$critical_count" -gt 0 ]; then
        log_warn "CRITICAL: $critical_count items require immediate attention"
        # mail -s "CRITICAL: RegRecord Compliance Alert" regulatory-team@company.com < "$alert_file"
    fi

    log_info "Alert generation completed"
    return $E_SUCCESS
}

# --- Main ---------------------------------------------------------------------

main() {
    local mode="${1:---full}"

    # Create log directory
    mkdir -p "$LOG_DIR" 2>/dev/null || true

    log_separator
    log_info "RegRecord Compliance Check - $SCRIPT_NAME"
    log_info "Mode: $mode | Batch Date: $BATCH_DATE"
    log_info "PID: $$ | Host: $(hostname)"
    log_separator

    acquire_lock

    case "$mode" in
        --precheck)
            run_precheck
            ;;
        --run)
            run_compliance_check
            ;;
        --alerts)
            run_alerts
            ;;
        --full)
            run_precheck || exit $?
            run_compliance_check || exit $?
            run_alerts || exit $?
            ;;
        *)
            echo "Usage: $SCRIPT_NAME {--precheck|--run|--alerts|--full}" >&2
            exit 1
            ;;
    esac

    local rc=$?
    log_separator
    log_info "Compliance check finished with exit code: $rc"
    exit $rc
}

main "$@"
