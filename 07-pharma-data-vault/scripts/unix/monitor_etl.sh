#!/usr/bin/env bash
###############################################################################
# PharmaDataVault - ETL Monitoring Script
#
# Monitors the ETL pipeline status and alerts on issues:
#   - Check ETL job completion status
#   - Verify row counts in key tables
#   - Alert on SLA breaches
#   - Monitor database space and connections
#   - Check data freshness
#
# Usage:
#   monitor_etl.sh --check <check_type> [--date YYYYMMDD]
#
# Check Types:
#   sla             - Check if daily ETL completed within SLA
#   row_counts      - Verify expected row counts in vault tables
#   mart_quality    - Post-mart-refresh quality validation
#   freshness       - Check data freshness (last load dates)
#   all             - Run all checks
#
# Exit Codes:
#   0 - All checks passed
#   1 - Warning (non-critical issues)
#   2 - Critical alert
#   8 - Invalid arguments
###############################################################################

set -euo pipefail

readonly SCRIPT_NAME="$(basename "$0")"
readonly LOG_DIR="/var/log/pharma_vault"
readonly DB_USER="etl_user"
readonly DB_TNS="pharma_vault"
readonly SQLPLUS_CONN="${DB_USER}/@${DB_TNS}"

PROCESS_DATE="$(date +%Y%m%d)"
CHECK_TYPE=""
OVERALL_STATUS=0

# SLA threshold: daily ETL must complete by this hour (24h format)
readonly SLA_HOUR=6
readonly SLA_MINUTE=0

# Row count thresholds (minimum expected per day)
readonly MIN_HUB_DRUG_DAILY=0          # Some days may have no new drugs
readonly MIN_SAT_AE_DAILY=10           # Expect at least 10 AEs per day
readonly MIN_FACT_ENROLLMENT_TOTAL=100  # Minimum total enrollments in mart

# Data freshness threshold (hours)
readonly MAX_DATA_AGE_HOURS=26  # Allow 2 extra hours past 24h cycle

# =============================================================================
# Logging
# =============================================================================

readonly LOG_FILE="${LOG_DIR}/monitor_etl_${PROCESS_DATE}.log"

log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  MONITOR: $1"
    echo "${msg}" | tee -a "${LOG_FILE}"
}

log_warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  MONITOR: $1"
    echo "${msg}" | tee -a "${LOG_FILE}" >&2
    [ ${OVERALL_STATUS} -lt 1 ] && OVERALL_STATUS=1
}

log_critical() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [CRIT]  MONITOR: $1"
    echo "${msg}" | tee -a "${LOG_FILE}" >&2
    OVERALL_STATUS=2
}

send_alert() {
    local severity="$1"
    local subject="$2"
    local body="$3"
    local recipients="data-eng@company.com"

    if [ "${severity}" = "CRITICAL" ]; then
        recipients="data-eng@company.com;pharma-dba@company.com;etl-oncall@company.com"
    fi

    if command -v mailx &>/dev/null; then
        echo "${body}" | mailx -s "[${severity}] PharmaVault: ${subject}" "${recipients}" 2>/dev/null || true
    fi
}

run_sql_query() {
    local query="$1"
    sqlplus -S "${SQLPLUS_CONN}" <<-EOSQL
        SET HEADING OFF FEEDBACK OFF VERIFY OFF PAGESIZE 0 LINESIZE 200 TRIMSPOOL ON
        ${query}
        EXIT;
EOSQL
}

# =============================================================================
# Check Functions
# =============================================================================

check_sla() {
    log_info "=== SLA COMPLIANCE CHECK ==="

    local current_hour
    current_hour=$(date +%H)
    local current_minute
    current_minute=$(date +%M)

    # Check if daily ETL batch completed
    local etl_status
    etl_status=$(run_sql_query "
        SELECT STATUS || '|' || TO_CHAR(END_TIMESTAMP, 'HH24:MI:SS')
        FROM ETL_BATCH_CONTROL
        WHERE BATCH_TYPE = 'DRUG'
          AND TRUNC(START_TIMESTAMP) = TO_DATE('${PROCESS_DATE}', 'YYYYMMDD')
          AND ROWNUM = 1
        ORDER BY BATCH_ID DESC;
    " | tr -d '[:space:]')

    if [ -z "${etl_status}" ]; then
        if [ "${current_hour}" -ge "${SLA_HOUR}" ]; then
            log_critical "SLA BREACH: No ETL batch found for ${PROCESS_DATE} and it is past ${SLA_HOUR}:00"
            send_alert "CRITICAL" "SLA Breach - No ETL batch" \
                "No ETL batch was found for ${PROCESS_DATE}. Current time: $(date '+%H:%M'). SLA: ${SLA_HOUR}:00."
        else
            log_info "No ETL batch yet for ${PROCESS_DATE}, but within SLA window"
        fi
        return
    fi

    local batch_status
    batch_status=$(echo "${etl_status}" | cut -d'|' -f1)
    local completion_time
    completion_time=$(echo "${etl_status}" | cut -d'|' -f2)

    if [ "${batch_status}" = "FAILED" ]; then
        log_critical "ETL batch FAILED for ${PROCESS_DATE}"
        send_alert "CRITICAL" "ETL Batch Failed" \
            "Daily ETL batch failed for ${PROCESS_DATE}. Check ETL_ERROR_LOG for details."
    elif [ "${batch_status}" = "SUCCESS" ]; then
        local completion_hour
        completion_hour=$(echo "${completion_time}" | cut -d':' -f1)
        if [ "${completion_hour}" -gt "${SLA_HOUR}" ]; then
            log_warn "SLA WARNING: ETL completed at ${completion_time} (SLA: ${SLA_HOUR}:00)"
            send_alert "WARNING" "SLA Near-Miss" \
                "ETL completed at ${completion_time}, after SLA deadline of ${SLA_HOUR}:00."
        else
            log_info "SLA met: ETL completed at ${completion_time}"
        fi
    elif [ "${batch_status}" = "RUNNING" ]; then
        if [ "${current_hour}" -ge "${SLA_HOUR}" ]; then
            log_critical "SLA BREACH: ETL still running at $(date '+%H:%M')"
            send_alert "CRITICAL" "SLA Breach - ETL Still Running" \
                "ETL is still running at $(date '+%H:%M'). SLA deadline: ${SLA_HOUR}:00."
        else
            log_info "ETL batch is running (within SLA window)"
        fi
    fi
}

check_row_counts() {
    log_info "=== ROW COUNT VERIFICATION ==="

    # Check hub table counts
    local tables=("HUB_DRUG" "HUB_PATIENT" "HUB_CLINICAL_TRIAL" "HUB_ADVERSE_EVENT" "HUB_FACILITY")

    for table in "${tables[@]}"; do
        local count
        count=$(run_sql_query "SELECT COUNT(*) FROM ${table};" | tr -d '[:space:]')
        log_info "  ${table}: ${count} total rows"

        if [ -z "${count}" ] || [ "${count}" = "0" ]; then
            log_warn "${table} has ZERO rows"
        fi
    done

    # Check today's load counts
    local today_ae_count
    today_ae_count=$(run_sql_query "
        SELECT COUNT(*)
        FROM SAT_ADVERSE_EVENT_DETAILS
        WHERE TRUNC(LOAD_DATE) = TRUNC(SYSTIMESTAMP);
    " | tr -d '[:space:]')

    log_info "  Today's SAT_ADVERSE_EVENT_DETAILS loads: ${today_ae_count}"

    if [ -n "${today_ae_count}" ] && [ "${today_ae_count}" -lt "${MIN_SAT_AE_DAILY}" ]; then
        log_warn "Low AE count today: ${today_ae_count} (expected >= ${MIN_SAT_AE_DAILY})"
    fi

    # Check fact table totals
    local fact_enrollment_count
    fact_enrollment_count=$(run_sql_query "SELECT COUNT(*) FROM FACT_TRIAL_ENROLLMENT;" | tr -d '[:space:]')
    log_info "  FACT_TRIAL_ENROLLMENT total: ${fact_enrollment_count}"

    if [ -n "${fact_enrollment_count}" ] && [ "${fact_enrollment_count}" -lt "${MIN_FACT_ENROLLMENT_TOTAL}" ]; then
        log_warn "Low enrollment fact count: ${fact_enrollment_count} (expected >= ${MIN_FACT_ENROLLMENT_TOTAL})"
    fi

    # Check for ETL errors today
    local error_count
    error_count=$(run_sql_query "
        SELECT COUNT(*)
        FROM ETL_ERROR_LOG
        WHERE TRUNC(ERROR_TIMESTAMP) = TRUNC(SYSTIMESTAMP)
          AND ERROR_CODE != 0;
    " | tr -d '[:space:]')

    log_info "  ETL errors today: ${error_count}"
    if [ -n "${error_count}" ] && [ "${error_count}" -gt 100 ]; then
        log_warn "High ETL error count today: ${error_count}"
    fi
}

check_freshness() {
    log_info "=== DATA FRESHNESS CHECK ==="

    # Check last load date for each satellite table
    local sat_tables=("SAT_DRUG_DETAILS" "SAT_PATIENT_DEMOGRAPHICS" "SAT_CLINICAL_TRIAL_DETAILS" "SAT_ADVERSE_EVENT_DETAILS" "SAT_DRUG_MANUFACTURING")

    for table in "${sat_tables[@]}"; do
        local last_load
        last_load=$(run_sql_query "
            SELECT TO_CHAR(MAX(LOAD_DATE), 'YYYY-MM-DD HH24:MI:SS')
            FROM ${table};
        " | tr -d '[:space:]')

        local hours_old
        hours_old=$(run_sql_query "
            SELECT ROUND((SYSDATE - CAST(MAX(LOAD_DATE) AS DATE)) * 24, 1)
            FROM ${table};
        " | tr -d '[:space:]')

        log_info "  ${table}: last load=${last_load}, age=${hours_old}h"

        if [ -n "${hours_old}" ]; then
            # Compare as integers (remove decimal)
            local hours_int
            hours_int=${hours_old%%.*}
            if [ "${hours_int}" -gt "${MAX_DATA_AGE_HOURS}" ]; then
                log_warn "${table} data is stale: ${hours_old} hours old (threshold: ${MAX_DATA_AGE_HOURS}h)"
            fi
        fi
    done
}

check_mart_quality() {
    log_info "=== DATA MART QUALITY CHECK ==="

    # Orphan fact records (FK integrity)
    local orphan_patients
    orphan_patients=$(run_sql_query "
        SELECT COUNT(*)
        FROM FACT_TRIAL_ENROLLMENT fe
        WHERE NOT EXISTS (
            SELECT 1 FROM DIM_PATIENT dp
            WHERE dp.DIM_PATIENT_KEY = fe.DIM_PATIENT_KEY
        );
    " | tr -d '[:space:]')

    log_info "  Orphan patient FKs in FACT_TRIAL_ENROLLMENT: ${orphan_patients}"
    if [ -n "${orphan_patients}" ] && [ "${orphan_patients}" -gt 0 ]; then
        log_warn "Found ${orphan_patients} orphan patient references in enrollment facts"
    fi

    local orphan_drugs
    orphan_drugs=$(run_sql_query "
        SELECT COUNT(*)
        FROM FACT_ADVERSE_EVENTS fa
        WHERE NOT EXISTS (
            SELECT 1 FROM DIM_DRUG dd
            WHERE dd.DIM_DRUG_KEY = fa.DIM_DRUG_KEY
        );
    " | tr -d '[:space:]')

    log_info "  Orphan drug FKs in FACT_ADVERSE_EVENTS: ${orphan_drugs}"
    if [ -n "${orphan_drugs}" ] && [ "${orphan_drugs}" -gt 0 ]; then
        log_warn "Found ${orphan_drugs} orphan drug references in AE facts"
    fi

    # Dimension currency check
    local stale_dims
    stale_dims=$(run_sql_query "
        SELECT COUNT(*)
        FROM DIM_DRUG
        WHERE IS_CURRENT = 'Y'
          AND EFFECTIVE_DATE < SYSDATE - 365;
    " | tr -d '[:space:]')

    log_info "  Dimension records >1 year without update: ${stale_dims}"

    # MV refresh status
    local mv_status
    mv_status=$(run_sql_query "
        SELECT MVIEW_NAME || ': ' || TO_CHAR(LAST_REFRESH_DATE, 'YYYY-MM-DD HH24:MI')
        FROM USER_MVIEWS
        WHERE MVIEW_NAME LIKE 'MV_%'
        ORDER BY MVIEW_NAME;
    ")
    log_info "  Materialized view refresh status:"
    echo "${mv_status}" | while read -r line; do
        [ -n "${line}" ] && log_info "    ${line}"
    done

    log_info "Mart quality check completed"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    cat <<-EOF
Usage: ${SCRIPT_NAME} --check <check_type> [--date YYYYMMDD]

Check Types:
  sla             Check SLA compliance
  row_counts      Verify row counts
  mart_quality    Post-mart quality validation
  freshness       Check data freshness
  all             Run all checks
EOF
    exit 8
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --check)    CHECK_TYPE="$2"; shift 2 ;;
            --date)     PROCESS_DATE="$2"; shift 2 ;;
            --help|-h)  usage ;;
            *)          log_critical "Unknown argument: $1"; usage ;;
        esac
    done

    [ -z "${CHECK_TYPE}" ] && { log_critical "Check type not specified"; usage; }

    mkdir -p "${LOG_DIR}"

    log_info "======================================================="
    log_info "PharmaDataVault ETL Monitor - Check: ${CHECK_TYPE}"
    log_info "Date: ${PROCESS_DATE} | Host: $(hostname)"
    log_info "======================================================="

    case "${CHECK_TYPE}" in
        sla)           check_sla ;;
        row_counts)    check_row_counts ;;
        mart_quality)  check_mart_quality ;;
        freshness)     check_freshness ;;
        all)
            check_sla
            check_row_counts
            check_freshness
            check_mart_quality
            ;;
        *)
            log_critical "Unknown check type: ${CHECK_TYPE}"
            usage
            ;;
    esac

    log_info "======================================================="
    if [ ${OVERALL_STATUS} -eq 0 ]; then
        log_info "All checks PASSED"
    elif [ ${OVERALL_STATUS} -eq 1 ]; then
        log_warn "Checks completed with WARNINGS"
    else
        log_critical "Checks completed with CRITICAL issues"
    fi
    log_info "======================================================="

    exit ${OVERALL_STATUS}
}

main "$@"
