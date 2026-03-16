#!/usr/bin/env bash
###############################################################################
# PharmaDataVault - Daily ETL Shell Script
#
# Called by Control-M (or cron) to execute the daily ETL pipeline phases.
# Each phase performs specific operations and returns exit codes for
# the scheduler to evaluate.
#
# Usage:
#   daily_etl.sh --phase <phase_name> [--date YYYYMMDD]
#
# Phases:
#   check_files      - Verify daily feed files have arrived
#   stage_drugs      - Stage drug feed data
#   stage_patients   - Stage patient demographic data
#   stage_trials_ae  - Stage trial and adverse event data
#   load_vault       - Load raw vault (hubs, links, satellites)
#   quality_checks   - Run data quality validation
#   business_vault   - Refresh PIT tables and bridge tables
#   archive          - Archive files and clean staging
#   pre_mart_validate - Pre-mart refresh validation
#
# Exit Codes:
#   0 - Success
#   1 - Configuration error
#   2 - File arrival failure
#   3 - Database connection failure
#   4 - ETL execution failure
#   5 - Data quality failure (warning)
#   8 - Invalid arguments
###############################################################################

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly PROJECT_ROOT="/opt/pharma_vault"

# Directories
readonly STAGING_DIR="/data/pharma/staging"
readonly ARCHIVE_DIR="/data/pharma/archive"
readonly LOG_DIR="/var/log/pharma_vault"
readonly SQL_DIR="${PROJECT_ROOT}/sql"
readonly LOCK_DIR="/var/run/pharma_vault"

# Database connection
readonly DB_USER="etl_user"
readonly DB_TNS="pharma_vault"
readonly SQLPLUS_CONN="${DB_USER}/@${DB_TNS}"

# File patterns for each feed type
readonly DRUG_PATTERN="drug_feed_*.csv"
readonly PATIENT_PATTERN="patient_feed_*.csv"
readonly TRIAL_PATTERN="trial_feed_*.csv"
readonly AE_PATTERN="ae_feed_*.csv"
readonly MFG_PATTERN="mfg_feed_*.dat"

# Thresholds
readonly FILE_WAIT_TIMEOUT=3600     # 1 hour max wait for files
readonly FILE_CHECK_INTERVAL=60     # Check every 60 seconds
readonly MIN_FILE_SIZE=100          # Minimum file size in bytes
readonly MAX_SQLPLUS_RETRIES=2
readonly STAGING_RETENTION_DAYS=30  # Keep staging data for 30 days

# Today's date (override with --date)
PROCESS_DATE="$(date +%Y%m%d)"

# =============================================================================
# Logging Functions
# =============================================================================

readonly LOG_FILE="${LOG_DIR}/daily_etl_${PROCESS_DATE}.log"

log_info() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [INFO]  ${SCRIPT_NAME}: $1"
    echo "${msg}" | tee -a "${LOG_FILE}"
}

log_warn() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [WARN]  ${SCRIPT_NAME}: $1"
    echo "${msg}" | tee -a "${LOG_FILE}" >&2
}

log_error() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] ${SCRIPT_NAME}: $1"
    echo "${msg}" | tee -a "${LOG_FILE}" >&2
}

log_separator() {
    echo "========================================================================" >> "${LOG_FILE}"
}

# =============================================================================
# Utility Functions
# =============================================================================

send_notification() {
    local subject="$1"
    local body="$2"
    local recipients="${3:-data-eng@company.com}"

    if command -v mailx &>/dev/null; then
        echo "${body}" | mailx -s "${subject}" "${recipients}" 2>/dev/null || true
    else
        log_warn "mailx not available; notification not sent: ${subject}"
    fi
}

acquire_lock() {
    local lock_name="$1"
    local lock_file="${LOCK_DIR}/${lock_name}.lock"

    mkdir -p "${LOCK_DIR}"

    if [ -f "${lock_file}" ]; then
        local lock_pid
        lock_pid=$(cat "${lock_file}" 2>/dev/null || echo "0")
        if kill -0 "${lock_pid}" 2>/dev/null; then
            log_error "Another instance is running (PID ${lock_pid}). Lock: ${lock_file}"
            return 1
        else
            log_warn "Stale lock file found. Removing: ${lock_file}"
            rm -f "${lock_file}"
        fi
    fi

    echo $$ > "${lock_file}"
    log_info "Lock acquired: ${lock_file} (PID $$)"
    return 0
}

release_lock() {
    local lock_name="$1"
    local lock_file="${LOCK_DIR}/${lock_name}.lock"

    rm -f "${lock_file}"
    log_info "Lock released: ${lock_file}"
}

check_db_connection() {
    log_info "Testing database connectivity..."

    local result
    result=$(sqlplus -S "${SQLPLUS_CONN}" <<-EOSQL
        SET HEADING OFF FEEDBACK OFF VERIFY OFF
        SELECT 'DB_OK' FROM DUAL;
        EXIT;
EOSQL
    )

    if echo "${result}" | grep -q "DB_OK"; then
        log_info "Database connection verified successfully"
        return 0
    else
        log_error "Database connection failed: ${result}"
        return 3
    fi
}

run_sqlplus() {
    local sql_file="$1"
    local description="$2"
    local attempt=0
    local result

    while [ ${attempt} -lt ${MAX_SQLPLUS_RETRIES} ]; do
        attempt=$((attempt + 1))
        log_info "Executing ${description} (attempt ${attempt}/${MAX_SQLPLUS_RETRIES})"

        result=$(sqlplus -S "${SQLPLUS_CONN}" <<-EOSQL
            SET SERVEROUTPUT ON SIZE UNLIMITED
            SET LINESIZE 200
            SET TRIMSPOOL ON
            WHENEVER SQLERROR EXIT SQL.SQLCODE
            WHENEVER OSERROR EXIT FAILURE

            @${sql_file}

            EXIT SUCCESS;
EOSQL
        )
        local exit_code=$?

        # Log the SQL*Plus output
        echo "${result}" >> "${LOG_FILE}"

        if [ ${exit_code} -eq 0 ]; then
            log_info "${description} completed successfully"
            return 0
        else
            log_warn "${description} failed with exit code ${exit_code}"
            if [ ${attempt} -lt ${MAX_SQLPLUS_RETRIES} ]; then
                log_info "Retrying in 30 seconds..."
                sleep 30
            fi
        fi
    done

    log_error "${description} failed after ${MAX_SQLPLUS_RETRIES} attempts"
    return 4
}

validate_file() {
    local filepath="$1"
    local filename
    filename=$(basename "${filepath}")

    # Check file exists and is readable
    if [ ! -r "${filepath}" ]; then
        log_error "File not readable: ${filepath}"
        return 1
    fi

    # Check minimum file size
    local file_size
    file_size=$(stat -c%s "${filepath}" 2>/dev/null || stat -f%z "${filepath}" 2>/dev/null)
    if [ "${file_size}" -lt "${MIN_FILE_SIZE}" ]; then
        log_error "File too small (${file_size} bytes < ${MIN_FILE_SIZE}): ${filename}"
        return 1
    fi

    # Verify checksum if .md5 companion exists
    local md5_file="${filepath}.md5"
    if [ -f "${md5_file}" ]; then
        local expected_md5
        expected_md5=$(awk '{print $1}' "${md5_file}")
        local actual_md5
        actual_md5=$(md5sum "${filepath}" | awk '{print $1}')

        if [ "${expected_md5}" != "${actual_md5}" ]; then
            log_error "Checksum mismatch for ${filename}: expected=${expected_md5}, actual=${actual_md5}"
            return 1
        fi
        log_info "Checksum verified for ${filename}"
    fi

    log_info "File validated: ${filename} (${file_size} bytes)"
    return 0
}

# =============================================================================
# Phase Functions
# =============================================================================

phase_check_files() {
    log_info "Phase: FILE ARRIVAL CHECK for date ${PROCESS_DATE}"
    log_separator

    local elapsed=0
    local files_found=0
    local expected_files=0

    while [ ${elapsed} -lt ${FILE_WAIT_TIMEOUT} ]; do
        files_found=0
        expected_files=0

        # Check for each feed type
        for pattern in "${DRUG_PATTERN}" "${PATIENT_PATTERN}" "${TRIAL_PATTERN}" "${AE_PATTERN}"; do
            expected_files=$((expected_files + 1))
            local matching_files
            matching_files=$(find "${STAGING_DIR}" -maxdepth 1 -name "${pattern}" -newer "${STAGING_DIR}" -type f 2>/dev/null | wc -l)
            if [ "${matching_files}" -gt 0 ]; then
                files_found=$((files_found + 1))
            fi
        done

        if [ ${files_found} -ge ${expected_files} ]; then
            log_info "All ${expected_files} expected feed files found"
            break
        fi

        log_info "Waiting for files: ${files_found}/${expected_files} found (${elapsed}s elapsed)"
        sleep "${FILE_CHECK_INTERVAL}"
        elapsed=$((elapsed + FILE_CHECK_INTERVAL))
    done

    if [ ${files_found} -lt ${expected_files} ]; then
        log_error "Timeout: only ${files_found}/${expected_files} files arrived after ${FILE_WAIT_TIMEOUT}s"
        send_notification \
            "[CRITICAL] PharmaVault: Feed files missing" \
            "Only ${files_found}/${expected_files} daily feed files arrived by timeout. Date: ${PROCESS_DATE}"
        return 2
    fi

    # Validate all found files
    local validation_failures=0
    for filepath in "${STAGING_DIR}"/*_feed_*.{csv,dat} 2>/dev/null; do
        [ -f "${filepath}" ] || continue
        if ! validate_file "${filepath}"; then
            validation_failures=$((validation_failures + 1))
        fi
    done

    if [ ${validation_failures} -gt 0 ]; then
        log_error "${validation_failures} file(s) failed validation"
        return 2
    fi

    log_info "File arrival check completed: all files present and valid"
    return 0
}

phase_stage_drugs() {
    log_info "Phase: STAGE DRUG DATA"
    log_separator

    check_db_connection || return 3

    run_sqlplus \
        "${SQL_DIR}/etl/staging/stage_drug_data.sql" \
        "Drug data staging and cleansing" || return 4

    log_info "Drug staging phase completed"
    return 0
}

phase_stage_patients() {
    log_info "Phase: STAGE PATIENT DATA"
    log_separator

    check_db_connection || return 3

    # Call staging procedure via SQL*Plus anonymous block
    local result
    result=$(sqlplus -S "${SQLPLUS_CONN}" <<-EOSQL
        SET SERVEROUTPUT ON SIZE UNLIMITED
        WHENEVER SQLERROR EXIT SQL.SQLCODE

        DECLARE
            v_batch_id NUMBER;
            v_rows     NUMBER;
        BEGIN
            SP_STAGE_PATIENT_DATA(
                p_source_file   => 'patient_feed_${PROCESS_DATE}.csv',
                p_batch_id      => v_batch_id,
                p_rows_loaded   => v_rows
            );
            DBMS_OUTPUT.PUT_LINE('Batch ID: ' || v_batch_id);
            DBMS_OUTPUT.PUT_LINE('Rows loaded: ' || v_rows);
        END;
        /

        EXIT SUCCESS;
EOSQL
    )
    local exit_code=$?
    echo "${result}" >> "${LOG_FILE}"

    if [ ${exit_code} -ne 0 ]; then
        log_error "Patient staging failed"
        return 4
    fi

    log_info "Patient staging phase completed"
    return 0
}

phase_stage_trials_ae() {
    log_info "Phase: STAGE TRIALS AND ADVERSE EVENTS"
    log_separator

    check_db_connection || return 3

    # Stage trial data
    run_sqlplus \
        "${SQL_DIR}/etl/staging/stage_trial_data.sql" \
        "Trial data staging" || return 4

    # Stage adverse event data
    run_sqlplus \
        "${SQL_DIR}/etl/staging/stage_ae_data.sql" \
        "Adverse event data staging" || return 4

    log_info "Trial and AE staging phase completed"
    return 0
}

phase_load_vault() {
    log_info "Phase: LOAD RAW VAULT"
    log_separator

    check_db_connection || return 3

    # Load hubs first
    run_sqlplus \
        "${SQL_DIR}/etl/loading/load_hub_drug.sql" \
        "Load HUB_DRUG" || return 4

    # Load links (depends on hubs)
    run_sqlplus \
        "${SQL_DIR}/etl/loading/load_links.sql" \
        "Load link tables" || return 4

    # Load satellites (depends on hubs)
    run_sqlplus \
        "${SQL_DIR}/etl/loading/load_satellites.sql" \
        "Load satellite tables" || return 4

    log_info "Raw vault loading phase completed"
    return 0
}

phase_quality_checks() {
    log_info "Phase: DATA QUALITY CHECKS"
    log_separator

    # Run Python-based quality checks
    if command -v python3 &>/dev/null; then
        python3 -m pharma_vault.quality.data_quality --batch-date "${PROCESS_DATE}" 2>&1 | tee -a "${LOG_FILE}"
        local exit_code=${PIPESTATUS[0]}

        if [ ${exit_code} -ne 0 ]; then
            log_warn "Data quality checks reported issues (exit code ${exit_code})"
            return 5
        fi
    else
        log_warn "Python3 not available; skipping quality checks"
    fi

    log_info "Data quality checks completed"
    return 0
}

phase_business_vault() {
    log_info "Phase: BUSINESS VAULT REFRESH"
    log_separator

    check_db_connection || return 3

    # Refresh PIT tables
    run_sqlplus \
        "${SQL_DIR}/ddl/business_vault/pit_tables.sql" \
        "Refresh PIT tables" || return 4

    # Refresh bridge tables
    run_sqlplus \
        "${SQL_DIR}/ddl/business_vault/bridge_tables.sql" \
        "Refresh bridge tables" || return 4

    log_info "Business vault refresh completed"
    return 0
}

phase_archive() {
    log_info "Phase: ARCHIVE AND CLEANUP"
    log_separator

    local archive_subdir="${ARCHIVE_DIR}/${PROCESS_DATE:0:6}"
    mkdir -p "${archive_subdir}"

    # Archive processed files
    local archived_count=0
    for filepath in "${STAGING_DIR}"/*_feed_*.{csv,dat} "${STAGING_DIR}"/*_feed_*.md5 2>/dev/null; do
        [ -f "${filepath}" ] || continue
        local filename
        filename=$(basename "${filepath}")
        local timestamp
        timestamp=$(date +%H%M%S)
        mv "${filepath}" "${archive_subdir}/${filename%.}_${timestamp}" 2>/dev/null || true
        archived_count=$((archived_count + 1))
    done

    log_info "Archived ${archived_count} files to ${archive_subdir}"

    # Purge old staging data
    local result
    result=$(sqlplus -S "${SQLPLUS_CONN}" <<-EOSQL
        SET SERVEROUTPUT ON
        WHENEVER SQLERROR EXIT SQL.SQLCODE

        DECLARE
            v_deleted NUMBER;
        BEGIN
            DELETE FROM STG_DRUG_RAW
            WHERE LOAD_TIMESTAMP < SYSTIMESTAMP - INTERVAL '${STAGING_RETENTION_DAYS}' DAY;
            v_deleted := SQL%ROWCOUNT;
            DBMS_OUTPUT.PUT_LINE('Purged ' || v_deleted || ' old STG_DRUG_RAW rows');

            DELETE FROM STG_DRUG_CLEAN
            WHERE LOAD_TIMESTAMP < SYSTIMESTAMP - INTERVAL '${STAGING_RETENTION_DAYS}' DAY;
            v_deleted := SQL%ROWCOUNT;
            DBMS_OUTPUT.PUT_LINE('Purged ' || v_deleted || ' old STG_DRUG_CLEAN rows');

            COMMIT;
        END;
        /
        EXIT SUCCESS;
EOSQL
    )
    echo "${result}" >> "${LOG_FILE}"

    # Compress old log files
    find "${LOG_DIR}" -name "daily_etl_*.log" -mtime +7 -exec gzip {} \; 2>/dev/null

    log_info "Archive and cleanup completed"
    return 0
}

# =============================================================================
# Main Entry Point
# =============================================================================

usage() {
    cat <<-EOF
Usage: ${SCRIPT_NAME} --phase <phase_name> [--date YYYYMMDD]

Phases:
  check_files       Verify daily feed files have arrived
  stage_drugs       Stage drug feed data
  stage_patients    Stage patient demographic data
  stage_trials_ae   Stage trial and adverse event data
  load_vault        Load raw vault (hubs, links, satellites)
  quality_checks    Run data quality validation
  business_vault    Refresh PIT tables and bridge tables
  archive           Archive files and clean staging
  pre_mart_validate Pre-mart refresh validation
EOF
    exit 8
}

main() {
    local phase=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --phase)
                phase="$2"
                shift 2
                ;;
            --date)
                PROCESS_DATE="$2"
                shift 2
                ;;
            --help|-h)
                usage
                ;;
            *)
                log_error "Unknown argument: $1"
                usage
                ;;
        esac
    done

    if [ -z "${phase}" ]; then
        log_error "Phase not specified"
        usage
    fi

    # Ensure log directory exists
    mkdir -p "${LOG_DIR}"

    log_info "============================================"
    log_info "PharmaDataVault Daily ETL - Phase: ${phase}"
    log_info "Process Date: ${PROCESS_DATE}"
    log_info "Host: $(hostname)"
    log_info "User: $(whoami)"
    log_info "PID: $$"
    log_info "============================================"

    # Acquire lock for this phase
    acquire_lock "pharma_etl_${phase}" || exit 1

    # Execute the requested phase
    local exit_code=0
    case "${phase}" in
        check_files)        phase_check_files       ;;
        stage_drugs)        phase_stage_drugs        ;;
        stage_patients)     phase_stage_patients     ;;
        stage_trials_ae)    phase_stage_trials_ae    ;;
        load_vault)         phase_load_vault         ;;
        quality_checks)     phase_quality_checks     ;;
        business_vault)     phase_business_vault     ;;
        archive)            phase_archive            ;;
        pre_mart_validate)  phase_quality_checks     ;;
        *)
            log_error "Unknown phase: ${phase}"
            release_lock "pharma_etl_${phase}"
            exit 8
            ;;
    esac
    exit_code=$?

    # Release lock
    release_lock "pharma_etl_${phase}"

    if [ ${exit_code} -eq 0 ]; then
        log_info "Phase '${phase}' completed successfully"
    else
        log_error "Phase '${phase}' failed with exit code ${exit_code}"
    fi

    exit ${exit_code}
}

main "$@"
