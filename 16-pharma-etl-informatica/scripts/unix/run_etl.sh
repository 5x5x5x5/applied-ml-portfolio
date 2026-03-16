#!/usr/bin/env bash
# ===========================================================================
# PharmaFlow ETL Runner Script
#
# UNIX shell wrapper for PharmaFlow Python ETL pipelines.
# Called by Control-M job scheduler or can be run manually.
#
# Usage:
#   run_etl.sh <command> <pipeline> [options]
#
# Commands:
#   validate  - Validate source files (checksum, row count, structure)
#   execute   - Execute an ETL pipeline
#   report    - Generate and email a load report
#   archive   - Archive source files and cleanup staging
#
# Examples:
#   run_etl.sh validate /data/incoming/drug_supplier_*.csv
#   run_etl.sh execute drug_master --batch-id 12345 --date 2026-03-05
#   run_etl.sh execute faers_transform --quarter 2025Q4 --batch-id 12345
#   run_etl.sh report drug_master --batch-id 12345 --email team@co.com
#   run_etl.sh archive drug_master --date 2026-03-05
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Environment Configuration
# ---------------------------------------------------------------------------
export PHARMAFLOW_HOME="${PHARMAFLOW_HOME:-/opt/pharmaflow}"
export PHARMAFLOW_LOG_DIR="${PHARMAFLOW_HOME}/logs"
export PHARMAFLOW_DATA_DIR="${PHARMAFLOW_HOME}/data"
export PHARMAFLOW_REJECT_DIR="${PHARMAFLOW_HOME}/reject"
export PHARMAFLOW_ARCHIVE_DIR="${PHARMAFLOW_HOME}/archive"
export PHARMAFLOW_CONFIG="${PHARMAFLOW_HOME}/config/pharmaflow.env"

# Python environment
export PYTHON_BIN="${PHARMAFLOW_HOME}/.venv/bin/python"
export UV_BIN="${PHARMAFLOW_HOME}/.venv/bin/uv"

# Database connectivity (sourced from config or environment)
export ORACLE_HOME="${ORACLE_HOME:-/u01/app/oracle/product/19c}"
export TNS_ADMIN="${TNS_ADMIN:-${ORACLE_HOME}/network/admin}"
export NLS_LANG="${NLS_LANG:-AMERICAN_AMERICA.AL32UTF8}"

# Logging
SCRIPT_NAME="$(basename "$0")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${PHARMAFLOW_LOG_DIR}/${SCRIPT_NAME%.sh}_${TIMESTAMP}.log"
PID_FILE="/tmp/pharmaflow_${$}.pid"

# Exit codes
readonly EXIT_SUCCESS=0
readonly EXIT_VALIDATION_FAIL=1
readonly EXIT_ETL_FAIL=2
readonly EXIT_ARCHIVE_FAIL=3
readonly EXIT_CONFIG_ERROR=4
readonly EXIT_LOCK_FAIL=5

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

log_msg() {
    local level="$1"
    shift
    local msg="$*"
    local ts
    ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${ts}] [${level}] [PID:$$] ${msg}" | tee -a "${LOG_FILE}"
}

log_info()  { log_msg "INFO"  "$@"; }
log_warn()  { log_msg "WARN"  "$@"; }
log_error() { log_msg "ERROR" "$@"; }
log_fatal() { log_msg "FATAL" "$@"; }

# Rotate logs older than 30 days
rotate_logs() {
    log_info "Rotating logs older than 30 days in ${PHARMAFLOW_LOG_DIR}"
    find "${PHARMAFLOW_LOG_DIR}" -name "*.log" -mtime +30 -exec gzip {} \; 2>/dev/null || true
    find "${PHARMAFLOW_LOG_DIR}" -name "*.log.gz" -mtime +90 -delete 2>/dev/null || true
}

# Setup directories
setup_directories() {
    local dirs=("${PHARMAFLOW_LOG_DIR}" "${PHARMAFLOW_REJECT_DIR}" "${PHARMAFLOW_ARCHIVE_DIR}")
    for dir in "${dirs[@]}"; do
        if [[ ! -d "${dir}" ]]; then
            mkdir -p "${dir}"
            log_info "Created directory: ${dir}"
        fi
    done
}

# Load environment config file
load_config() {
    if [[ -f "${PHARMAFLOW_CONFIG}" ]]; then
        # shellcheck source=/dev/null
        source "${PHARMAFLOW_CONFIG}"
        log_info "Loaded configuration from ${PHARMAFLOW_CONFIG}"
    else
        log_warn "Config file not found: ${PHARMAFLOW_CONFIG} (using defaults)"
    fi
}

# Cleanup on exit
cleanup() {
    local exit_code=$?
    rm -f "${PID_FILE}" 2>/dev/null || true
    if [[ ${exit_code} -ne 0 ]]; then
        log_error "Script exited with code ${exit_code}"
    fi
    log_info "=== PharmaFlow ETL Runner finished (exit=${exit_code}) ==="
}
trap cleanup EXIT

# Send email notification
send_notification() {
    local subject="$1"
    local body="$2"
    local recipients="${3:-${PHARMAFLOW_NOTIFY_EMAIL:-etl-team@pharma.com}}"

    if command -v mail &>/dev/null; then
        echo "${body}" | mail -s "${subject}" "${recipients}" 2>/dev/null || true
    elif command -v sendmail &>/dev/null; then
        {
            echo "Subject: ${subject}"
            echo "To: ${recipients}"
            echo ""
            echo "${body}"
        } | sendmail "${recipients}" 2>/dev/null || true
    else
        log_warn "No mail command available for notification"
    fi
}

# ---------------------------------------------------------------------------
# File Validation Functions
# ---------------------------------------------------------------------------

# Validate file exists and is non-empty
validate_file_exists() {
    local filepath="$1"
    if [[ ! -f "${filepath}" ]]; then
        log_error "File not found: ${filepath}"
        return 1
    fi
    if [[ ! -s "${filepath}" ]]; then
        log_error "File is empty: ${filepath}"
        return 1
    fi
    log_info "File exists and is non-empty: ${filepath} ($(wc -c < "${filepath}") bytes)"
    return 0
}

# Validate file checksum against expected
validate_checksum() {
    local filepath="$1"
    local expected_checksum="${2:-}"

    if [[ -z "${expected_checksum}" ]]; then
        # If no expected checksum, just compute and log it
        local computed
        computed="$(md5sum "${filepath}" | awk '{print $1}')"
        log_info "File checksum: ${filepath} -> MD5: ${computed}"
        return 0
    fi

    local computed
    computed="$(md5sum "${filepath}" | awk '{print $1}')"
    if [[ "${computed}" == "${expected_checksum}" ]]; then
        log_info "Checksum verified: ${filepath}"
        return 0
    else
        log_error "Checksum mismatch for ${filepath}: expected=${expected_checksum} got=${computed}"
        return 1
    fi
}

# Count rows in a file (excluding header)
count_file_rows() {
    local filepath="$1"
    local has_header="${2:-true}"
    local count

    count="$(wc -l < "${filepath}")"
    if [[ "${has_header}" == "true" ]]; then
        count=$((count - 1))
    fi

    log_info "Row count for ${filepath}: ${count} data rows"
    echo "${count}"
}

# Validate all source files for a pipeline
do_validate() {
    local file_pattern="$1"
    shift
    local delimiter=","
    local error_count=0

    # Parse additional options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --delimiter) delimiter="$2"; shift 2 ;;
            *) shift ;;
        esac
    done

    log_info "=== Validating source files: ${file_pattern} ==="

    # Expand glob
    local files
    files=( ${file_pattern} )

    if [[ ${#files[@]} -eq 0 ]]; then
        log_error "No files matching pattern: ${file_pattern}"
        return ${EXIT_VALIDATION_FAIL}
    fi

    for filepath in "${files[@]}"; do
        log_info "--- Validating: ${filepath} ---"

        # Check existence and size
        if ! validate_file_exists "${filepath}"; then
            error_count=$((error_count + 1))
            continue
        fi

        # Compute checksum
        validate_checksum "${filepath}"

        # Count rows
        local rows
        rows="$(count_file_rows "${filepath}" "true")"
        if [[ "${rows}" -eq 0 ]]; then
            log_warn "File has no data rows: ${filepath}"
            error_count=$((error_count + 1))
        fi

        # Validate delimiter consistency (check first 10 lines)
        local expected_cols
        expected_cols="$(head -1 "${filepath}" | awk -F"${delimiter}" '{print NF}')"
        local inconsistent
        inconsistent="$(head -10 "${filepath}" | awk -F"${delimiter}" -v exp="${expected_cols}" 'NF != exp {count++} END {print count+0}')"
        if [[ "${inconsistent}" -gt 0 ]]; then
            log_warn "Delimiter inconsistency in ${filepath}: ${inconsistent} lines differ from header (${expected_cols} cols)"
        fi

        log_info "Validated: ${filepath} (${rows} rows, ${expected_cols} columns)"
    done

    if [[ ${error_count} -gt 0 ]]; then
        log_error "Validation completed with ${error_count} error(s)"
        return ${EXIT_VALIDATION_FAIL}
    fi

    log_info "=== All files validated successfully ==="
    return ${EXIT_SUCCESS}
}

# ---------------------------------------------------------------------------
# ETL Execution
# ---------------------------------------------------------------------------

do_execute() {
    local pipeline="$1"
    shift
    local batch_id=""
    local process_date=""
    local quarter=""

    # Parse options
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --batch-id) batch_id="$2"; shift 2 ;;
            --date)     process_date="$2"; shift 2 ;;
            --quarter)  quarter="$2"; shift 2 ;;
            *)          shift ;;
        esac
    done

    batch_id="${batch_id:-$(date +%Y%m%d%H%M%S)}"
    process_date="${process_date:-$(date +%Y-%m-%d)}"

    log_info "=== Executing ETL Pipeline ==="
    log_info "Pipeline:     ${pipeline}"
    log_info "Batch ID:     ${batch_id}"
    log_info "Process Date: ${process_date}"
    log_info "Quarter:      ${quarter:-N/A}"

    # Write PID file for monitoring
    echo "$$" > "${PID_FILE}"

    # Build Python command based on pipeline
    local python_cmd
    case "${pipeline}" in
        drug_master)
            python_cmd="${PYTHON_BIN} -m pharma_flow.pipelines.drug_master_etl"
            python_cmd+=" --batch-id ${batch_id}"
            python_cmd+=" --date ${process_date}"
            ;;
        clinical_trial)
            python_cmd="${PYTHON_BIN} -m pharma_flow.pipelines.clinical_trial_etl"
            python_cmd+=" --batch-id ${batch_id}"
            python_cmd+=" --date ${process_date}"
            ;;
        faers_*)
            python_cmd="${PYTHON_BIN} -m pharma_flow.pipelines.adverse_event_etl"
            python_cmd+=" --batch-id ${batch_id}"
            python_cmd+=" --quarter ${quarter}"
            python_cmd+=" --stage ${pipeline#faers_}"
            ;;
        *)
            log_fatal "Unknown pipeline: ${pipeline}"
            return ${EXIT_ETL_FAIL}
            ;;
    esac

    log_info "Executing: ${python_cmd}"

    # Execute with timing
    local start_epoch
    start_epoch="$(date +%s)"

    # Run Python ETL and capture output
    local etl_log="${PHARMAFLOW_LOG_DIR}/${pipeline}_${batch_id}.log"
    if eval "${python_cmd}" >> "${etl_log}" 2>&1; then
        local end_epoch
        end_epoch="$(date +%s)"
        local elapsed=$((end_epoch - start_epoch))
        log_info "ETL pipeline '${pipeline}' completed successfully in ${elapsed}s"

        # Record success metrics
        log_info "ETL_METRIC pipeline=${pipeline} batch=${batch_id} status=SUCCESS elapsed=${elapsed}s"
        return ${EXIT_SUCCESS}
    else
        local exit_code=$?
        local end_epoch
        end_epoch="$(date +%s)"
        local elapsed=$((end_epoch - start_epoch))

        log_error "ETL pipeline '${pipeline}' FAILED (exit=${exit_code}) after ${elapsed}s"
        log_error "Check detailed log: ${etl_log}"

        # Check for bad files
        local bad_files
        bad_files="$(find "${PHARMAFLOW_REJECT_DIR}" -name "*${batch_id}*" -newer "${etl_log}" 2>/dev/null)"
        if [[ -n "${bad_files}" ]]; then
            log_error "Reject files found:"
            echo "${bad_files}" | while read -r f; do
                log_error "  ${f} ($(wc -l < "${f}") rows)"
            done
        fi

        # Send failure notification
        send_notification \
            "[FAILED] PharmaFlow ${pipeline} - Batch ${batch_id}" \
            "Pipeline: ${pipeline}\nBatch: ${batch_id}\nDate: ${process_date}\nDuration: ${elapsed}s\nExit code: ${exit_code}\nLog: ${etl_log}"

        return ${EXIT_ETL_FAIL}
    fi
}

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

do_report() {
    local pipeline="$1"
    shift
    local batch_id=""
    local email_to=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --batch-id) batch_id="$2"; shift 2 ;;
            --date)     shift 2 ;;
            --email)    email_to="$2"; shift 2 ;;
            *)          shift ;;
        esac
    done

    log_info "=== Generating Report for ${pipeline} (batch: ${batch_id}) ==="

    local report_file="${PHARMAFLOW_LOG_DIR}/report_${pipeline}_${batch_id}.txt"

    {
        echo "============================================="
        echo "PharmaFlow ETL Load Report"
        echo "============================================="
        echo "Pipeline:  ${pipeline}"
        echo "Batch ID:  ${batch_id}"
        echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "============================================="
        echo ""

        # Extract metrics from log files
        local etl_log="${PHARMAFLOW_LOG_DIR}/${pipeline}_${batch_id}.log"
        if [[ -f "${etl_log}" ]]; then
            echo "--- Session Statistics ---"
            grep -E "source_rows|target_rows|throughput|elapsed" "${etl_log}" 2>/dev/null || echo "  No metrics found in log"
            echo ""

            echo "--- Errors ---"
            local error_count
            error_count="$(grep -c "ERROR" "${etl_log}" 2>/dev/null || echo 0)"
            echo "  Total errors: ${error_count}"
            if [[ "${error_count}" -gt 0 ]]; then
                grep "ERROR" "${etl_log}" | tail -10
            fi
        else
            echo "  Log file not found: ${etl_log}"
        fi

        echo ""
        echo "--- Reject Files ---"
        find "${PHARMAFLOW_REJECT_DIR}" -name "*${batch_id}*" 2>/dev/null | while read -r f; do
            echo "  ${f}: $(wc -l < "${f}") rows"
        done

        echo ""
        echo "============================================="
        echo "End of Report"
        echo "============================================="
    } > "${report_file}"

    log_info "Report written to: ${report_file}"

    if [[ -n "${email_to}" ]]; then
        send_notification \
            "PharmaFlow Report: ${pipeline} (${batch_id})" \
            "$(cat "${report_file}")" \
            "${email_to}"
        log_info "Report emailed to: ${email_to}"
    fi

    return ${EXIT_SUCCESS}
}

# ---------------------------------------------------------------------------
# Archive and Cleanup
# ---------------------------------------------------------------------------

do_archive() {
    local pipeline="$1"
    shift
    local process_date=""
    local quarter=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --date)    process_date="$2"; shift 2 ;;
            --quarter) quarter="$2"; shift 2 ;;
            *)         shift ;;
        esac
    done

    process_date="${process_date:-$(date +%Y-%m-%d)}"
    local archive_subdir="${PHARMAFLOW_ARCHIVE_DIR}/${pipeline}/${process_date}"

    log_info "=== Archiving files for ${pipeline} (date: ${process_date}) ==="

    mkdir -p "${archive_subdir}"

    # Archive source files based on pipeline
    case "${pipeline}" in
        drug_master)
            local source_dir="/data/incoming"
            if compgen -G "${source_dir}/drug_supplier_*.csv" > /dev/null 2>&1; then
                for f in "${source_dir}"/drug_supplier_*.csv; do
                    gzip -c "${f}" > "${archive_subdir}/$(basename "${f}").gz"
                    log_info "Archived: ${f} -> ${archive_subdir}/"
                    rm -f "${f}"
                done
            fi
            ;;
        faers)
            local faers_dir="/data/incoming/faers/${quarter}"
            if [[ -d "${faers_dir}" ]]; then
                tar -czf "${archive_subdir}/faers_${quarter}.tar.gz" -C "${faers_dir}" . 2>/dev/null || true
                log_info "Archived FAERS ${quarter} to ${archive_subdir}/"
            fi
            ;;
    esac

    # Cleanup old archives (older than 365 days)
    find "${PHARMAFLOW_ARCHIVE_DIR}" -name "*.gz" -mtime +365 -delete 2>/dev/null || true
    find "${PHARMAFLOW_ARCHIVE_DIR}" -name "*.tar.gz" -mtime +365 -delete 2>/dev/null || true

    # Cleanup old reject files (older than 90 days)
    find "${PHARMAFLOW_REJECT_DIR}" -name "*.csv" -mtime +90 -delete 2>/dev/null || true

    log_info "=== Archive complete ==="
    return ${EXIT_SUCCESS}
}

# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

main() {
    if [[ $# -lt 1 ]]; then
        echo "Usage: ${SCRIPT_NAME} <command> [args...]"
        echo "Commands: validate, execute, report, archive"
        exit ${EXIT_CONFIG_ERROR}
    fi

    local command="$1"
    shift

    # Initialize
    setup_directories
    load_config
    rotate_logs

    log_info "=== PharmaFlow ETL Runner started ==="
    log_info "Command: ${command}"
    log_info "Args:    $*"
    log_info "Host:    $(hostname)"
    log_info "User:    $(whoami)"
    log_info "PID:     $$"

    case "${command}" in
        validate)
            do_validate "$@"
            ;;
        execute)
            do_execute "$@"
            ;;
        report)
            do_report "$@"
            ;;
        archive)
            do_archive "$@"
            ;;
        *)
            log_fatal "Unknown command: ${command}"
            exit ${EXIT_CONFIG_ERROR}
            ;;
    esac
}

main "$@"
