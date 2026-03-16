#!/usr/bin/env bash
################################################################################
# RegRecord - Audit Trail Archival Script
#
# Archives old audit trail records, validates integrity, and compresses
# archived data. Designed for monthly Control-M scheduling.
#
# Usage:
#   audit_archive.sh --archive     # Archive old records
#   audit_archive.sh --validate    # Validate archive integrity
#   audit_archive.sh --compress    # Compress old archive files
#   audit_archive.sh --full        # Run all steps
#
# Environment Variables:
#   REGRECORD_HOME           - Application home directory
#   ORACLE_SID               - Oracle database SID
#   ORACLE_HOME              - Oracle home directory
#   ARCHIVE_CUTOFF_MONTHS    - Months to keep in active table (default: 6)
#   PURGE_CUTOFF_YEARS       - Years to keep in archive (default: 7)
#   BATCH_SIZE               - Records per batch for archival (default: 10000)
#   LOG_DIR                  - Log directory
#   ARCHIVE_DIR              - Archive export directory
################################################################################

set -euo pipefail
IFS=$'\n\t'

# --- Configuration -----------------------------------------------------------

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly REGRECORD_HOME="${REGRECORD_HOME:-/opt/regrecord}"
readonly LOG_DIR="${LOG_DIR:-/var/log/regrecord}"
readonly ARCHIVE_DIR="${ARCHIVE_DIR:-/backup/regrecord/audit_archive}"
readonly ARCHIVE_CUTOFF="${ARCHIVE_CUTOFF_MONTHS:-6}"
readonly PURGE_CUTOFF="${PURGE_CUTOFF_YEARS:-7}"
readonly BATCH_SIZE="${BATCH_SIZE:-10000}"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/audit_archive_${TIMESTAMP}.log"
readonly LOCK_FILE="/tmp/regrecord_audit_archive.lock"

# Oracle environment
readonly ORACLE_SID="${ORACLE_SID:-REGDB}"
readonly ORACLE_HOME="${ORACLE_HOME:-/opt/oracle/product/19c}"
readonly DB_USER="${DB_USER:-svc_regrecord}"
readonly DB_CONNECT="${DB_CONNECT:-${DB_USER}@${ORACLE_SID}}"
readonly SQL_DIR="${REGRECORD_HOME}/sql"

# Exit codes
readonly E_SUCCESS=0
readonly E_DB_FAIL=1
readonly E_ARCHIVE_FAIL=2
readonly E_VALIDATE_FAIL=3
readonly E_COMPRESS_FAIL=4
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

# --- Helper: Run SQL ----------------------------------------------------------

run_sql() {
    local sql_text="$1"
    local description="${2:-SQL execution}"

    log_info "Executing SQL: $description"

    # In production, this would use sqlplus. For portability, we use a wrapper.
    local sql_file
    sql_file=$(mktemp /tmp/regrecord_sql_XXXXX.sql)

    cat > "$sql_file" <<SQL_EOF
SET SERVEROUTPUT ON SIZE UNLIMITED
SET FEEDBACK ON
SET TIMING ON
SET LINESIZE 200
SET PAGESIZE 1000

WHENEVER SQLERROR EXIT SQL.SQLCODE ROLLBACK

${sql_text}

EXIT SUCCESS;
SQL_EOF

    # Execute via sqlplus (or simulate for non-Oracle environments)
    if command -v sqlplus &>/dev/null; then
        sqlplus -s "${DB_CONNECT}" @"${sql_file}" 2>&1 | tee -a "$LOG_FILE"
        local rc=${PIPESTATUS[0]}
    else
        log_info "[SIMULATE] Would execute SQL via sqlplus:"
        cat "$sql_file" | tee -a "$LOG_FILE"
        local rc=0
    fi

    rm -f "$sql_file"
    return $rc
}

# --- Archive ------------------------------------------------------------------

run_archive() {
    log_info "=========================================="
    log_info "Starting audit trail archival"
    log_info "Cutoff: $ARCHIVE_CUTOFF months ago"
    log_info "Batch size: $BATCH_SIZE records"
    log_info "=========================================="

    local start_time
    start_time=$(date +%s)

    # Create archive directory
    mkdir -p "$ARCHIVE_DIR/${TIMESTAMP}" 2>/dev/null || {
        log_error "Cannot create archive directory"
        return $E_ARCHIVE_FAIL
    }

    # Step 1: Count records to archive
    local count_sql="
SELECT COUNT(*) AS archive_count
FROM AUDIT_TRAIL
WHERE changed_at < ADD_MONTHS(SYSTIMESTAMP, -${ARCHIVE_CUTOFF});
"
    log_info "Step 1: Counting records to archive..."
    run_sql "$count_sql" "Count records to archive"

    # Step 2: Create archive table if not exists
    local create_archive_sql="
BEGIN
    EXECUTE IMMEDIATE '
        CREATE TABLE AUDIT_TRAIL_ARCHIVE AS
        SELECT * FROM AUDIT_TRAIL WHERE 1=0';
EXCEPTION
    WHEN OTHERS THEN
        IF SQLCODE != -955 THEN
            RAISE;
        END IF;
END;
/
"
    log_info "Step 2: Ensuring archive table exists..."
    run_sql "$create_archive_sql" "Create archive table"

    # Step 3: Archive records in batches
    local archive_sql="
DECLARE
    v_batch_count   NUMBER := 0;
    v_total_count   NUMBER := 0;
    v_cutoff_date   TIMESTAMP := ADD_MONTHS(SYSTIMESTAMP, -${ARCHIVE_CUTOFF});
BEGIN
    LOOP
        -- Insert batch into archive
        INSERT INTO AUDIT_TRAIL_ARCHIVE
        SELECT * FROM AUDIT_TRAIL
        WHERE changed_at < v_cutoff_date
          AND ROWNUM <= ${BATCH_SIZE}
          AND (id, changed_at) NOT IN (
              SELECT id, changed_at FROM AUDIT_TRAIL_ARCHIVE
          );

        v_batch_count := SQL%ROWCOUNT;

        IF v_batch_count = 0 THEN
            EXIT;
        END IF;

        v_total_count := v_total_count + v_batch_count;

        -- Delete archived records from main table
        DELETE FROM AUDIT_TRAIL
        WHERE changed_at < v_cutoff_date
          AND ROWNUM <= ${BATCH_SIZE}
          AND (id, changed_at) IN (
              SELECT id, changed_at FROM AUDIT_TRAIL_ARCHIVE
          );

        COMMIT;

        DBMS_OUTPUT.PUT_LINE('Archived batch: ' || v_batch_count ||
                             ' records (total: ' || v_total_count || ')');

        -- Yield to other processes
        DBMS_LOCK.SLEEP(0.1);
    END LOOP;

    COMMIT;

    DBMS_OUTPUT.PUT_LINE('========================================');
    DBMS_OUTPUT.PUT_LINE('Archival complete: ' || v_total_count || ' records');
    DBMS_OUTPUT.PUT_LINE('========================================');
END;
/
"
    log_info "Step 3: Archiving records in batches of $BATCH_SIZE..."
    run_sql "$archive_sql" "Archive audit records"

    # Step 4: Export archived data to flat file for cold storage
    local export_sql="
SET COLSEP '|'
SET HEADSEP OFF
SET TRIMSPOOL ON

SPOOL ${ARCHIVE_DIR}/${TIMESTAMP}/audit_archive_export.csv

SELECT id || '|' || table_name || '|' || record_id || '|' ||
       action || '|' || changed_by || '|' ||
       TO_CHAR(changed_at, 'YYYY-MM-DD HH24:MI:SS')
FROM AUDIT_TRAIL_ARCHIVE
WHERE changed_at >= ADD_MONTHS(SYSTIMESTAMP, -$(( ARCHIVE_CUTOFF + 1 )))
  AND changed_at < ADD_MONTHS(SYSTIMESTAMP, -${ARCHIVE_CUTOFF});

SPOOL OFF
"
    log_info "Step 4: Exporting archived data to flat file..."
    run_sql "$export_sql" "Export archived data"

    # Step 5: Purge very old records (beyond retention period)
    local purge_sql="
DECLARE
    v_purge_date    TIMESTAMP := ADD_MONTHS(SYSTIMESTAMP, -$(( PURGE_CUTOFF * 12 )));
    v_purged        NUMBER;
BEGIN
    DELETE FROM AUDIT_TRAIL_ARCHIVE
    WHERE changed_at < v_purge_date;

    v_purged := SQL%ROWCOUNT;
    COMMIT;

    DBMS_OUTPUT.PUT_LINE('Purged ' || v_purged || ' records older than ${PURGE_CUTOFF} years');
END;
/
"
    log_info "Step 5: Purging records older than $PURGE_CUTOFF years..."
    run_sql "$purge_sql" "Purge old archive records"

    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))
    log_info "Archival completed in ${duration}s"

    return $E_SUCCESS
}

# --- Validate -----------------------------------------------------------------

run_validate() {
    log_info "=========================================="
    log_info "Starting archive integrity validation"
    log_info "=========================================="

    # Check record count consistency
    local validate_sql="
DECLARE
    v_main_count        NUMBER;
    v_archive_count     NUMBER;
    v_main_min_date     TIMESTAMP;
    v_archive_max_date  TIMESTAMP;
    v_gap_records       NUMBER;
BEGIN
    SELECT COUNT(*), MIN(changed_at)
    INTO v_main_count, v_main_min_date
    FROM AUDIT_TRAIL;

    SELECT COUNT(*), MAX(changed_at)
    INTO v_archive_count, v_archive_max_date
    FROM AUDIT_TRAIL_ARCHIVE;

    DBMS_OUTPUT.PUT_LINE('Active audit trail: ' || v_main_count || ' records');
    DBMS_OUTPUT.PUT_LINE('  Oldest record: ' || TO_CHAR(v_main_min_date, 'YYYY-MM-DD'));
    DBMS_OUTPUT.PUT_LINE('Archive: ' || v_archive_count || ' records');
    DBMS_OUTPUT.PUT_LINE('  Newest archived: ' || TO_CHAR(v_archive_max_date, 'YYYY-MM-DD'));

    -- Check for gaps: archive max date should be close to main min date
    IF v_main_min_date IS NOT NULL AND v_archive_max_date IS NOT NULL THEN
        IF v_main_min_date - v_archive_max_date > INTERVAL '1' DAY THEN
            DBMS_OUTPUT.PUT_LINE('WARNING: Gap detected between archive and main table');
            DBMS_OUTPUT.PUT_LINE('  Archive ends: ' || TO_CHAR(v_archive_max_date, 'YYYY-MM-DD'));
            DBMS_OUTPUT.PUT_LINE('  Main starts:  ' || TO_CHAR(v_main_min_date, 'YYYY-MM-DD'));
        ELSE
            DBMS_OUTPUT.PUT_LINE('PASS: No gaps between archive and main table');
        END IF;
    END IF;

    -- Check for orphaned records
    SELECT COUNT(*) INTO v_gap_records
    FROM AUDIT_TRAIL a
    WHERE NOT EXISTS (
        SELECT 1 FROM AUDIT_TRAIL_ARCHIVE aa
        WHERE aa.id = a.id AND aa.changed_at = a.changed_at
    )
    AND a.changed_at < ADD_MONTHS(SYSTIMESTAMP, -${ARCHIVE_CUTOFF})
    AND ROWNUM <= 100;

    IF v_gap_records > 0 THEN
        DBMS_OUTPUT.PUT_LINE('WARNING: ' || v_gap_records || ' records in main table older than cutoff');
    ELSE
        DBMS_OUTPUT.PUT_LINE('PASS: All old records properly archived');
    END IF;

    -- Check for duplicates
    SELECT COUNT(*) INTO v_gap_records
    FROM (
        SELECT id, changed_at, COUNT(*) cnt
        FROM AUDIT_TRAIL_ARCHIVE
        GROUP BY id, changed_at
        HAVING COUNT(*) > 1
    );

    IF v_gap_records > 0 THEN
        DBMS_OUTPUT.PUT_LINE('WARNING: ' || v_gap_records || ' duplicate records in archive');
    ELSE
        DBMS_OUTPUT.PUT_LINE('PASS: No duplicate records in archive');
    END IF;

    DBMS_OUTPUT.PUT_LINE('========================================');
    DBMS_OUTPUT.PUT_LINE('Validation complete');
END;
/
"
    run_sql "$validate_sql" "Validate archive integrity"

    # Verify exported files
    if [ -d "$ARCHIVE_DIR" ]; then
        local file_count
        file_count=$(find "$ARCHIVE_DIR" -name "*.csv" -mtime -1 | wc -l)
        log_info "Recent export files found: $file_count"

        # Check file sizes
        find "$ARCHIVE_DIR" -name "*.csv" -mtime -1 -exec ls -lh {} \; | tee -a "$LOG_FILE"
    fi

    log_info "Archive validation completed"
    return $E_SUCCESS
}

# --- Compress -----------------------------------------------------------------

run_compress() {
    log_info "=========================================="
    log_info "Starting archive compression"
    log_info "=========================================="

    local compress_dir="${ARCHIVE_DIR}"
    local compressed_count=0
    local total_saved=0

    if [ ! -d "$compress_dir" ]; then
        log_warn "Archive directory does not exist: $compress_dir"
        return $E_SUCCESS
    fi

    # Find and compress uncompressed CSV files older than specified period
    while IFS= read -r -d '' file; do
        local original_size
        original_size=$(stat -c%s "$file" 2>/dev/null || stat -f%z "$file" 2>/dev/null || echo "0")

        log_info "Compressing: $(basename "$file") ($(( original_size / 1024 )) KB)"

        gzip -9 "$file" 2>/dev/null && {
            local compressed_size
            compressed_size=$(stat -c%s "${file}.gz" 2>/dev/null || stat -f%z "${file}.gz" 2>/dev/null || echo "0")
            local saved=$(( original_size - compressed_size ))
            total_saved=$(( total_saved + saved ))
            compressed_count=$(( compressed_count + 1 ))
            log_info "  Compressed: $(( compressed_size / 1024 )) KB (saved $(( saved / 1024 )) KB)"
        } || {
            log_warn "Failed to compress: $file"
        }
    done < <(find "$compress_dir" -name "*.csv" -mtime +30 -print0 2>/dev/null)

    log_info "Compression complete: $compressed_count files, $(( total_saved / 1024 / 1024 )) MB saved"

    # Generate checksum for compressed files
    local checksum_file="${compress_dir}/checksums_${TIMESTAMP}.sha256"
    find "$compress_dir" -name "*.gz" -newer "$LOG_FILE" -exec sha256sum {} \; > "$checksum_file" 2>/dev/null || true
    log_info "Checksums saved to: $checksum_file"

    return $E_SUCCESS
}

# --- Main ---------------------------------------------------------------------

main() {
    local mode="${1:---full}"

    mkdir -p "$LOG_DIR" 2>/dev/null || true

    log_info "=========================================="
    log_info "RegRecord Audit Archive - $SCRIPT_NAME"
    log_info "Mode: $mode"
    log_info "PID: $$ | Host: $(hostname)"
    log_info "Archive cutoff: $ARCHIVE_CUTOFF months"
    log_info "Purge cutoff: $PURGE_CUTOFF years"
    log_info "=========================================="

    acquire_lock

    case "$mode" in
        --archive)
            run_archive
            ;;
        --validate)
            run_validate
            ;;
        --compress)
            run_compress
            ;;
        --full)
            run_archive || exit $?
            run_validate || exit $?
            run_compress || exit $?
            ;;
        *)
            echo "Usage: $SCRIPT_NAME {--archive|--validate|--compress|--full}" >&2
            exit 1
            ;;
    esac

    local rc=$?
    log_info "Audit archive finished with exit code: $rc"
    exit $rc
}

main "$@"
