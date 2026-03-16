#!/usr/bin/env bash
################################################################################
# RegRecord - Database Backup Script with Encryption
#
# Performs full or incremental backups of the RegRecord database with
# AES-256 encryption, checksums, retention management, and remote copy.
#
# Usage:
#   backup_records.sh --full          # Full database backup
#   backup_records.sh --incremental   # Incremental backup
#   backup_records.sh --audit-only    # Backup audit trail only
#   backup_records.sh --verify        # Verify last backup
#   backup_records.sh --cleanup       # Remove old backups
#
# Environment Variables:
#   REGRECORD_HOME       - Application home directory
#   ORACLE_SID           - Oracle database SID
#   ORACLE_HOME          - Oracle home directory
#   BACKUP_DIR           - Local backup directory
#   REMOTE_BACKUP_DIR    - Remote backup destination (scp)
#   ENCRYPTION_KEY_FILE  - Path to AES-256 encryption key
#   RETENTION_DAYS       - Days to keep local backups (default: 30)
#   REMOTE_RETENTION_DAYS - Days to keep remote backups (default: 90)
#   LOG_DIR              - Log directory
################################################################################

set -euo pipefail
IFS=$'\n\t'

# --- Configuration -----------------------------------------------------------

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly REGRECORD_HOME="${REGRECORD_HOME:-/opt/regrecord}"
readonly LOG_DIR="${LOG_DIR:-/var/log/regrecord}"
readonly BACKUP_DIR="${BACKUP_DIR:-/backup/regrecord}"
readonly REMOTE_BACKUP_HOST="${REMOTE_BACKUP_HOST:-backup-server}"
readonly REMOTE_BACKUP_DIR="${REMOTE_BACKUP_DIR:-/remote_backup/regrecord}"
readonly REMOTE_BACKUP_USER="${REMOTE_BACKUP_USER:-svc_backup}"
readonly ENCRYPTION_KEY_FILE="${ENCRYPTION_KEY_FILE:-/opt/regrecord/.backup_key}"
readonly RETENTION_DAYS="${RETENTION_DAYS:-30}"
readonly REMOTE_RETENTION_DAYS="${REMOTE_RETENTION_DAYS:-90}"
readonly TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
readonly LOG_FILE="${LOG_DIR}/backup_${TIMESTAMP}.log"
readonly LOCK_FILE="/tmp/regrecord_backup.lock"

# Oracle environment
readonly ORACLE_SID="${ORACLE_SID:-REGDB}"
readonly ORACLE_HOME="${ORACLE_HOME:-/opt/oracle/product/19c}"
readonly DB_USER="${DB_USER:-svc_regrecord}"

# Backup tables
readonly TABLES=(
    "DRUG"
    "AGENCY"
    "REGULATORY_SUBMISSION"
    "SUBMISSION_DOCUMENT"
    "APPROVAL_RECORD"
    "LABELING_CHANGE"
    "COMPLIANCE_RECORD"
    "AUDIT_TRAIL"
    "PSEUDO_RECORD"
    "REIDENTIFICATION_LOG"
)

# Exit codes
readonly E_SUCCESS=0
readonly E_CONFIG_FAIL=1
readonly E_BACKUP_FAIL=2
readonly E_ENCRYPT_FAIL=3
readonly E_VERIFY_FAIL=4
readonly E_REMOTE_FAIL=5
readonly E_LOCK_FAIL=6

# --- Logging ------------------------------------------------------------------

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] INFO  ${SCRIPT_NAME}: $1" | tee -a "$LOG_FILE"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] WARN  ${SCRIPT_NAME}: $1" | tee -a "$LOG_FILE" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR ${SCRIPT_NAME}: $1" | tee -a "$LOG_FILE" >&2
}

# --- Lock Management ----------------------------------------------------------

acquire_lock() {
    if [ -f "$LOCK_FILE" ]; then
        local pid
        pid=$(cat "$LOCK_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Another backup is running (PID: $pid). Exiting."
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

# --- Encryption ---------------------------------------------------------------

generate_encryption_key() {
    if [ ! -f "$ENCRYPTION_KEY_FILE" ]; then
        log_info "Generating new AES-256 encryption key..."
        openssl rand -base64 32 > "$ENCRYPTION_KEY_FILE"
        chmod 600 "$ENCRYPTION_KEY_FILE"
        log_info "Encryption key generated at: $ENCRYPTION_KEY_FILE"
        log_warn "IMPORTANT: Store a copy of the encryption key in a secure vault!"
    fi
}

encrypt_file() {
    local input_file="$1"
    local output_file="${input_file}.enc"

    if [ ! -f "$ENCRYPTION_KEY_FILE" ]; then
        log_error "Encryption key file not found: $ENCRYPTION_KEY_FILE"
        return $E_ENCRYPT_FAIL
    fi

    log_info "Encrypting: $(basename "$input_file")"

    openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \
        -in "$input_file" \
        -out "$output_file" \
        -pass "file:${ENCRYPTION_KEY_FILE}" \
        2>>"$LOG_FILE"

    if [ $? -eq 0 ]; then
        local orig_size
        orig_size=$(stat -c%s "$input_file" 2>/dev/null || stat -f%z "$input_file" 2>/dev/null || echo "0")
        local enc_size
        enc_size=$(stat -c%s "$output_file" 2>/dev/null || stat -f%z "$output_file" 2>/dev/null || echo "0")
        log_info "  Encrypted: $(( enc_size / 1024 )) KB (original: $(( orig_size / 1024 )) KB)"

        # Remove unencrypted file
        rm -f "$input_file"
        return $E_SUCCESS
    else
        log_error "Encryption failed for: $input_file"
        return $E_ENCRYPT_FAIL
    fi
}

decrypt_file() {
    local input_file="$1"
    local output_file="${input_file%.enc}"

    openssl enc -aes-256-cbc -d -salt -pbkdf2 -iter 100000 \
        -in "$input_file" \
        -out "$output_file" \
        -pass "file:${ENCRYPTION_KEY_FILE}" \
        2>>"$LOG_FILE"
}

# --- Backup Functions ---------------------------------------------------------

run_full_backup() {
    log_info "=========================================="
    log_info "Starting FULL database backup"
    log_info "=========================================="

    local start_time
    start_time=$(date +%s)
    local backup_subdir="${BACKUP_DIR}/full_${TIMESTAMP}"
    mkdir -p "$backup_subdir"

    # Pre-backup checks
    generate_encryption_key

    local total_size=0
    local table_count=0
    local failed_tables=0

    for table in "${TABLES[@]}"; do
        log_info "Backing up table: $table"

        local dump_file="${backup_subdir}/${table}_${TIMESTAMP}.dmp"
        local log_exp="${backup_subdir}/${table}_${TIMESTAMP}_exp.log"

        # Export using expdp (Data Pump) or simulate
        if command -v expdp &>/dev/null; then
            expdp "${DB_USER}@${ORACLE_SID}" \
                tables="$table" \
                directory=BACKUP_DIR \
                dumpfile="${table}_${TIMESTAMP}.dmp" \
                logfile="${table}_${TIMESTAMP}_exp.log" \
                compression=ALL \
                2>>"$LOG_FILE" || {
                log_error "Failed to export table: $table"
                failed_tables=$(( failed_tables + 1 ))
                continue
            }
        else
            # Simulate export for non-Oracle environments
            log_info "  [SIMULATE] expdp ${DB_USER}@${ORACLE_SID} tables=$table"
            # Create a placeholder file
            echo "-- Data dump for $table at $TIMESTAMP" > "$dump_file"
            echo "-- This is a simulated backup for demonstration" >> "$dump_file"
        fi

        # Encrypt the dump file
        if [ -f "$dump_file" ]; then
            encrypt_file "$dump_file" || {
                log_error "Failed to encrypt: $dump_file"
                failed_tables=$(( failed_tables + 1 ))
                continue
            }

            local file_size
            file_size=$(stat -c%s "${dump_file}.enc" 2>/dev/null || stat -f%z "${dump_file}.enc" 2>/dev/null || echo "0")
            total_size=$(( total_size + file_size ))
            table_count=$(( table_count + 1 ))
        fi
    done

    # Generate manifest file
    local manifest="${backup_subdir}/manifest.json"
    cat > "$manifest" <<MANIFEST_EOF
{
    "backup_type": "FULL",
    "timestamp": "$TIMESTAMP",
    "database": "$ORACLE_SID",
    "host": "$(hostname)",
    "tables_backed_up": $table_count,
    "tables_failed": $failed_tables,
    "total_size_bytes": $total_size,
    "encryption": "AES-256-CBC",
    "retention_days": $RETENTION_DAYS,
    "backup_dir": "$backup_subdir",
    "checksums": {
$(cd "$backup_subdir" && \
  find . -name "*.enc" -exec sha256sum {} \; 2>/dev/null | \
  awk '{printf "        \"%s\": \"%s\",\n", $2, $1}' | \
  sed '$ s/,$//')
    }
}
MANIFEST_EOF

    log_info "Manifest generated: $manifest"

    # Generate overall checksum
    local checksum_file="${backup_subdir}/SHA256SUMS"
    (cd "$backup_subdir" && sha256sum *.enc manifest.json > "$checksum_file" 2>/dev/null) || true

    # Copy to remote if configured
    if [ -n "${REMOTE_BACKUP_HOST:-}" ]; then
        copy_to_remote "$backup_subdir"
    fi

    local end_time
    end_time=$(date +%s)
    local duration=$(( end_time - start_time ))

    log_info "=========================================="
    log_info "Full backup completed"
    log_info "  Tables: $table_count/$((${#TABLES[@]}))"
    log_info "  Failed: $failed_tables"
    log_info "  Size: $(( total_size / 1024 / 1024 )) MB"
    log_info "  Duration: ${duration}s"
    log_info "  Location: $backup_subdir"
    log_info "=========================================="

    [ $failed_tables -eq 0 ] && return $E_SUCCESS || return $E_BACKUP_FAIL
}

run_audit_only_backup() {
    log_info "=========================================="
    log_info "Starting AUDIT-ONLY backup"
    log_info "=========================================="

    local backup_subdir="${BACKUP_DIR}/audit_${TIMESTAMP}"
    mkdir -p "$backup_subdir"

    generate_encryption_key

    for table in "AUDIT_TRAIL" "REIDENTIFICATION_LOG" "PSEUDO_RECORD"; do
        log_info "Backing up table: $table"

        local dump_file="${backup_subdir}/${table}_${TIMESTAMP}.dmp"

        if command -v expdp &>/dev/null; then
            expdp "${DB_USER}@${ORACLE_SID}" \
                tables="$table" \
                directory=BACKUP_DIR \
                dumpfile="${table}_${TIMESTAMP}.dmp" \
                logfile="${table}_${TIMESTAMP}_exp.log" \
                compression=ALL \
                2>>"$LOG_FILE" || {
                log_error "Failed to export: $table"
                continue
            }
        else
            echo "-- Audit data dump for $table at $TIMESTAMP" > "$dump_file"
        fi

        if [ -f "$dump_file" ]; then
            encrypt_file "$dump_file"
        fi
    done

    # Generate checksum
    (cd "$backup_subdir" && sha256sum *.enc > SHA256SUMS 2>/dev/null) || true

    log_info "Audit-only backup completed: $backup_subdir"
    return $E_SUCCESS
}

# --- Verify -------------------------------------------------------------------

run_verify() {
    log_info "=========================================="
    log_info "Verifying last backup"
    log_info "=========================================="

    # Find the most recent backup directory
    local latest_backup
    latest_backup=$(find "$BACKUP_DIR" -maxdepth 1 -type d -name "full_*" -o -name "audit_*" | \
        sort -r | head -1)

    if [ -z "$latest_backup" ]; then
        log_error "No backup found in $BACKUP_DIR"
        return $E_VERIFY_FAIL
    fi

    log_info "Verifying backup: $latest_backup"

    # Verify checksums
    local checksum_file="${latest_backup}/SHA256SUMS"
    if [ -f "$checksum_file" ]; then
        log_info "Verifying checksums..."
        (cd "$latest_backup" && sha256sum -c "$checksum_file" 2>&1) | tee -a "$LOG_FILE"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log_error "Checksum verification FAILED!"
            return $E_VERIFY_FAIL
        fi
        log_info "Checksum verification: PASSED"
    else
        log_warn "No checksum file found; skipping verification"
    fi

    # Verify encrypted files can be decrypted
    local enc_files
    enc_files=$(find "$latest_backup" -name "*.enc" -type f)
    local verified=0
    local failed=0

    for enc_file in $enc_files; do
        log_info "Verifying encryption: $(basename "$enc_file")"
        if decrypt_file "$enc_file" 2>/dev/null; then
            local decrypted="${enc_file%.enc}"
            if [ -f "$decrypted" ] && [ -s "$decrypted" ]; then
                verified=$(( verified + 1 ))
                rm -f "$decrypted"  # Remove decrypted test file
            else
                failed=$(( failed + 1 ))
                log_error "  Decryption produced empty file"
            fi
        else
            failed=$(( failed + 1 ))
            log_error "  Decryption FAILED"
        fi
    done

    # Check manifest
    local manifest="${latest_backup}/manifest.json"
    if [ -f "$manifest" ]; then
        log_info "Manifest contents:"
        cat "$manifest" | tee -a "$LOG_FILE"
    fi

    log_info "=========================================="
    log_info "Verification complete"
    log_info "  Verified: $verified files"
    log_info "  Failed: $failed files"
    log_info "=========================================="

    [ $failed -eq 0 ] && return $E_SUCCESS || return $E_VERIFY_FAIL
}

# --- Cleanup ------------------------------------------------------------------

run_cleanup() {
    log_info "=========================================="
    log_info "Cleaning up old backups"
    log_info "Local retention: $RETENTION_DAYS days"
    log_info "=========================================="

    local removed=0
    local freed=0

    # Remove local backups older than retention period
    while IFS= read -r -d '' dir; do
        local dir_size
        dir_size=$(du -sk "$dir" 2>/dev/null | awk '{print $1}')
        log_info "Removing old backup: $(basename "$dir") (${dir_size} KB)"
        rm -rf "$dir"
        removed=$(( removed + 1 ))
        freed=$(( freed + dir_size ))
    done < <(find "$BACKUP_DIR" -maxdepth 1 -type d \
        \( -name "full_*" -o -name "audit_*" \) \
        -mtime "+${RETENTION_DAYS}" -print0 2>/dev/null)

    log_info "Cleanup complete: $removed directories removed, $(( freed / 1024 )) MB freed"
    return $E_SUCCESS
}

# --- Remote Copy --------------------------------------------------------------

copy_to_remote() {
    local source_dir="$1"

    log_info "Copying backup to remote: ${REMOTE_BACKUP_USER}@${REMOTE_BACKUP_HOST}:${REMOTE_BACKUP_DIR}"

    if command -v scp &>/dev/null; then
        scp -r -o ConnectTimeout=30 -o StrictHostKeyChecking=no \
            "$source_dir" \
            "${REMOTE_BACKUP_USER}@${REMOTE_BACKUP_HOST}:${REMOTE_BACKUP_DIR}/" \
            2>>"$LOG_FILE" && {
            log_info "Remote copy successful"
        } || {
            log_warn "Remote copy failed (non-fatal)"
        }
    else
        log_info "[SIMULATE] scp -r $source_dir ${REMOTE_BACKUP_USER}@${REMOTE_BACKUP_HOST}:${REMOTE_BACKUP_DIR}/"
    fi
}

# --- Main ---------------------------------------------------------------------

main() {
    local mode="${1:---full}"

    mkdir -p "$LOG_DIR" "$BACKUP_DIR" 2>/dev/null || true

    log_info "=========================================="
    log_info "RegRecord Backup - $SCRIPT_NAME"
    log_info "Mode: $mode"
    log_info "PID: $$ | Host: $(hostname)"
    log_info "Backup dir: $BACKUP_DIR"
    log_info "=========================================="

    acquire_lock

    case "$mode" in
        --full)
            run_full_backup
            ;;
        --incremental)
            log_info "Incremental backup delegates to full for this version"
            run_full_backup
            ;;
        --audit-only)
            run_audit_only_backup
            ;;
        --verify)
            run_verify
            ;;
        --cleanup)
            run_cleanup
            ;;
        *)
            echo "Usage: $SCRIPT_NAME {--full|--incremental|--audit-only|--verify|--cleanup}" >&2
            exit 1
            ;;
    esac

    local rc=$?
    log_info "Backup finished with exit code: $rc"
    exit $rc
}

main "$@"
