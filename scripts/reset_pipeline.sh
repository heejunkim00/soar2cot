#!/bin/bash
#
# Pipeline 완전 초기화 스크립트
# 파이프라인을 처음부터 다시 실행하고 싶을 때 사용합니다.
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=========================================="
echo "Pipeline Reset Script"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Delete progress.json (task completion tracking)"
echo "  2. Truncate database tables (instructions and guess)"
echo "  3. Clean up old log directories (optional)"
echo ""
echo "WARNING: This will permanently delete all progress and results!"
echo ""

# Confirm with user
read -p "Are you sure you want to reset the pipeline? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Reset cancelled."
    exit 0
fi

# Step 1: Delete progress.json
PROGRESS_FILE="$PROJECT_ROOT/data/progress.json"
if [ -f "$PROGRESS_FILE" ]; then
    echo ""
    echo "[1/3] Deleting progress.json..."
    rm "$PROGRESS_FILE"
    echo "✓ Deleted: $PROGRESS_FILE"
else
    echo ""
    echo "[1/3] progress.json not found (already clean)"
fi

# Step 2: Truncate database tables
echo ""
echo "[2/3] Truncating database tables..."

# Load database credentials from .env
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | grep NEON_DSN | xargs)
fi

if [ -z "$NEON_DSN" ]; then
    echo "ERROR: NEON_DSN not found in .env file"
    exit 1
fi

# Extract connection details from DSN
# Format: postgresql://user:password@host/database
DB_USER=$(echo $NEON_DSN | sed -n 's/.*\/\/\([^:]*\):.*/\1/p')
DB_PASSWORD=$(echo $NEON_DSN | sed -n 's/.*:\/\/[^:]*:\([^@]*\)@.*/\1/p')
DB_HOST=$(echo $NEON_DSN | sed -n 's/.*@\([^\/]*\)\/.*/\1/p')
DB_NAME=$(echo $NEON_DSN | sed -n 's/.*\/\([^?]*\).*/\1/p')

echo "Database: $DB_NAME @ $DB_HOST"
echo "User: $DB_USER"

# Truncate tables (CASCADE to handle foreign key constraints)
PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -d $DB_NAME << EOF
-- Truncate guess table first (child table)
TRUNCATE TABLE guess CASCADE;
-- Truncate instructions table
TRUNCATE TABLE instructions CASCADE;

-- Verify counts
SELECT COUNT(*) as instructions_count FROM instructions;
SELECT COUNT(*) as guess_count FROM guess;
EOF

echo "✓ Database tables truncated"

# Step 3: Clean up old logs (optional)
echo ""
echo "[3/3] Cleaning up old logs..."
read -p "Do you want to delete old log directories? (yes/no): " clean_logs

if [ "$clean_logs" == "yes" ]; then
    LOGS_DIR="$PROJECT_ROOT/logs"
    if [ -d "$LOGS_DIR" ]; then
        # Keep current log directory, delete older ones
        LOG_COUNT=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "run_*" | wc -l)
        if [ "$LOG_COUNT" -gt 0 ]; then
            echo "Found $LOG_COUNT log directories"
            echo "Deleting old logs..."
            find "$LOGS_DIR" -maxdepth 1 -type d -name "run_*" -exec rm -rf {} +
            echo "✓ Old logs deleted"
        else
            echo "No log directories found"
        fi
    else
        echo "Logs directory not found"
    fi
else
    echo "Keeping old logs"
fi

echo ""
echo "=========================================="
echo "Pipeline Reset Complete!"
echo "=========================================="
echo ""
echo "You can now run the pipeline from scratch:"
echo "  python -m src.run"
echo ""
