#!/bin/bash

# Script to clean an example directory while preserving config.ini and prior.prior files
# Usage: ./clean_example.sh <example_directory>
# Example: ./clean_example.sh example_1

set -e  # Exit on error

# Check if directory name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <example_directory>"
    echo "Example: $0 example_1"
    exit 1
fi

EXAMPLE_DIR="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/$EXAMPLE_DIR"

# Check if directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory $TARGET_DIR does not exist"
    exit 1
fi

# Create unique temporary names using timestamp and directory name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TEMP_CONFIG="${SCRIPT_DIR}/.tmp_config_${EXAMPLE_DIR}_${TIMESTAMP}.ini"
TEMP_PRIOR="${SCRIPT_DIR}/.tmp_prior_${EXAMPLE_DIR}_${TIMESTAMP}.prior"

echo "Cleaning $TARGET_DIR..."

# Step 1: Copy config.ini and prior.prior to one directory above (if they exist)
CONFIG_BACKED_UP=false
PRIOR_BACKED_UP=false

if [ -f "$TARGET_DIR/config.ini" ]; then
    echo "Backing up config.ini..."
    cp "$TARGET_DIR/config.ini" "$TEMP_CONFIG"
    CONFIG_BACKED_UP=true
fi

if [ -f "$TARGET_DIR/prior.prior" ]; then
    echo "Backing up prior.prior..."
    cp "$TARGET_DIR/prior.prior" "$TEMP_PRIOR"
    PRIOR_BACKED_UP=true
fi

# Step 2: Remove all files in the directory
echo "Removing files in $TARGET_DIR..."
rm -f "$TARGET_DIR"/*

# Step 3: Move the config.ini and prior.prior back
if [ "$CONFIG_BACKED_UP" = true ]; then
    echo "Restoring config.ini..."
    mv "$TEMP_CONFIG" "$TARGET_DIR/config.ini"
fi

if [ "$PRIOR_BACKED_UP" = true ]; then
    echo "Restoring prior.prior..."
    mv "$TEMP_PRIOR" "$TARGET_DIR/prior.prior"
fi

echo "✓ Cleanup complete for $EXAMPLE_DIR!"
echo "  - All files removed except config.ini and prior.prior"
