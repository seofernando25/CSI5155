#!/bin/bash
# Archive script for offline deployment
# Archives processed datasets, scaledcnn, svm, main.py, and pyproject.toml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCHIVE_NAME="project_archive_$(date +%Y%m%d_%H%M%S).tar.gz"
ARCHIVE_PATH="${SCRIPT_DIR}/${ARCHIVE_NAME}"

echo "Creating archive: ${ARCHIVE_NAME}"
echo ""

# Check if required directories/files exist
MISSING=0

if [ ! -d "${SCRIPT_DIR}/.cache/processed_datasets" ]; then
    echo "WARNING: .cache/processed_datasets not found (will be skipped)"
    MISSING=1
fi

if [ ! -d "${SCRIPT_DIR}/scaledcnn" ]; then
    echo "ERROR: scaledcnn/ directory not found"
    exit 1
fi

if [ ! -d "${SCRIPT_DIR}/svm" ]; then
    echo "ERROR: svm/ directory not found"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/main.py" ]; then
    echo "ERROR: main.py not found"
    exit 1
fi

if [ ! -f "${SCRIPT_DIR}/pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found"
    exit 1
fi

# Create archive
cd "${SCRIPT_DIR}"

# Build tar command with conditional inclusion
TAR_CMD="tar -czf ${ARCHIVE_PATH}"

# Add main.py
TAR_CMD="${TAR_CMD} main.py"

# Add pyproject.toml
TAR_CMD="${TAR_CMD} pyproject.toml"

# Add scaledcnn directory (excluding __pycache__)
TAR_CMD="${TAR_CMD} scaledcnn/ --exclude='scaledcnn/__pycache__' --exclude='scaledcnn/**/__pycache__'"

# Add svm directory (excluding __pycache__)
TAR_CMD="${TAR_CMD} svm/ --exclude='svm/__pycache__' --exclude='svm/**/__pycache__'"

# Add processed datasets if they exist
if [ -d "${SCRIPT_DIR}/.cache/processed_datasets" ]; then
    TAR_CMD="${TAR_CMD} .cache/processed_datasets/"
fi

# Execute tar command
eval "${TAR_CMD}"

# Get archive size
ARCHIVE_SIZE=$(du -h "${ARCHIVE_PATH}" | cut -f1)

echo ""
echo "Archive created successfully!"
echo "  File: ${ARCHIVE_NAME}"
echo "  Size: ${ARCHIVE_SIZE}"
echo ""
echo "To extract on target machine:"
echo "  tar -xzf ${ARCHIVE_NAME}"

