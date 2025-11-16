#!/bin/bash
# Extract script for offline deployment
# Extracts the archived project components

set -e

if [ $# -eq 0 ]; then
    echo "Usage: $0 <archive_file.tar.gz>"
    echo "Example: $0 project_archive_20251116_123456.tar.gz"
    exit 1
fi

ARCHIVE_FILE="$1"

if [ ! -f "${ARCHIVE_FILE}" ]; then
    echo "ERROR: Archive file not found: ${ARCHIVE_FILE}"
    exit 1
fi

echo "Extracting archive: ${ARCHIVE_FILE}"
echo ""

# Extract archive
tar -xzf "${ARCHIVE_FILE}"

echo ""
echo "Extraction complete!"
echo ""
echo "Extracted components:"
[ -f "main.py" ] && echo "  ✓ main.py"
[ -f "pyproject.toml" ] && echo "  ✓ pyproject.toml"
[ -d "scaledcnn" ] && echo "  ✓ scaledcnn/"
[ -d "svm" ] && echo "  ✓ svm/"
[ -d ".cache/processed_datasets" ] && echo "  ✓ .cache/processed_datasets/"

