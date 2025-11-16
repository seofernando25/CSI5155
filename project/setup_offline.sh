#!/bin/bash
set -e

rm -rf .venv
virtualenv .venv
source .venv/bin/activate
pip install -e .
echo "Setup complete! Activate with: source .venv/bin/activate"

