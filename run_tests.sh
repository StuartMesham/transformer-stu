#!/bin/bash
set -e

git ls-files "*.py" | xargs pytype --keep-going --jobs=auto
python -m pytest
