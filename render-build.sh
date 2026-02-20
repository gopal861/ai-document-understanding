#!/usr/bin/env bash
set -e

pip install -r requirements.txt

export PLAYWRIGHT_BROWSERS_PATH=/opt/render/project/.playwright

playwright install chromium




