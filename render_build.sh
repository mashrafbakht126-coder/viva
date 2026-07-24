#!/usr/bin/env bash
set -e
pip install --upgrade pip
pip install -r requirements.txt --prefer-binary
python -m spacy download en_core_web_sm
