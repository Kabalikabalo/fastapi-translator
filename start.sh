#!/bin/sh
pip install -r requirements.txt
python -m spacy download fr_core_news_md  # Manually download the model
uvicorn translator_api:app --host 0.0.0.0 --port $PORT
