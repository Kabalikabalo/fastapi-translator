#!/bin/sh
uvicorn translator_api:app --host 0.0.0.0 --port $PORT
