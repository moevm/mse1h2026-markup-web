#!/bin/bash
docker-compose up -d

python3 -m venv ./frontend-server/.venv
./frontend-server/.venv/bin/pip install -r ./frontend-server/requirements.txt
./frontend-server/.venv/bin/python ./frontend-server/main.py