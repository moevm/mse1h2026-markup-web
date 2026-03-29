docker-compose up -d 

python -m venv .\frontend-server\.venv
.\frontend-server\.venv\Scripts\pip install -r .\frontend-server\requirements.txt
.\frontend-server\.venv\Scripts\python .\frontend-server\main.py