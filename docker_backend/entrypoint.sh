#!/bin/bash
set -e

echo "Ожидание запуска PostgreSQL..."
until python -c "
import psycopg2, os
psycopg2.connect(
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    dbname=os.getenv('DB_NAME')
)
" 2>/dev/null; do
    echo "PostgreSQL недоступен, ждём..."
    sleep 2
done

echo "PostgreSQL готов, создаём таблицы..."
python -c "from activate import create_db_tables; create_db_tables()"

echo "Запускаем бэкенд..."
exec uvicorn main_backend:app --host 0.0.0.0 --port 8000
