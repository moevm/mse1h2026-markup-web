# mse-markup-web

## Установка и запуск
Перед установкой необходимо иметь docker на устройстве и python3
- для linux/macos необходимо запустить run.sh
- для windows необходимо запустить run.bat

## Проверка работоспособности

### Признаки успешной сборки и запуска

После выполнения `run.sh` / `run.bat` в терминале должны появиться строки, подтверждающие запуск фронтенд-сервера:

```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:5678 (Press CTRL+C to quit)
```

Чтобы убедиться, что все Docker-контейнеры собрались и запущены, выполните:

```bash
docker ps
```

Ожидаемый вывод — три контейнера со статусом `Up`:

```
CONTAINER ID   IMAGE              COMMAND                  STATUS          PORTS
xxxxxxxxxxxx   <project>-backend  ...                      Up X seconds    127.0.0.1:8000->8000/tcp
xxxxxxxxxxxx   dpage/pgadmin4     ...                      Up X seconds    127.0.0.1:5050->80/tcp
xxxxxxxxxxxx   postgres:16-alpine ...                      Up X seconds
```

Если какой-либо контейнер отсутствует или имеет статус `Exited`, проверьте логи командой:

```bash
docker logs <CONTAINER ID>
```

---

После выполнения `run.sh` / `run.bat` должны быть доступны следующие адреса:

| Сервис | URL |
|---|---|
| Веб-приложение (фронтенд) | http://localhost:5678 |
| API бэкенда | http://localhost:8000 |
| pgAdmin (БД) | http://localhost:5050 |

### Что должно работать

**Главная страница** (`http://localhost:5678`)
- Открывается страница с приветствием и анимацией
- Отображается секция «Недавние датасеты» (пустая при первом запуске)
- Кнопки «Загрузить новый датасет» и «Выбрать существующий» кликабельны

**Страница датасетов** (`http://localhost:5678/datasets`)
- Открывается список датасетов (пустой при первом запуске)
- Доступна кнопка добавления нового датасета

**Рабочая область** (`http://localhost:5678/work`)
- Открывается интерфейс разметки изображений

**Статистика** (`http://localhost:5678/stats`)
- Открывается страница со статистикой и графиками

**Проверка API бэкенда**

Открыть в браузере http://localhost:8000/api/getDatasets — должен вернуться JSON-ответ вида `[]` (пустой список при первом запуске).

Swagger-документация API доступна по адресу http://localhost:8000/docs.

**pgAdmin** (`http://localhost:5050`)
- Логин: `admin@admin.com`, пароль: `admin`
- База данных `markup_db` должна быть доступна для подключения

