# otus_mlops_ct

Материалы к открытому уроку Отус "MLFlow и переобучение ML-моделей"

## Запуск mlflow + postgres + minio

В папке `data-gen`:

### Установка переменных окружения

Создайте файлик `.env`, в который добавьте примерно следующее:

```bash
# Имя пользователя для системной БД MLFlow
PG_USER=mlflow
# Пароль пользователя для системной БД MLFlow
PG_PASSWORD=mlflow
# Имя пользователя для системной БД MLFlow
PG_DATABASE=mlflow
# Имя бакета в Minio, куда MLFlow будет складывать артефакты
MLFLOW_BUCKET_NAME=otus-mlflow-bucket
# Имя суперпользователя в Minio
MINIO_ROOT_USER=admin
# Пароль суперпользователя в Minio
MINIO_ROOT_PASSWORD=dZh5222@
# URL, по которому MLFlow дозванивается до Minio
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
# URL, по которому MLFlow Tracking Server
MLFLOW_TRACKING_URI=http://localhost:5000
# Идентификатор ключа доступа в S3 (Minio)
MLFLOW_AWS_ACCESS_KEY_ID=pmwGHCLBuSTM5eHzyWmB
# Ключ доступа в Minio
MLFLOW_AWS_SECRET_ACCESS_KEY=26GOdhDyvBP78wEPuh4nP76KPUbeSNjZKWXGINtq
```

### Запуск сервисов

Выполните:

```bash
docker-compose up -d
```

Когда контейнеры стартанут, на: 

* `http://localhost:9001/` будет доступен GUI Minio (логин/пароль – переменные `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` в `.env`);
* на `http://localhost:5050/` будет доступен GUI MLFlow;

## Запуск сервиса генерации данных

```bash
fastapi dev data-gen/src/main.py
```