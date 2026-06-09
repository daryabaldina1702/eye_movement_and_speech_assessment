# EyeTracking Service

Сервис для загрузки и анализа данных eye-tracking и аудиозаписей с использованием Django REST Framework и моделей машинного обучения.

## Требования

* Python 3.12+
* PostgreSQL 14+
* pip

## Клонирование проекта

```bash
git clone <repository_url>
cd eyetracking_service
```

## Создание виртуального окружения

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## Установка зависимостей

```bash
pip install -r requirements.txt
```

## Настройка базы данных PostgreSQL

Создать базу данных:

```sql
CREATE DATABASE eyetracking_db;
```

В файле `eyetracking_service/settings.py` проверить параметры подключения:

```python
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'eyetracking_db',
        'USER': 'postgres',
        'PASSWORD': '1234',
        'HOST': 'localhost',
        'PORT': '5432',
    }
}
```

При необходимости заменить значения на свои.

## Применение миграций

```bash
python manage.py migrate
```

## Создание администратора (необязательно)

```bash
python manage.py createsuperuser
```

## Запуск сервера

```bash
python manage.py runserver
```

После запуска сервис будет доступен по адресу:

```text
http://127.0.0.1:8000/
```

## Swagger API

Интерактивная документация:

```text
http://127.0.0.1:8000/swagger/
```

## Основные API-эндпоинты

### Загрузка данных

```text
POST /upload/single/
POST /upload/multiple/
```

### Метрики

```text
GET /trials/
GET /blocks/
GET /qc/
```

### Работа с аудио

```text
POST /speech/upload/
GET /speech/
```

### Предсказания ML

```text
POST /ml/predict/<recording_id>/
POST /ml/predict-unimodal/<recording_id>/
GET /ml/history/<recording_id>/
```

### Генерация PDF-отчёта

```text
GET /report/pdf/<recording_id>/
```

## Запуск тестов

Запустить все тесты проекта:

```bash
python manage.py test
```

Запустить тесты конкретного приложения:

```bash
python manage.py test pipeline
```

Запустить конкретный модуль с тестами:

```bash
python manage.py test pipeline.tests
```

Запустить отдельный тестовый метод:

```bash
python manage.py test pipeline.tests.TestClassName.test_method_name
```