# Используем официальный образ Python (версия 3.10, как в вашем amvera.yaml)
FROM python:3.10-slim

# Устанавливаем ffmpeg — он необходим вашему коду для конвертации аудио
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь остальной код приложения
COPY . .

# Открываем порт 80 (Amvera ожидает, что приложение будет слушать этот порт)
EXPOSE 80

# Команда для запуска вашего FastAPI-приложения через uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
