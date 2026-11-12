FROM python:3.12-slim

WORKDIR /app
RUN pip install --upgrade pip

RUN pip install aiohttp==3.12.15

COPY main.py main.py

ENTRYPOINT ["python", "main.py"]