FROM python:3.10-slim

WORKDIR /app
RUN pip install --no-cache-dir flask prometheus-client
COPY app.py /app/app.py

EXPOSE 8080
CMD ["python", "app.py"]

