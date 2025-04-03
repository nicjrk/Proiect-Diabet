# Imagine de bază cu Python
FROM python:3.10-slim

# Directorul de lucru
WORKDIR /app

# Copiază requirements și instalează
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiază tot proiectul
COPY . .

# Expune portul
EXPOSE 8080

# Comanda de start
CMD ["gunicorn", "-b", ":8080", "main:app"]
