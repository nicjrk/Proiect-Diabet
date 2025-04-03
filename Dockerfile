# Imagine de bază cu Python
FROM python:3.10-slim

# Directorul de lucru
WORKDIR /app

# Copiază requirements.txt din root
COPY requirements.txt .

# Instalează dependințele
RUN pip install --no-cache-dir -r requirements.txt

# Copiază tot codul aplicației din subdirectorul diabet-app
COPY . .

# Expune portul folosit de aplicație
EXPOSE 8080

# Pornește aplicația Flask cu gunicorn
CMD ["gunicorn", "-b", ":8080", "main:app"]
