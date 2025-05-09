FROM python:3.11-slim

WORKDIR /app

# Installa le dipendenze di sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia i file di configurazione delle dipendenze
COPY pyproject.toml uv.lock ./

# Installa le dipendenze Python
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir gunicorn

# Copia il codice dell'applicazione
COPY . .

# Crea la directory per gli upload se non esiste
#RUN mkdir -p uploads

# Espone la porta su cui l'applicazione si avvierà
EXPOSE 5000

# Comando per avviare l'applicazione
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--reuse-port", "--workers", "4", "main:app"]