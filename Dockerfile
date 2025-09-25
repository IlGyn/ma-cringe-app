# Базовый образ Python 3.12 slim
########## Builder stage ##########
FROM python:3.12-slim AS builder

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем только requirements для лучшего кэширования слоёв
COPY requirements.txt ./

# Создаём изолированное окружение и ставим зависимости
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt


########## Runtime stage ##########
FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    tini \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Непривилегированный пользователь
RUN useradd -m -u 1000 appuser

# Копируем установленное окружение и исходники
COPY --from=builder /opt/venv /opt/venv
COPY . /app
RUN chown -R appuser:appuser /app

EXPOSE 8501

USER appuser

ENTRYPOINT ["tini", "--"]
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Healthcheck (лёгкий, без curl/wget)
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; url='http://127.0.0.1:8501/_stcore/health'; sys.exit(0 if urllib.request.urlopen(url, timeout=3).getcode()==200 else 1)" || exit 1