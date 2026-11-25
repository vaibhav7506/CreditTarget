# Multi-stage production container for CreditTarget: Python 3.11 + OpenJDK 17
FROM eclipse-temurin:17-jre-jammy AS jre

FROM python:3.11-slim-bookworm

ENV JAVA_HOME=/opt/java/openjdk
COPY --from=jre $JAVA_HOME $JAVA_HOME
ENV PATH="${JAVA_HOME}/bin:${PATH}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYSPARK_PYTHON=python3 \
    PYSPARK_DRIVER_PYTHON=python3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md Makefile /app/
COPY configs/ /app/configs/
COPY src/ /app/src/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

EXPOSE 8000

CMD ["credittarget", "serve", "--host", "0.0.0.0", "--port", "8000"]
