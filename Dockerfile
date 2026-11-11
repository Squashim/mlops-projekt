# Etap 1: budowanie, instalacja zaleznosci do wydzielonego katalogu
FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-api.txt
# Etap 2: finalny obraz uruchomieniowy
FROM python:3.12-slim
WORKDIR /app
# Przeniesienie gotowych pakietow z etapu budowania
COPY --from=builder /install /usr/local
# Konfiguracja i kod aplikacji
COPY config.yaml .
COPY src/ ./src/
# Utworzenie grupy i użytkownika nieuprzywilejowanego oraz nadanie uprawnień do katalogu aplikacji
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser && \
    chown -R appuser:appgroup /app

# Przejście na użytkownika nieuprzywilejowanego
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]