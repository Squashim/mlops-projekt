import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow.sklearn
import numpy as np
import yaml
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

SCIEZKA_KONFIGURACJI = Path(__file__).resolve().parent.parent / "config.yaml"

with open(SCIEZKA_KONFIGURACJI, encoding="utf-8") as plik:
    KONFIGURACJA = yaml.safe_load(plik)

MLFLOW_URI = os.getenv("MLFLOW_URI", KONFIGURACJA["mlflow"]["tracking_uri"])
NAZWA_MODELU = KONFIGURACJA["model"]["nazwa"]
ALIAS_MODELU = KONFIGURACJA["model"]["alias"]
LICZBA_CECH = KONFIGURACJA["model"]["liczba_cech"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ladowanie modelu przy starcie aplikacji i sprzatanie przy
    zamknieciu."""
    mlflow.set_tracking_uri(MLFLOW_URI)
    app.state.model = mlflow.sklearn.load_model(
        f"models:/{NAZWA_MODELU}@{ALIAS_MODELU}"
    )
    yield
    app.state.model = None


class ZadaniePredykcji(BaseModel):
    cechy: list[float] = Field(
        ...,
        min_length=LICZBA_CECH,
        max_length=LICZBA_CECH,
        description="Wektor cech pojedynczej obserwacji",
    )


class ZadaniePredykcjiBatch(BaseModel):
    obserwacje: list[ZadaniePredykcji] = Field(
        description="Lista wektorów cech do predykcji"
    )


class OdpowiedzPredykcji(BaseModel):
    klasa: int = Field(description="Przewidziana etykieta klasy")
    prawdopodobienstwo: float = Field(
        ge=0.0,
        le=1.0,
        description="Prawdopodobienstwo przewidzianej klasy",
    )


class OdpowiedzPredykcjiBatch(BaseModel):
    predykcje: list[OdpowiedzPredykcji] = Field(
        description="Lista wyników predykcji dla przekazanej paczki"
    )


class OdpowiedzModelInfo(BaseModel):
    model_name: str = Field(description="Nazwa modelu")
    model_version: str = Field(description="Wersja modelu")
    metrics: dict[str, float] = Field(
        description="Metryki modelu z ostatniego uruchomienia"
    )


app = FastAPI(
    title="API projektu semestralnego",
    description="Serwowanie modelu klasyfikacyjnego z rejestru MLflow",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    model_zaladowany = getattr(app.state, "model", None) is not None
    return {"status": "ok", "model_zaladowany": model_zaladowany}


@app.post("/predict", response_model=OdpowiedzPredykcji)
def predict(zadanie: ZadaniePredykcji):
    obserwacja = np.array(zadanie.cechy).reshape(1, -1)
    klasa = int(app.state.model.predict(obserwacja)[0])
    prawdopodobienstwa = app.state.model.predict_proba(obserwacja)[0]
    return OdpowiedzPredykcji(
        klasa=klasa,
        prawdopodobienstwo=float(prawdopodobienstwa[klasa]),
    )


@app.post("/predict-batch", response_model=OdpowiedzPredykcjiBatch)
def predict_batch(zadanie_batch: ZadaniePredykcjiBatch):
    macierz_cech = np.array([obs.cechy for obs in zadanie_batch.obserwacje])

    klasy = app.state.model.predict(macierz_cech)
    prawdopodobienstwa = app.state.model.predict_proba(macierz_cech)

    wyniki = []
    for klasa, prawdopodobienstwo_dla_klas in zip(klasy, prawdopodobienstwa):
        klasa_int = int(klasa)
        wyniki.append(
            OdpowiedzPredykcji(
                klasa=klasa_int,
                prawdopodobienstwo=float(prawdopodobienstwo_dla_klas[klasa_int]),
            )
        )

    return OdpowiedzPredykcjiBatch(predykcje=wyniki)


@app.get("/model-info", response_model=OdpowiedzModelInfo)
def model_info():
    client = MlflowClient(tracking_uri=MLFLOW_URI)

    model_version_details = client.get_model_version_by_alias(
        name=NAZWA_MODELU, alias=ALIAS_MODELU
    )
    run = client.get_run(model_version_details.run_id)

    return OdpowiedzModelInfo(
        model_name=NAZWA_MODELU,
        model_version=model_version_details.version,
        metrics=run.data.metrics,
    )
