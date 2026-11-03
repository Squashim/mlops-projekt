import numpy as np
from fastapi.testclient import TestClient

from src.api import LICZBA_CECH, app


class MockModel:
    """Obiekt zastępczy (mock) udający model klasyfikacyjny z scikit-learn."""

    def predict(self, X):
        return np.ones(X.shape[0], dtype=int)

    def predict_proba(self, X):
        proba = np.zeros((X.shape[0], 2))
        proba[:, 0] = 0.1
        proba[:, 1] = 0.9
        return proba


app.state.model = MockModel()
client = TestClient(app)


def test_health():
    """Test punktu /health - weryfikuje czy aplikacja działa i czy model
    jest przypisany."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_zaladowany": True}


def test_predict_success():
    """Test punktu /predict - poprawna predykcja."""
    prawidlowe_cechy = [0.5] * LICZBA_CECH
    response = client.post("/predict", json={"cechy": prawidlowe_cechy})
    assert response.status_code == 200
    dane = response.json()
    assert dane["klasa"] == 1
    assert dane["prawdopodobienstwo"] == 0.9


def test_predict_validation_error():
    """Test punktu /predict - odrzucenie z powodu braku wymaganej
    liczby cech (kod 422)."""
    nieprawidlowe_cechy = [0.5] * (LICZBA_CECH - 1)
    response = client.post("/predict", json={"cechy": nieprawidlowe_cechy})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_predict_batch_success():
    """Test punktu /predict-batch - poprawne przetwarzanie
    paczki obserwacji."""
    obserwacje = [{"cechy": [0.1] * LICZBA_CECH}, {"cechy": [0.9] * LICZBA_CECH}]
    response = client.post("/predict-batch", json={"obserwacje": obserwacje})
    assert response.status_code == 200
    dane = response.json()

    assert "predykcje" in dane
    assert len(dane["predykcje"]) == 2

    for predykcja in dane["predykcje"]:
        assert predykcja["klasa"] == 1
        assert predykcja["prawdopodobienstwo"] == 0.9
