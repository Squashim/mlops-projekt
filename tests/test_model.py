import numpy as np

from src.data import load_data, split_data
from src.evaluate import evaluate_model
from src.train import train_model


def test_load_data():
    """Test danych: wymiary, kompletnosc i liczba klas."""
    X, y = load_data()
    # Liczba obserwacji w X i y musi byc zgodna
    assert X.shape[0] == y.shape[0]
    # Zbior nie moze zawierac wartosci pustych
    assert not np.isnan(X).any()
    # Problem jest klasyfikacja binarna, oczekiwane sa dwie klasy
    assert len(np.unique(y)) == 2


def test_ksztalt_predykcji():
    """Test kontraktu: ksztalt i dziedzina predykcji."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    predykcje = model.predict(X_test)
    # Liczba predykcji musi odpowiadac liczbie obserwacji testowych
    assert predykcje.shape[0] == X_test.shape[0]
    # Predykcje moga przyjmowac wylacznie wartosci etykiet klas
    assert set(np.unique(predykcje)).issubset({0, 1})


def test_prog_trafnosci():
    """Test walidacyjny: minimalna akceptowalna jakosc modelu."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evaluate_model(model, X_test, y_test)
    # Model gorszy od progu oznacza regresje jakosci w projekcie
    assert accuracy > 0.9


def test_wielkosc_podzialu_danych():
    """Weryfikacja zgodności liczności podzbiorów i proporcji."""
    X, y = load_data()
    test_size = 0.2

    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

    assert X_train.shape[0] == y_train.shape[0], "Niezgodna liczba X_train i y_train"
    assert X_test.shape[0] == y_test.shape[0], "Niezgodna liczba X_test i y_test"

    assert X_train.shape[0] + X_test.shape[0] == X.shape[0], (
        "Zgubiono obserwacje podczas podziału"
    )

    oczekiwany_rozmiar_testowy = int(X.shape[0] * test_size)
    assert abs(X_test.shape[0] - oczekiwany_rozmiar_testowy) <= 1, (
        "Rozmiar testowy nie odpowiada parametrowi test_size"
    )


def test_stratyfikacja_proporcji_klas():
    """Sprawdzenie zachowania proporcji klas po podziale stratyfikowanym."""
    X, y = load_data()
    test_size = 0.2
    _, _, y_train, y_test = split_data(X, y, test_size=test_size)

    oryginalna_proporcja = np.mean(y == 1)

    train_proporcja = np.mean(y_train == 1)
    test_proporcja = np.mean(y_test == 1)

    assert np.isclose(train_proporcja, oryginalna_proporcja, atol=0.05), (
        f"Zaburzone proporcje w train: oczekiwano {oryginalna_proporcja:.2f}, "
        f"jest {train_proporcja:.2f}"
    )
    assert np.isclose(test_proporcja, oryginalna_proporcja, atol=0.05), (
        f"Zaburzone proporcje w test: oczekiwano {oryginalna_proporcja:.2f}, "
        f"jest {test_proporcja:.2f}"
    )
