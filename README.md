Tytuł projektu: mlops-projekt\
Numer albumu: 99660

### Ćw.1
| Narzędzie | Czas instalacji pakietów (numpy, scikit-learn) | Plik definicji zależności | Sposób odtworzenia środowiska | Kiedy stosować? |
| :--- | :--- | :--- | :--- | :--- |
| **venv** | 33 sekundy | `requirements.txt` | `pip install -r requirements.txt` | Proste projekty w Pythonie |
| **uv** | 512 milisekund | `pyproject.toml`, `uv.lock` lub `requirements.txt` | `uv pip install -r requirements.txt` | Nowoczesne projekty Pythonowe, potoki CI/CD |
| **conda** | 11,5 sekundy | `environment.yml` | `conda env create -f environment.yml` | Korzystanie z zewnętrznych bibliotek oraz akceleracji GPU (CUDA) |
| **DVC** | Nie dotyczy | Pliki `*.dvc` | `dvc pull` | Zarządzanie dużymi zbiorami danych i ich wersjonowanie (w odróżnieniu od Gita, który służy tylko do śledzenia kodu, DVC wersjonuje duże pliki bez obciążania repozytorium, trzymając je w zewnętrznej pamięci) |

## Porównanie rozmiarów obrazów Docker

Wdrożono wieloetapowe budowanie obrazu (multi-stage build), co pozwoliło na zmniejszenie jego ostatecznej wagi.

| Wariant budowania | Rozmiar obrazu |
| :--- | :--- |
| **Jednoetapowy (Single-stage)** | 1.68 GB |
| **Wieloetapowy (Multi-stage)** | 1.28 GB |

**Źródło różnicy:**
W klasycznym podejściu jednoetapowym obraz końcowy zawiera pamięć podręczną menedżera pakietów (np. cache pip) oraz narzędzia budujące. Zastosowanie wariantu wieloetapowego pozwala na instalację ciężkich bibliotek (jak scikit-learn czy MLflow) w etapie "builder", a następnie skopiowanie wyłącznie wybudowanych pakietów do finalnego obrazu. Dzięki temu udało się zaoszczędzić około 400 MB, eliminując zbędne pliki instalacyjne i pozostałości po procesie budowania.