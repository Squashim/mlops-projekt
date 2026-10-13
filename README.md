Tytuł projektu: mlops-projekt\
Numer albumu: 99660

### Ćw.1
| Narzędzie | Czas instalacji pakietów (numpy, scikit-learn) | Plik definicji zależności | Sposób odtworzenia środowiska | Kiedy stosować? |
| :--- | :--- | :--- | :--- | :--- |
| **venv** | 33 sekundy | `requirements.txt` | `pip install -r requirements.txt` | Proste projekty w Pythonie |
| **uv** | 512 milisekund | `pyproject.toml`, `uv.lock` lub `requirements.txt` | `uv pip install -r requirements.txt` | Nowoczesne projekty Pythonowe, potoki CI/CD |
| **conda** | 11,5 sekundy | `environment.yml` | `conda env create -f environment.yml` | Korzystanie z zewnętrznych bibliotek oraz akceleracji GPU (CUDA) |
| **DVC** | Nie dotyczy | Pliki `*.dvc` | `dvc pull` | Zarządzanie dużymi zbiorami danych i ich wersjonowanie (w odróżnieniu od Gita, który służy tylko do śledzenia kodu, DVC wersjonuje duże pliki bez obciążania repozytorium, trzymając je w zewnętrznej pamięci) |