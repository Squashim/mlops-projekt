import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")


def main():
    client = MlflowClient()
    model_name = "breast_cancer"

    # 1. Pobranie obu wersji modelu z rejestru
    version_1 = client.get_model_version(name=model_name, version=1)
    version_2 = client.get_model_version(name=model_name, version=2)

    # 2. Odczytanie uruchomień powiązanych z każdą wersją
    run_v1 = client.get_run(version_1.run_id)
    run_v2 = client.get_run(version_2.run_id)

    # Odczytanie metryk
    acc_v1 = run_v1.data.metrics.get("accuracy", 0.0)
    f1_v1 = run_v1.data.metrics.get("f1_macro", 0.0)

    acc_v2 = run_v2.data.metrics.get("accuracy", 0.0)
    f1_v2 = run_v2.data.metrics.get("f1_macro", 0.0)

    # 3. Wypisanie porównania
    print(f"--- Porównanie modelu: {model_name} ---")
    print(f"Wersja 1: Accuracy = {acc_v1:.4f}, F1_macro = {f1_v1:.4f}")
    print(f"Wersja 2: Accuracy = {acc_v2:.4f}, F1_macro = {f1_v2:.4f}\n")

    # 4. Przepięcie aliasu champion
    if f1_v1 >= f1_v2:
        best_version = "1"
    else:
        best_version = "2"

    client.set_registered_model_alias(
        name=model_name, alias="champion", version=best_version
    )
    print(f"Alias 'champion' przypisano do wersji: {best_version}\n")


if __name__ == "__main__":
    main()
