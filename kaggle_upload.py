import os
import json
from kaggle.api.kaggle_api_extended import KaggleApi

def upload_to_kaggle_notebook(notebook_path, dataset_name="arc-agi-solver"):
    """
    Uploaduje kod do nowego notebooka na Kaggle
    
    Args:
        notebook_path: ścieżka do pliku .py lub .ipynb
        dataset_name: nazwa datasetu na Kaggle
    """
    api = KaggleApi()
    api.authenticate()
    
    # Sprawdź czy dataset istnieje, jeśli nie - utwórz
    try:
        api.dataset_metadata(dataset_name)
    except:
        print(f"Tworzenie nowego datasetu: {dataset_name}")
        api.dataset_create_new(
            folder="./",  # folder z plikami do upload
            convert_to_csv=False,
            dir_mode="zip",
            metadata={
                "title": "ARC AGI Solver",
                "id": dataset_name,
                "licenses": [{"name": "CC0-1.0"}]
            }
        )
    
    # Upload pliku do datasetu
    api.dataset_create_version(
        folder="./",
        version_notes="Updated solver code",
        convert_to_csv=False,
        dir_mode="zip"
    )
    
    print(f"✅ Kod został uploadowany do datasetu: {dataset_name}")

def create_kaggle_notebook(notebook_path, title="ARC AGI Solver"):
    """
    Tworzy nowy notebook na Kaggle z Twoim kodem
    """
    api = KaggleApi()
    api.authenticate()
    
    # Wczytaj kod z pliku
    with open(notebook_path, 'r', encoding='utf-8') as f:
        code_content = f.read()
    
    # Utwórz notebook w formacie JSON
    notebook_data = {
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.8.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4,
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n\nARC AGI Solver - automatyczne rozwiązywanie zadań ARC"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [code_content]
            }
        ]
    }
    
    # Zapisz jako .ipynb
    notebook_file = "temp_notebook.ipynb"
    with open(notebook_file, 'w', encoding='utf-8') as f:
        json.dump(notebook_data, f, indent=2)
    
    print(f"✅ Utworzono notebook: {notebook_file}")
    print("Możesz go teraz ręcznie uploadować na Kaggle lub użyć API")

if __name__ == "__main__":
    # Przykład użycia
    print("Wybierz opcję:")
    print("1. Upload do datasetu")
    print("2. Utwórz notebook")
    
    choice = input("Twój wybór (1/2): ")
    
    if choice == "1":
        upload_to_kaggle_notebook("arc_agi_solver.py")
    elif choice == "2":
        create_kaggle_notebook("arc_agi_solver.py")
    else:
        print("Nieprawidłowy wybór") 