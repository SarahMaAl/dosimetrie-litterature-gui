# Outil de dosimétrie – méthodes de la littérature (GUI)

Application **Tkinter** d’aide à la dosimétrie (radiothérapie) :
- Prescriptions (PTV/OAR/PRV)
- %Rx → Gy (auto) : 80, 90, 95, 98, 100, 105, 107, 110 %
- Marges par gradient g [%/mm] : mode **Manuel** (D1/D2) ou **Structures** (Rx×% / absolu)
- Anneaux (rings) TG-263 : noms, δ pour Dmax, Dmean au centre
- Ruler Gradient : conversions % ↔ mm ↔ Gy
- Conseiller heuristique de pondération
- Checklist d’optimisation (rappel de g ACTUEL)
- DVH & Indices (ICRU 83, gEUD, Paddick) [optionnel] + tracé
- Calculette Dose ↔ %Rx
- Lint de nomenclature (TG-263 + style PTV Relatif/cGy)

## Prérequis
- Python **3.10+**
- Tkinter (inclus avec Python sur Windows/macOS ; sur Linux : `sudo apt-get install python3-tk`)
- Voir `requirements.txt` pour les dépendances Python.

## Installation
```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
