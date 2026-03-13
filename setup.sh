# Créer l'environnement virtuel
python -m venv venv

# L'activer
# Sur Mac/Linux :
#source venv/bin/activate
# Sur Windows :
venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt


# Structure finale du repo GitHub :

nom_repo/
├── app.py
├── requirements.txt
└── setup.sh        ← optionnel, uniquement en local


# Note importante : `setup.sh` et `venv/` ne sont pas à pousser sur GitHub
# Streamlit Cloud lit uniquement `requirements.txt` 
# et installe les packages automatiquement dans son propre environnement. Ajoute `venv/` à ton `.gitignore`

**`.gitignore`**
```
venv/
__pycache__/
*.pyc
modele_allocine/
logs/
.env