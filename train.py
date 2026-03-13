
# !pip install streamlit
# !pip install evaluate


# IMPORTS & CONFIGURATION


import re                        # Module de regex (expressions régulières) pour nettoyer les textes
import time                      # Pour mesurer la durée de l'entraînement
import warnings                  # Gestion des avertissements Python
warnings.filterwarnings("ignore") # Supprime les warnings non critiques pour garder les logs lisibles

import numpy as np               # Calculs numériques (argmax, moyenne, etc.)
import torch                     # Framework deep learning PyTorch
import torch.nn as nn            # Module neural networks de PyTorch (couches, loss, etc.)
import streamlit as st           # Framework pour créer l'interface web interactive

from collections import Counter  # Compteur d'occurrences — utilisé pour mesurer la distribution des classes
from datasets import load_dataset # Bibliothèque Hugging Face pour charger des datasets publics
import evaluate                  # Bibliothèque Hugging Face pour calculer des métriques (accuracy, F1…)

from transformers import (
    AutoTokenizer,                        # Charge automatiquement le bon tokenizer selon le modèle
    AutoModelForSequenceClassification,   # Charge un modèle BERT pré-entraîné adapté à la classification
    TrainingArguments,                    # Objet de configuration de l'entraînement (lr, epochs, etc.)
    Trainer,                              # Boucle d'entraînement clé-en-main de Hugging Face
    EarlyStoppingCallback,                # Arrête l'entraînement si le score ne s'améliore plus
    pipeline,                             # Interface simplifiée pour l'inférence (non utilisée ici finalement)
    DataCollatorWithPadding,              # Padde dynamiquement les batchs à la même longueur
)

from sklearn.metrics import classification_report, confusion_matrix
# classification_report • affiche précision/rappel/F1 par classe
# confusion_matrix      • tableau vrais/faux positifs/négatifs



# CONSTANTES


MODEL_NAME               = "camembert-base"    # Modèle BERT entraîné sur du français (équivalent de BERT mais pour le français)
OUTPUT_DIR               = "./modele_allocine"  # Dossier où le modèle fine-tuné sera sauvegardé
NUM_LABELS               = 2                    # 2 classes : 0 = Négatif, 1 = Positif
MAX_LENGTH               = 256                  # Nb max de tokens par texte (les textes plus longs sont tronqués)
BATCH_SIZE               = 16                   # Nb d'exemples traités simultanément — impact direct sur la RAM GPU
LEARNING_RATE            = 2e-5                 # Taux d'apprentissage faible : standard pour le fine-tuning BERT
NUM_EPOCHS               = 3                    # Nb de passages complets sur les données d'entraînement
DROPOUT_RATE             = 0.3                  # Taux de dropout : 30% des neurones désactivés aléatoirement pour éviter l'overfitting
WEIGHT_DECAY             = 1e-4                 # Régularisation L2 : pénalise les grands poids pour éviter l'overfitting
EARLY_STOPPING_PATIENCE  = 2                    # Stop si pas d'amélioration pendant 2 epochs consécutives
TRAIN_SIZE               = 5000                 # Sous-ensemble d'entraînement (le dataset complet fait 160k — réduit pour la démo)
TEST_SIZE                = 1000                 # Sous-ensemble de test



# ÉTAPE 1 — DÉFINITION DU PROBLÈME


# On définit clairement la tâche NLP avant de coder quoi que ce soit.
# Indispensable pour choisir les bons outils (modèle, métriques, dataset).
print("ÉTAPE 1 — DÉFINITION DU PROBLÈME NLP")
print("""
Tâche        : Classification binaire de sentiments   # On veut prédire 2 classes (pas de régression)
Entrée       : Critique de film en français            # Texte libre, longueur variable
Sortie       : 0 = Négatif  |  1 = Positif            # Label discret binaire
Dataset      : AlloFilm (Hugging Face)                 # Données réelles de critiques de films
Architecture : CamemBERT (Transfer Learning)           # On réutilise un modèle déjà entraîné sur du français
""")



# ÉTAPE 2 — COLLECTE DES DONNÉES


print("Chargement du dataset AlloFilm depuis Hugging Face...")
dataset = load_dataset("allocine")
# load_dataset télécharge et met en cache le dataset "allocine" depuis le Hub Hugging Face
# Il contient ~160k critiques de films françaises avec un label pos/neg

print(f"\nStructure : {dataset}")                              # Affiche les splits disponibles (train/test/validation)
print(f"Premier exemple :")
print(f"  Texte : {dataset['train'][0]['review'][:150]}...")   # Aperçu des 150 premiers caractères d'une critique
print(f"  Label : {dataset['train'][0]['label']}")             # Vérifie que le label est bien 0 ou 1

distribution = Counter(dataset["train"]["label"])              # Compte combien d'exemples positifs vs négatifs
total         = len(dataset["train"])                          # Taille totale du train set
print(f"\nDistribution des classes (train) :")
print(f"  Positifs : {distribution[1]:6} ({distribution[1]/total*100:.1f}%)")   # Vérifie l'équilibre des classes
print(f"  Négatifs : {distribution[0]:6} ({distribution[0]/total*100:.1f}%)")   # Un déséquilibre fort nécessiterait un rééchantillonnage

if TRAIN_SIZE:
    dataset["train"]      = dataset["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
    # shuffle(seed=42) : mélange aléatoire reproductible (seed fixe = mêmes résultats à chaque run)
    # select(range(TRAIN_SIZE)) : garde seulement les N premiers exemples après mélange
    dataset["test"]       = dataset["test"].shuffle(seed=42).select(range(TEST_SIZE))
    dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(TEST_SIZE // 2))
    # TEST_SIZE // 2 = 500 exemples pour la validation (moitié du test set)
    print(f"\n[Info] Sous-ensemble : {TRAIN_SIZE} train / {TEST_SIZE} test")



# ÉTAPE 3 — NETTOYAGE DES DONNÉES


print("ÉTAPE 3 — NETTOYAGE DES DONNÉES")

def nettoyer_texte(texte: str) -> str:
    """
    Nettoie un texte avant tokenisation et inférence.
    Définie ici et réutilisée à l'inférence (Étape 10) — cohérence train/test obligatoire.
    """
    if not isinstance(texte, str):          # Sécurité : retourne "" si l'entrée n'est pas une chaîne (NaN, None, etc.)
        return ""
    texte = re.sub(r"<[^>]+>",          " ", texte)  # Supprime les balises HTML (<b>, <br>, etc.)
    texte = re.sub(r"&[a-z]+;",         " ", texte)  # Supprime les entités HTML (&amp;, &eacute;, etc.)
    texte = re.sub(r"http\S+|www\.\S+", " ", texte)  # Supprime les URLs (inutiles pour le sentiment)
    texte = re.sub(r"([!?.]){2,}",   r"\1", texte)   # Réduit "!!!" • "!" (évite le sur-tokenisation)
    texte = re.sub(r"[\n\r\t]+",        " ", texte)  # Remplace retours à la ligne et tabulations par des espaces
    texte = re.sub(r" {2,}",            " ", texte)  # Supprime les espaces multiples consécutifs
    return texte.strip()                              # Supprime les espaces en début/fin de chaîne

print("Nettoyage des textes...")
dataset = dataset.map(
    lambda ex: {"review": [nettoyer_texte(t) for t in ex["review"]]},
    # Applique nettoyer_texte sur chaque critique du dataset
    batched=True,  # Traite les données par lots pour aller plus vite
)
print("Nettoyage terminé !")

# Démonstration visuelle du nettoyage :
exemple_brut = "Ce film était <b>magnifique</b> !!!!! Vraiment\n\npas déçu du tout !"
print(f"\nAvant : {exemple_brut}")
print(f"Après : {nettoyer_texte(exemple_brut)}")   # Montre l'effet concret de la fonction



# ÉTAPE 4 — TOKENISATION


print("ÉTAPE 4 — TOKENISATION")

print(f"Chargement du tokenizer '{MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# AutoTokenizer charge le tokenizer exact correspondant à CamemBERT
# Il doit être le même à l'entraînement ET à l'inférence pour cohérence
print(f"Vocabulaire : {tokenizer.vocab_size:,} tokens")   # CamemBERT a ~32k tokens

exemple_tok = "J'adore les films de science-fiction extraordinaires !"
print(f"\nExemple :")
print(f"  Texte  : {exemple_tok}")
print(f"  Tokens : {tokenizer.tokenize(exemple_tok)}")   # Montre comment le texte est découpé en sous-mots
print(f"  IDs    : {tokenizer.encode(exemple_tok)}")     # Chaque token • un entier (indice dans le vocabulaire)

print(f"\nTokenisation des données (MAX_LENGTH={MAX_LENGTH})...")
dataset_tokenise = dataset.map(
    lambda ex: tokenizer(
        ex["review"],
        truncation=True,      # Coupe les textes trop longs à MAX_LENGTH tokens
        padding="max_length", # Padde tous les textes à exactement MAX_LENGTH (requis pour les batchs)
        max_length=MAX_LENGTH,
    ),
    batched=True,   # Traitement en lots pour la performance
)
print("Tokenisation terminée !")

premier = dataset_tokenise["train"][0]
print(f"\nStructure après tokenisation :")
print(f"  Clés disponibles  : {list(premier.keys())}")          # Nouvelles colonnes ajoutées par le tokenizer
print(f"  input_ids (début) : {premier['input_ids'][:10]}...")  # Séquence d'entiers représentant les tokens
print(f"  attention_mask    : {premier['attention_mask'][:10]}...")
# attention_mask : 1 = token réel, 0 = padding • dit au modèle d'ignorer les tokens de padding
print(f"  label             : {premier['label']}")



# ÉTAPE 5 — WORD EMBEDDINGS


print("ÉTAPE 5 — WORD EMBEDDINGS")

embedding_dim    = 128                                           # Taille du vecteur représentant chaque token (128 dimensions)
couche_embedding = nn.Embedding(tokenizer.vocab_size, embedding_dim)
# nn.Embedding : table de correspondance token_id • vecteur dense
# C'est une couche apprise : au fil de l'entraînement, les mots similaires auront des vecteurs proches

ids_demo = torch.tensor(dataset_tokenise["train"][0]["input_ids"][:5])   # Prend les 5 premiers token IDs d'un exemple
vecteurs = couche_embedding(ids_demo)                                      # Convertit chaque ID en vecteur de 128 dimensions
print(f"Illustration nn.Embedding :")
print(f"  IDs d'entrée       : {ids_demo.tolist()}")          # Ex: [5, 1234, 678, ...]
print(f"  Shape des vecteurs : {vecteurs.shape}")             # (5, 128) : 5 tokens × 128 dimensions
print(f"  Premier vecteur    : {vecteurs[0][:8].tolist()}...") # Aperçu des 8 premières valeurs du vecteur



# ÉTAPE 6 — CONSTRUCTION DU MODÈLE


print("ÉTAPE 6 — CONSTRUCTION DU MODÈLE")

class ModeleNLP_LSTM(nn.Module):
    """LSTM illustratif — montré pour comparaison pédagogique avec CamemBERT."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout_p=0.3):
        super().__init__()                                                   # Initialise la classe parente nn.Module (obligatoire)
        self.embed         = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # padding_idx=0 : les tokens de padding (id=0) auront un vecteur nul, sans gradient
        self.dropout_embed = nn.Dropout(p=dropout_p)                         # Dropout après l'embedding pour régulariser
        self.lstm          = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        # LSTM : mémorise les dépendances à long terme dans la séquence de tokens
        # batch_first=True : le tensor est (batch, seq_len, features) plutôt que (seq_len, batch, features)
        self.dropout_fc    = nn.Dropout(p=dropout_p)                         # Dropout avant la couche finale
        self.fc            = nn.Linear(hidden_dim, num_classes)              # Couche de classification finale

    def forward(self, input_ids, **kwargs):
        x           = self.embed(input_ids)      # Token IDs • vecteurs d'embedding
        x           = self.dropout_embed(x)      # Dropout pour régularisation
        _, (h_n, _) = self.lstm(x)               # h_n = état caché final du LSTM (résumé de la séquence)
        h_n         = self.dropout_fc(h_n[-1])   # h_n[-1] = sortie de la dernière couche LSTM
        return self.fc(h_n)                      # Projection vers 2 classes (logits bruts)

modele_lstm    = ModeleNLP_LSTM(tokenizer.vocab_size, 128, 256, NUM_LABELS, DROPOUT_RATE)
nb_params_lstm = sum(p.numel() for p in modele_lstm.parameters() if p.requires_grad)
# p.numel() = nombre d'éléments dans le tenseur de paramètres
# requires_grad=True = paramètres appris (excluent les buffers figés)
print(f"Modèle A — LSTM from scratch : {nb_params_lstm:,} paramètres")   # ~quelques millions, entraîné de zéro

print(f"\nModèle B — CamemBERT (Transfer Learning) ← UTILISÉ :")
modele_camembert = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_LABELS,
)
# from_pretrained : charge les poids pré-entraînés sur des milliards de mots français
# num_labels=2 : ajoute une tête de classification binaire par-dessus le modèle BERT
nb_params_bert = sum(p.numel() for p in modele_camembert.parameters() if p.requires_grad)
print(f"  Paramètres  : {nb_params_bert:,}")                   # ~110M paramètres — déjà entraînés sur du français
print(f"  Architecture : 12 couches Transformer + 144 têtes d'attention")



# ÉTAPE 7 — ENTRAÎNEMENT


print("ÉTAPE 7 — ENTRAÎNEMENT DU MODÈLE")

metrique_accuracy = evaluate.load("accuracy")
# Charge la métrique accuracy depuis Hugging Face evaluate
# Sera appelée après chaque epoch pour évaluer le modèle sur la validation

def compute_metrics(eval_pred):
    logits, labels = eval_pred              # logits = scores bruts du modèle, labels = vrais labels
    predictions    = np.argmax(logits, axis=1)
    # argmax sur les logits = classe prédite (0 ou 1 selon le score le plus élevé)
    return metrique_accuracy.compute(predictions=predictions, references=labels)
    # Retourne un dict {"accuracy": 0.93} utilisé par le Trainer pour le early stopping

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Collateur dynamique : padde chaque batch à la longueur du plus long exemple du batch
# Plus efficace que padding="max_length" fixe car réduit les calculs inutiles

training_args = TrainingArguments(
    output_dir                   = OUTPUT_DIR,          # Dossier de sauvegarde des checkpoints
    num_train_epochs             = NUM_EPOCHS,          # Nombre d'epochs d'entraînement
    per_device_train_batch_size  = BATCH_SIZE,          # Taille de batch à l'entraînement (par GPU/CPU)
    per_device_eval_batch_size   = BATCH_SIZE * 2,      # Batch d'évaluation plus grand (pas de gradient • moins de mémoire)
    learning_rate                = LEARNING_RATE,       # Taux d'apprentissage pour l'optimizer AdamW
    weight_decay                 = WEIGHT_DECAY,        # Régularisation L2 intégrée dans AdamW
    eval_strategy                = "epoch",             # Évalue sur la validation à la fin de chaque epoch
    save_strategy                = "epoch",             # Sauvegarde un checkpoint à la fin de chaque epoch
    load_best_model_at_end       = True,                # Recharge automatiquement le meilleur checkpoint à la fin
    metric_for_best_model        = "accuracy",          # Critère pour choisir le "meilleur" modèle
    greater_is_better            = True,                # Plus l'accuracy est haute, mieux c'est
    lr_scheduler_type            = "cosine",            # Diminue le LR selon une courbe cosinus (évite les oscillations en fin d'entraînement)
    warmup_steps                 = 50,                  # Augmente progressivement le LR sur les 50 premiers steps (stabilise le début)
    logging_steps                = 50,                  # Affiche les logs tous les 50 steps
    seed                         = 42,                  # Reproductibilité des résultats
    fp16                         = torch.cuda.is_available(),
    # fp16=True si GPU disponible : utilise la précision float16 • 2× plus rapide, moins de VRAM
    dataloader_num_workers       = 0,                   # 0 = pas de parallélisme pour le chargement des données (plus stable en debug)
)

print(f"Configuration :")
print(f"  Epochs         : {NUM_EPOCHS}")
print(f"  Batch size     : {BATCH_SIZE}")
print(f"  Learning rate  : {LEARNING_RATE}")
print(f"  Early stopping : patience={EARLY_STOPPING_PATIENCE}")
print(f"  GPU disponible : {torch.cuda.is_available()}")

dataset_tokenise = dataset_tokenise.rename_column("label", "labels")
# Le Trainer Hugging Face attend une colonne nommée "labels" (avec 's') — on renomme pour respecter la convention
dataset_tokenise.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
# Convertit les colonnes en tenseurs PyTorch (le dataset est par défaut en format Python natif)

trainer = Trainer(
    model           = modele_camembert,              # Modèle à entraîner
    args            = training_args,                 # Configuration de l'entraînement
    train_dataset   = dataset_tokenise["train"],     # Données d'entraînement
    eval_dataset    = dataset_tokenise["validation"],# Données de validation (pour le monitoring)
    compute_metrics = compute_metrics,               # Fonction de calcul des métriques
    data_collator   = data_collator,                 # Collateur pour assembler les batchs
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    # Callback qui surveille l'accuracy de validation et arrête si pas d'amélioration pendant N epochs
)

print(f"\nLancement de l'entraînement ({len(dataset_tokenise['train'])} exemples)...")
debut        = time.time()          # Chronomètre le début de l'entraînement
train_result = trainer.train()      # Lance la boucle d'entraînement complète (forward + backward + optimizer)
print(f"\nEntraînement terminé en {(time.time()-debut)/60:.1f} min")   # Affiche la durée en minutes
print(f"Loss finale : {train_result.training_loss:.4f}")                # Loss = erreur finale (plus c'est bas, mieux c'est)

trainer.save_model(OUTPUT_DIR)          # Sauvegarde les poids du modèle fine-tuné
tokenizer.save_pretrained(OUTPUT_DIR)   # Sauvegarde le tokenizer associé (nécessaire pour l'inférence)
print(f"Modèle sauvegardé dans : {OUTPUT_DIR}/")



# ÉTAPE 8 — ÉVALUATION


print("ÉTAPE 8 — ÉVALUATION DU MODÈLE")

modele_camembert.eval()   # Passe en mode évaluation : désactive le dropout et le calcul de gradient (plus rapide)

print("Calcul des prédictions sur le jeu de test...")
sortie_test       = trainer.predict(dataset_tokenise["test"])
# predict() fait un forward pass sur tout le test set sans mise à jour des poids
predictions_test  = np.argmax(sortie_test.predictions, axis=1)   # Logits • classe prédite (argmax)
vrais_labels_test = sortie_test.label_ids                         # Vrais labels du test set

rapport = classification_report(
    vrais_labels_test,
    predictions_test,
    target_names=["Négatif (0)", "Positif (1)"],   # Noms lisibles des classes
    digits=4,                                        # Précision à 4 décimales
)
print(f"\nRapport de classification :")
print(rapport)
# Affiche : précision, rappel, F1-score et support pour chaque classe
# Précision = parmi les prédits positifs, combien sont vraiment positifs ?
# Rappel    = parmi les vrais positifs, combien ont été détectés ?
# F1        = moyenne harmonique précision/rappel

matrice = confusion_matrix(vrais_labels_test, predictions_test)
print(f"Matrice de confusion :")
print(f"             Prédit Nég  Prédit Pos")
print(f"  Vrai Nég :   {matrice[0][0]:6}       {matrice[0][1]:6}")   # Vrais négatifs | Faux positifs
print(f"  Vrai Pos :   {matrice[1][0]:6}       {matrice[1][1]:6}")   # Faux négatifs  | Vrais positifs

accuracy_finale = np.trace(matrice) / matrice.sum()
# np.trace = somme de la diagonale (vrais positifs + vrais négatifs)
# divisé par le total = accuracy globale
print(f"\nAccuracy finale : {accuracy_finale:.4f} ({accuracy_finale*100:.2f}%)")



# ÉTAPE 9 — RÉCAPITULATIF DES OPTIMISATIONS


print("ÉTAPE 9 — OPTIMISATIONS INTÉGRÉES")

print(f"  • Dropout          = {DROPOUT_RATE}")
# Désactive aléatoirement 30% des neurones • force le réseau à ne pas dépendre d'un neurone en particulier
print(f"  • Weight Decay     = {WEIGHT_DECAY}")
# Pénalise les grands poids • empêche le sur-apprentissage (régularisation L2)
print(f"  • Early Stopping   = patience {EARLY_STOPPING_PATIENCE}")
# Arrête avant d'overfitter si la validation ne s'améliore plus
print(f"  • Cosine Scheduler + warmup 10%")
# Le LR monte doucement au début (warmup) puis descend en cosinus • convergence plus stable
print(f"  • Mixed Precision  = {torch.cuda.is_available()}")
# FP16 sur GPU : divise la mémoire par 2 et accélère le calcul sans perte significative de précision