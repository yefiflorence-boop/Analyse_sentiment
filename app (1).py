import re                   # Expressions régulières pour nettoyer le texte (même logique que train.py)
import torch                # PyTorch — chargé ici surtout pour la détection GPU (device=-1)
import streamlit as st      # Framework web : transforme ce script Python en application interactive dans le navigateur
from transformers import pipeline   # Interface haut niveau de Hugging Face : charge un modèle + tokenizer en une ligne

# ── DOIT ÊTRE EN PREMIER ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analyseur de Sentiment 🎬",   # Titre affiché dans l'onglet du navigateur
    page_icon="🎬",                            # Favicon de la page
    layout="centered",                         # Contenu centré (vs "wide" qui prend toute la largeur)
)
# ⚠ Cette ligne DOIT être la première commande Streamlit du script
# Streamlit l'appelle avant de rendre quoi que ce soit — si elle vient après, ça plante

MAX_LENGTH = 256   # Limite de tokens envoyés au modèle — doit être cohérent avec ce qui a été utilisé à l'entraînement

# ── Nettoyage du texte ────────────────────────────────────────────────────────
def nettoyer_texte(texte: str) -> str:
    # Exactement la même fonction que dans train.py — CRITIQUE pour la cohérence train/inférence
    # Si on nettoie différemment à l'inférence qu'à l'entraînement, les performances chutent
    if not isinstance(texte, str):          # Sécurité : évite un crash si l'utilisateur envoie None ou un entier
        return ""
    texte = re.sub(r"<[^>]+>",          " ", texte)   # Supprime les balises HTML (<b>, <br/>, etc.)
    texte = re.sub(r"&[a-z]+;",         " ", texte)   # Supprime les entités HTML (&amp;, &eacute;, etc.)
    texte = re.sub(r"http\S+|www\.\S+", " ", texte)   # Supprime les URLs (non pertinentes pour le sentiment)
    texte = re.sub(r"([!?.]){2,}",   r"\1", texte)   # Réduit "!!!" → "!" (évite la sur-tokenisation)
    texte = re.sub(r"[\n\r\t]+",        " ", texte)   # Remplace sauts de ligne et tabulations par des espaces
    texte = re.sub(r" {2,}",            " ", texte)   # Supprime les espaces multiples consécutifs
    return texte.strip()                              # Supprime les espaces en début/fin de chaîne

# ── Chargement du modèle depuis HuggingFace Hub ───────────────────────────────
@st.cache_resource
# @st.cache_resource = décorateur Streamlit qui met le résultat en cache global
# Sans ce décorateur, le modèle serait rechargé à CHAQUE interaction utilisateur (clic, frappe…)
# → charger un modèle BERT prend 10-30 secondes : inacceptable en prod
# Avec ce décorateur : chargé UNE SEULE FOIS au démarrage, puis réutilisé pour toutes les requêtes
def charger_pipeline():
    return pipeline(
        "sentiment-analysis",    # Type de tâche : Hugging Face sait quel format de sortie produire
        # Modèle multilingue public disponible directement sans entraînement préalable
        # Il supporte le français parmi ~100 langues
        # ⚠ Pour utiliser ton propre modèle fine-tuné, remplacer par :
        # model = "ton-username/camembert-allocine"
        model     = "nlptown/bert-base-multilingual-uncased-sentiment",
        tokenizer = "nlptown/bert-base-multilingual-uncased-sentiment",  # Tokenizer associé au modèle (doit toujours correspondre)
        device    = -1,   # -1 = forcer le CPU (Streamlit Cloud n'a pas de GPU disponible)
    )                     # 0, 1, 2… désignerait un GPU spécifique

# ── Mapping labels → sentiment lisible ───────────────────────────────────────
def interpreter_label(label: str) -> tuple:
    # nlptown est un modèle d'analyse d'avis en étoiles (1 à 5) — pas directement pos/neg/neutre
    # Cette fonction fait la conversion : étoiles → sentiment humainement lisible
    nb_etoiles = int(label[0])   # "3 stars" → prend le 1er caractère → int("3") = 3
    if nb_etoiles <= 2:
        return "NÉGATIF", "😢", "red"      # 1-2 étoiles = sentiment négatif
    elif nb_etoiles == 3:
        return "NEUTRE", "😐", "orange"    # 3 étoiles = sentiment mitigé
    else:
        return "POSITIF", "😊", "green"    # 4-5 étoiles = sentiment positif
    # Retourne 3 valeurs : le label texte, l'emoji, la couleur → utilisés dans l'affichage

# ── Interface ─────────────────────────────────────────────────────────────────
st.title("🎬 Analyseur de Sentiment  (Critiques de Films de base)")   # Titre H1 de la page
st.markdown("Entrez une critique en **français** et le modèle prédit le sentiment.")
# st.markdown() supporte le Markdown : **gras**, *italique*, liens, etc.
st.divider()   # Ligne horizontale séparatrice (équivalent HTML <hr>)

texte_utilisateur = st.text_area(
    # Widget de saisie multi-lignes — retourne ce que l'utilisateur tape, sous forme de string
    label       = "Votre critique :",                                             # Libellé affiché au-dessus du champ
    placeholder = "Ex : Ce film était absolument magnifique, je recommande vivement !",  # Texte grisé quand le champ est vide
    height      = 150,                                                            # Hauteur du champ en pixels
)
# Streamlit re-exécute tout le script à chaque interaction
# texte_utilisateur contient donc toujours la valeur actuelle du champ

if st.button("Analyser", type="primary", use_container_width=True):
    # st.button() : crée un bouton, retourne True uniquement lors du clic
    # type="primary" : bouton coloré (style principal)
    # use_container_width=True : le bouton prend toute la largeur disponible

    if not texte_utilisateur.strip():
        # .strip() supprime les espaces — évite d'envoyer une chaîne vide ou juste des espaces au modèle
        st.warning("Veuillez saisir une critique avant d'analyser.")   # Bandeau orange d'avertissement
    else:
        analyseur = charger_pipeline()
        # Appelle la fonction cachée — retourne le pipeline déjà en mémoire si déjà chargé

        with st.spinner("Analyse en cours..."):
            # st.spinner() : affiche un loader animé pendant l'exécution du bloc
            # Indispensable car l'inférence peut prendre 1-3s sur CPU
            res = analyseur(
                nettoyer_texte(texte_utilisateur),   # Texte nettoyé avant envoi au modèle
                truncation=True,                      # Coupe si le texte dépasse MAX_LENGTH tokens
                max_length=MAX_LENGTH,
            )[0]
            # pipeline() retourne une liste — [0] prend le premier (et seul) résultat
            # Résultat : {"label": "4 stars", "score": 0.6734}

        label, emoji, couleur = interpreter_label(res["label"])
        # Déstructuration du tuple retourné par interpreter_label
        confiance             = res["score"]   # Score de confiance entre 0 et 1

        st.divider()
        st.markdown(f"### Résultat : :{couleur}[{emoji} {label}]")
        # Syntaxe Streamlit pour coloriser du texte : :red[texte], :green[texte], etc.
        # Le f-string insère dynamiquement la couleur et le label

        st.metric(label="Confiance du modèle", value=f"{confiance*100:.1f}%")
        # st.metric() : widget d'affichage de valeur clé (grand chiffre avec libellé)
        # :.1f → arrondi à 1 décimale (ex: 87.3%)

        st.progress(confiance)
        # Barre de progression entre 0.0 et 1.0 — visualise la confiance du modèle

        with st.expander("Détails"):
            # st.expander() : section repliable/dépliable — utile pour les infos secondaires
            st.write(f"**Texte nettoyé :** {nettoyer_texte(texte_utilisateur)}")   # Montre l'effet du nettoyage
            st.write(f"**Label brut :** `{res['label']}`")                          # Label original retourné par le modèle ("4 stars")
            st.write(f"**Score :** `{confiance:.4f}`")                              # Score brut à 4 décimales

st.divider()

# ── Exemples cliquables ───────────────────────────────────────────────────────
st.markdown("#### Exemples à tester")
col1, col2 = st.columns(2)
# st.columns(2) : divise l'espace en 2 colonnes côte à côte (comme une grille CSS)
# col1 et col2 sont des contextes — tout ce qu'on écrit dans "with col1:" s'affiche à gauche

with col1:
    st.markdown("😊 **Positifs**")
    st.code("Ce film est absolument magnifique !", language=None)
    # st.code() : affiche le texte dans un bloc monospace avec bouton "Copier"
    # language=None : pas de coloration syntaxique (c'est du texte, pas du code)
    st.code("Je recommande vivement ce chef-d'œuvre.", language=None)

with col2:
    st.markdown("😢 **Négatifs**")
    st.code("Quelle déception ! Scénario nul, deux heures perdues.", language=None)
    st.code("Le pire film de ma vie. Ennuyeux du début à la fin.", language=None)

st.caption("Modèle : nlptown/bert-base-multilingual-uncased-sentiment • Streamlit Cloud")
# st.caption() : texte petit et grisé — utilisé pour les métadonnées, crédits, notes de bas de page