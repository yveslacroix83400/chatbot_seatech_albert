import os
import re
import json
import pickle
import time
import uuid
try:
    import vosk
except ImportError:
    vosk = None
import io
import wave
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session
from dotenv import load_dotenv
load_dotenv() #Charge les données de l'environnement de développement dnas le fichier main.


ENABLE_VOSK = os.getenv("ENABLE_VOSK", "false").strip().lower() == "true"

if ENABLE_VOSK:
    if vosk is None:
        raise RuntimeError("ENABLE_VOSK=true mais le paquet vosk n'est pas installé.")
    import urllib.request
    import zipfile
    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"
    model_path = "models/vosk-model-small-fr-0.22"
    if not os.path.exists(model_path):
        print("Téléchargement du modèle vocal Vosk...")
        os.makedirs("models", exist_ok=True)
        urllib.request.urlretrieve(model_url, "model.zip")
        with zipfile.ZipFile("model.zip", "r") as zip_ref:
            zip_ref.extractall("models")
        os.remove("model.zip")

# Importations légères pour le RAG distant Albert
import numpy as np
ML_IMPORTS_SUCCESS = True


try:
    from groq import Groq
    GROQ_IMPORT_SUCCESS = True
except ImportError:
    GROQ_IMPORT_SUCCESS = False

try:
    from openai import OpenAI
    ALBERT_IMPORT_SUCCESS = True
except ImportError:
    ALBERT_IMPORT_SUCCESS = False

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("seatech_chatbot")

# Répertoires et fichiers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
USER_DB_DIR = os.path.join(BASE_DIR, "database", "user_database")
CACHE_DIR = os.path.join(BASE_DIR, "cache")
for directory in [DATA_DIR, USER_DB_DIR, CACHE_DIR]:
    os.makedirs(directory, exist_ok=True)
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "albert_chunk_embeddings.pkl")
CHUNKS_CACHE = os.path.join(CACHE_DIR, "albert_chunks_with_sources.pkl")
QA_STORAGE = os.path.join(CACHE_DIR, "user_qa_memory.json")

# ===== CONFIGURATION =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

ALBERT_API_KEY = os.getenv("ALBERT_API_KEY", "")
ALBERT_BASE_URL = os.getenv("ALBERT_BASE_URL", "https://albert.api.etalab.gouv.fr/v1")
ALBERT_MODEL = os.getenv("ALBERT_MODEL", "")
ALBERT_EMBEDDING_MODEL = os.getenv("ALBERT_EMBEDDING_MODEL", "bge-m3")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

LLM_MODEL = "openai/gpt-oss-20b" #llama-3.3-70b-versatile
CONFIDENCE_THRESHOLD = 0.93

# Acronymes utilisés
ACRONYMS = {
    "SN": "Systèmes Numériques",
    "MTX": "Matériaux",
    "APP": "Apprentissage",
    "FISE": "Formation Initiale Sous Statut d'Étudiant",
    "FISA": "Formation Initiale Sous Statut d'Apprenti",
    "UTLN": "Université de Toulon"
}

# Mapping des profils avec leurs descriptions et mots-clés de priorisation
profile_mapping = {
    "Étudiant": {
        "description": "Fournis des informations utiles aux étudiants actuels de SeaTech : cours, emplois du temps, projets, vie associative, stages.",
        "keywords": ["cours", "emploi du temps", "projet", "stage", "vie associative", "examen", "planning", "td", "tp", "association", "événement étudiant", "logement", "bourse"],
        "priority_topics": ["cours", "planning", "examens", "stages", "projets"]
    },
    "Enseignant": {
        "description": "Fournis des informations utiles aux enseignants de SeaTech : ressources pédagogiques, contacts administratifs, organisation des cours.",
        "keywords": ["ressources pédagogiques", "contact administratif", "organisation", "enseignement", "programme", "évaluation", "moodle", "scolarité", "administration"],
        "priority_topics": ["ressources", "administration", "organisation", "programmes"]
    },
    "Candidat": {
        "description": "Fournis des informations utiles aux candidats intéressés par SeaTech : admissions, concours, dossiers, procédures, spécialités disponibles.",
        "keywords": ["admission", "concours", "dossier", "procédure", "candidature", "inscription", "formation", "spécialité", "parcours", "prérequis", "sélection"],
        "priority_topics": ["admissions", "formations", "candidature", "spécialités"]
    },
    "Alumni": {
        "description": "Fournis des informations utiles aux anciens étudiants de SeaTech : réseau alumni, partenariats, événements, relations entreprises.",
        "keywords": ["alumni", "réseau", "partenariat", "entreprise", "carrière", "emploi", "contact professionnel", "événement alumni", "relation entreprise"],
        "priority_topics": ["réseau", "carrière", "partenariats", "emploi"]
    }
}

# Pour l'exemple, nous définissons CONTACTS vide (à compléter selon vos besoins)
CONTACTS = {}

# ===== INITIALISATION DES CLIENTS =====
if GROQ_IMPORT_SUCCESS:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Client GROQ initialisé")
    except Exception as e:
        logger.error(f"Erreur initialisation client GROQ: {e}")
        groq_client = None
else:
    groq_client = None
    logger.warning("GROQ non disponible - vérifiez l'installation")
if ALBERT_IMPORT_SUCCESS and ALBERT_API_KEY:
    try:
        albert_client = OpenAI(
            base_url=ALBERT_BASE_URL,
            api_key=ALBERT_API_KEY
        )
        logger.info("Client Albert préparé")
    except Exception as e:
        logger.error(f"Erreur initialisation client Albert: {e}")
        albert_client = None
else:
    albert_client = None
    logger.info("Client Albert inactif : clé absente ou SDK indisponible")

def test_albert_connection(test_message="Réponds uniquement par : connexion Albert réussie."):
    """
    Teste isolément la connexion à Albert API.

    Cette fonction n'est pas appelée automatiquement et ne modifie pas
    le moteur utilisé par generate_answer().
    """
    if albert_client is None:
        return {
            "success": False,
            "error": "Client Albert indisponible. Vérifiez ALBERT_API_KEY et le SDK openai."
        }

    if not ALBERT_MODEL:
        return {
            "success": False,
            "error": "ALBERT_MODEL n'est pas configuré."
        }

    try:
        response = albert_client.chat.completions.create(
            model=ALBERT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un assistant de test technique. Réponds très brièvement."
                },
                {
                    "role": "user",
                    "content": test_message
                }
            ],
            temperature=0.0,
            max_tokens=50,
            stream=False
        )

        answer = response.choices[0].message.content

        return {
            "success": True,
            "model": ALBERT_MODEL,
            "answer": answer
        }

    except Exception as e:
        logger.error(f"Échec du test Albert : {type(e).__name__}: {e}")

        return {
            "success": False,
            "error_type": type(e).__name__,
            "error": str(e)
        }

embedding_model = None
logger.info("RAG léger activé : embeddings fournis par Albert API")


def handle_role_selection(session_id, selected_role=None):
    """Gère la sélection de rôle utilisateur et initialise la conversation."""
    print(f"handle_role_selection appelé avec session_id={session_id} et selected_role={selected_role}")
    if session_id not in conversation_history_global:
        conversation_history_global[session_id] = []
    print(f"Session {session_id} - Rôle sélectionné : {selected_role}")
    # Si un rôle est sélectionné, l'enregistrer dans le dictionnaire global
    if selected_role and selected_role in profile_mapping:
        # Stocker le profil dans le dictionnaire global (plus fiable que Flask Session)
        user_profiles_global[session_id] = {
            'role': selected_role,
            'confirmed': True
        }
        
        # Message de bienvenue personnalisé selon le rôle
        role_config = profile_mapping[selected_role]
        welcome_message = f"""
        <p>Parfait ! Je vais adapter mes réponses pour un profil <strong>{selected_role}</strong>.</p>
        <p>{role_config['description']}</p>
        <p>Vous pouvez maintenant me poser vos questions sur SeaTech !</p>
        """
        
        conversation_history_global[session_id].append({
            "role": "assistant",
            "content": welcome_message,
            "is_system_message": True
        })
        
        logger.info(f"Rôle '{selected_role}' confirmé pour la session {session_id}")
        logger.info(f"user_profiles_global après confirmation: {user_profiles_global}")
        return selected_role
    
    return None
# ===== FONCTIONS UTILITAIRES =====
def detect_user_role(query, conversation_history=None):
    """
    Détecte automatiquement le rôle de l'utilisateur basé sur sa question et l'historique.
    Retourne le rôle détecté et un score de confiance.
    """
    query_lower = query.lower()
    role_scores = {}
    
    # Analyse basée sur les mots-clés
    for role, config in profile_mapping.items():
        score = 0
        keywords = config["keywords"]
        
        # Score basé sur la présence de mots-clés
        for keyword in keywords:
            if keyword in query_lower:
                score += 2
        
        # Score basé sur les sujets prioritaires (poids plus élevé)
        for priority in config["priority_topics"]:
            if priority in query_lower:
                score += 3
                
        role_scores[role] = score
    
    # Analyse de l'historique de conversation si disponible
    if conversation_history:
        for entry in conversation_history[-3:]:  # Dernières 3 interactions
            if entry['role'] == 'user':
                content_lower = entry['content'].lower()
                for role, config in profile_mapping.items():
                    for keyword in config["keywords"]:
                        if keyword in content_lower:
                            role_scores[role] += 1
    
    # Détection de phrases spécifiques
    role_patterns = {
        "Candidat": [r"comment (postuler|candidater|s'inscrire)", r"quelles sont les conditions", r"admission", r"je veux intégrer"],
        "Étudiant": [r"mes cours", r"mon emploi du temps", r"quand est l'examen", r"projet de fin d'étude"],
        "Enseignant": [r"ressources pour", r"contact administration", r"organisation des cours"],
        "Alumni": [r"réseau", r"ancien élève", r"après diplôme", r"carrière"]
    }
    
    for role, patterns in role_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                role_scores[role] += 4
    
    # Retourner le rôle avec le score le plus élevé
    if role_scores:
        best_role = max(role_scores, key=role_scores.get)
        confidence = role_scores[best_role]
        
        # Si aucun score significatif, retourner "Candidat" par défaut (plus général)
        if confidence == 0:
            return "Candidat", 0.1
        
        return best_role, min(confidence / 10, 1.0)  # Normaliser entre 0 et 1
    
    return "Candidat", 0.1

def filter_chunks_by_role(chunks_results, detected_role, role_confidence):
    """
    Filtre et réordonne les chunks en fonction du rôle détecté de l'utilisateur.
    """
    if role_confidence < 0.3:  # Si la confiance est faible, ne pas filtrer
        return chunks_results
    
    role_config = profile_mapping.get(detected_role, profile_mapping["Candidat"])
    keywords = role_config["keywords"]
    priority_topics = role_config["priority_topics"]
    
    filtered_results = []
    
    for chunk_text, source, original_score in chunks_results:
        chunk_lower = chunk_text.lower()
        role_relevance_score = 0
        
        # Score basé sur les mots-clés du rôle
        for keyword in keywords:
            if keyword in chunk_lower:
                role_relevance_score += 0.1
        
        # Score bonus pour les sujets prioritaires
        for priority in priority_topics:
            if priority in chunk_lower:
                role_relevance_score += 0.2
        
        # Score combiné (original + pertinence rôle)
        combined_score = original_score + (role_relevance_score * role_confidence)
        
        filtered_results.append((chunk_text, source, combined_score, role_relevance_score))
    
    # Trier par score combiné
    filtered_results = sorted(filtered_results, key=lambda x: x[2], reverse=True)
    
    # Reconvertir au format original en gardant le score combiné
    return [(text, source, combined_score) for text, source, combined_score, _ in filtered_results]

def expand_acronyms_in_query(query):
    """Étend les acronymes dans la requête avec une meilleure détection."""
    expanded_query = query
    for acronym, expansion in ACRONYMS.items():
        # Détection plus précise avec une expression régulière
        pattern = r'\b' + re.escape(acronym) + r'\b'
        if re.search(pattern, query, re.IGNORECASE):
            expanded_query = re.sub(pattern, f"{acronym} ({expansion})", expanded_query, flags=re.IGNORECASE)
            logger.info(f"Acronyme détecté et étendu: {acronym} -> {expansion}")
    
    # Si des acronymes ont été étendus, on ajoute une note
    if expanded_query != query:
        expanded_query += " " + " ".join([expansion for acronym, expansion in ACRONYMS.items() 
                                         if re.search(r'\b' + re.escape(acronym) + r'\b', query, re.IGNORECASE)])
    
    return expanded_query

def convert_markdown_to_html(text):
    """
    Conversion améliorée du Markdown vers HTML avec prise en charge de plus de formats.
    """
    # Gestion des titres (#, ##, etc.)
    def replace_heading(match):
        hashes = match.group(1)
        level = len(hashes)
        title = match.group(2).strip()
        return f"<h{level}>{title}</h{level}>"
    
    text = re.sub(r'^(#{1,6})\s+(.*)$', replace_heading, text, flags=re.MULTILINE)
    
    # Conversion des formats gras et italique
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.*?)\*', r'<em>\1</em>', text)
    text = re.sub(r'__(.*?)__', r'<strong>\1</strong>', text)  # Alternative pour le gras
    text = re.sub(r'_(.*?)_', r'<em>\1</em>', text)  # Alternative pour l'italique
    
    # Gestion des liens [texte](url)
    text = re.sub(r'\[(.*?)\]\((.*?)\)', r'<a href="\2" target="_blank">\1</a>', text)
    
    # Traitement des listes à puces et numérotées
    lines = text.splitlines()
    html_lines = []
    in_ul = False
    in_ol = False
    
    for line in lines:
        # Liste à puces
        if line.strip().startswith("* ") or line.strip().startswith("- "):
            if not in_ul:
                if in_ol:
                    html_lines.append("</ol>")
                    in_ol = False
                html_lines.append("<ul>")
                in_ul = True
            content = re.sub(r'^\s*[\*\-]\s+(.*)', r'\1', line)
            html_lines.append(f"<li>{content}</li>")
        
        # Liste numérotée
        elif re.match(r'^\s*\d+\.\s+', line):
            if not in_ol:
                if in_ul:
                    html_lines.append("</ul>")
                    in_ul = False
                html_lines.append("<ol>")
                in_ol = True
            content = re.sub(r'^\s*\d+\.\s+(.*)', r'\1', line)
            html_lines.append(f"<li>{content}</li>")
        
        # Ligne normale
        else:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            html_lines.append(line)
    
    # Fermer les listes si nécessaire
    if in_ul:
        html_lines.append("</ul>")
    if in_ol:
        html_lines.append("</ol>")
    
    text = "\n".join(html_lines)
    
    # Découpage en paragraphes pour les blocs de texte non déjà formatés
    paragraphs = []
    for block in re.split(r'\n\s*\n', text):
        block = block.strip()
        # Vérifier si le bloc contient déjà des balises HTML
        if not re.match(r'^<\/?(h\d|ul|ol|li|blockquote|pre|table)', block):
            if block:  # Ne pas ajouter de paragraphe vide
                block = f"<p>{block}</p>"
        paragraphs.append(block)
    
    return "\n".join(paragraphs)

def generate_basic_answer(query, context, found_info):
    """Fallback basique en l'absence du client GROQ."""
    answer = "<p>Désolé, le service de génération de réponse n'est pas disponible actuellement. Veuillez réessayer plus tard ou contacter l'administrateur.</p>"
    return answer

# ===== GESTION DES DONNÉES =====
def create_default_data():
    """Crée des fichiers de données par défaut si DATA_DIR est vide."""
    if not os.listdir(DATA_DIR):
        # Création d'un fichier d'acronymes
        with open(os.path.join(DATA_DIR, "acronymes.txt"), "w", encoding="utf-8") as f:
            f.write("# Acronymes utilisés à SeaTech\n\n")
            for acronym, meaning in ACRONYMS.items():
                f.write(f"{acronym}: {meaning}\n")
        # Fichier de contacts (seulement si CONTACTS est renseigné)
        with open(os.path.join(DATA_DIR, "contacts.txt"), "w", encoding="utf-8") as f:
            f.write("# Contacts importants à SeaTech\n\n")
            if CONTACTS:
                for name, info in CONTACTS.items():
                    f.write(f"{name}: {info.get('role', 'N/A')} - {info.get('email', 'N/A')}\n")
            else:
                f.write("Aucun contact défini.\n")
        # Fichier d'information générale
        with open(os.path.join(DATA_DIR, "info_generale.json"), "w", encoding="utf-8") as f:
            f.write("# SeaTech - École d'ingénieurs\n\n")
            f.write("SeaTech est une école d'ingénieurs de l'Université de Toulon (UTLN). ")
            f.write("Elle propose plusieurs formations d'ingénieur dont les spécialités SN (Systèmes Numériques) ")
            f.write("et MTX (Matériaux). Les formations peuvent être suivies en statut étudiant (FISE) ou en apprentissage (FISA).\n")

def load_data():
    """Charge toujours le corpus et le découpe en chunks.

    Le cache d'embeddings est validé séparément dans compute_embeddings().
    """
    create_default_data()
    chunks_with_sources = []
    valid_extensions = (".txt", ".md", ".html", ".csv", ".json")

    for filename in sorted(os.listdir(DATA_DIR)):
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.isfile(file_path) or not filename.endswith(valid_extensions):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            raw_chunks = [
                chunk.strip()
                for chunk in re.split(r"\n\s*\n", content)
                if chunk.strip()
            ]
            chunks_with_sources.extend((chunk, filename) for chunk in raw_chunks)
            logger.info(f"Fichier {filename} chargé : {len(raw_chunks)} chunks")
        except Exception as e:
            logger.error(f"Erreur chargement {filename}: {e}")

    logger.info(f"Corpus chargé : {len(chunks_with_sources)} chunks")
    return chunks_with_sources

def get_albert_embeddings(texts, batch_size=32):
    """Vectorise une chaîne ou une liste de chaînes avec Albert API."""
    if albert_client is None:
        raise RuntimeError("Client Albert indisponible pour les embeddings.")
    if not ALBERT_EMBEDDING_MODEL:
        raise RuntimeError("ALBERT_EMBEDDING_MODEL n'est pas configuré.")
    single_input = isinstance(texts, str)
    items = [texts] if single_input else list(texts)
    if not items:
        return np.empty((0, 0), dtype=np.float32)
    vectors = []
    for offset in range(0, len(items), batch_size):
        response = albert_client.embeddings.create(
            model=ALBERT_EMBEDDING_MODEL,
            input=items[offset:offset + batch_size],
            encoding_format="float"
        )
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
    array = np.asarray(vectors, dtype=np.float32)
    array /= np.clip(np.linalg.norm(array, axis=1, keepdims=True), 1e-12, None)
    return array[0] if single_input else array


def compute_embeddings(chunks_with_sources):
    """Calcule ou charge les embeddings Albert pour les chunks."""
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CHUNKS_CACHE):
        try:
            with open(EMBEDDINGS_CACHE, "rb") as f:
                cached_embeddings = pickle.load(f)
            with open(CHUNKS_CACHE, "rb") as f:
                cached_chunks = pickle.load(f)
            if (
                chunks_with_sources
                and len(cached_embeddings) == len(cached_chunks) == len(chunks_with_sources)
                and cached_chunks == chunks_with_sources
            ):
                logger.info(f"Cache Albert chargé : {len(cached_embeddings)} embeddings")
                return np.asarray(cached_embeddings, dtype=np.float32), cached_chunks
            logger.info("Cache Albert vide ou obsolète : recalcul nécessaire")
        except Exception as e:
            logger.error(f"Erreur chargement cache Albert: {e}")
    try:
        chunk_embeddings = get_albert_embeddings([chunk[0] for chunk in chunks_with_sources])
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump(chunk_embeddings, f)
        with open(CHUNKS_CACHE, "wb") as f:
            pickle.dump(chunks_with_sources, f)
        logger.info(f"Embeddings Albert calculés et sauvegardés : {len(chunk_embeddings)}")
        return chunk_embeddings, chunks_with_sources
    except Exception as e:
        logger.error(f"Erreur calcul embeddings Albert: {e}")
        return np.empty((0, 0), dtype=np.float32), chunks_with_sources


def setup_search_index(embeddings):
    """Prépare une matrice NumPy normalisée, sans FAISS."""
    if embeddings is None or embeddings.size == 0:
        logger.warning("Index NumPy indisponible, recherche par mots-clés activée")
        return None, False
    matrix = np.asarray(embeddings, dtype=np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)
    logger.info(f"Index NumPy créé avec {len(matrix)} vecteurs")
    return matrix, False


def search_similar_chunks_with_confirmed_role(query, index, is_faiss, embeddings, chunks_data, conversation_history=None, confirmed_role=None, top_n=5):
    """Recherche sémantique légère avec Albert embeddings et NumPy."""
    if index is None or not chunks_data:
        results = keyword_search(query, chunks_data, top_n)
    else:
        try:
            query_embedding = get_albert_embeddings(expand_acronyms_in_query(query))
            similarities = np.asarray(index, dtype=np.float32) @ query_embedding
            limit = min(top_n * 2, len(chunks_data))
            top_indices = np.argsort(similarities)[-limit:][::-1]
            results = [(chunks_data[i][0], chunks_data[i][1], float(similarities[i])) for i in top_indices if similarities[i] > 0.1]
        except Exception as e:
            logger.error(f"Erreur recherche Albert/NumPy: {e}")
            results = keyword_search(query, chunks_data, top_n)
    if confirmed_role:
        detected_role, role_confidence = confirmed_role, 1.0
    else:
        detected_role, role_confidence = detect_user_role(query, conversation_history)
    return filter_chunks_by_role(results, detected_role, role_confidence)[:top_n], detected_role, role_confidence

# 3. Fonction pour vérifier si l'utilisateur a confirmé son rôle
def is_role_confirmed(session_id):
    """Vérifie si l'utilisateur a confirmé son rôle."""
    confirmed = session_id in user_profiles_global and user_profiles_global[session_id].get('confirmed', False)
    logger.info(f"Vérification rôle pour session {session_id}: {confirmed}, profils={user_profiles_global}")
    return confirmed

def get_confirmed_role(session_id):
    """Récupère le rôle confirmé de l'utilisateur."""
    if session_id in user_profiles_global:
        return user_profiles_global[session_id].get('role', None)
    return None
def keyword_search(query, chunks_data, top_n=5):
    """Recherche par mots-clés en fallback."""
    query_terms = query.lower().split()
    results = []
    # Recherche d'acronymes
    for acronym in ACRONYMS:
        if acronym.lower() in query.lower():
            for chunk, source in chunks_data:
                if acronym in chunk:
                    results.append((chunk, source, 0.85))
    for chunk, source in chunks_data:
        score = sum(0.1 for term in query_terms if term in chunk.lower())
        if score > 0:
            results.append((chunk, source, score))
    if not results:
        results.append(("Aucune information spécifique trouvée.", "fallback.txt", 0.1))
    return sorted(results, key=lambda x: x[2], reverse=True)[:top_n]

# ===== FORMATAGE DES SOURCES ET LOGS =====
def format_sources(results, for_freddy=False, detected_role=None):
    """Formate les résultats en HTML pour l'affichage des sources."""
    html = '<div class="sources-container">'
    
    if detected_role:
        role_info = profile_mapping.get(detected_role, {})
        html += f'<div class="role-detection-info"><strong>Profil détecté:</strong> {detected_role}</div>'
    
    for text, source, score in results:
        relevance_class = "high-relevance" if score > 0.8 else "medium-relevance" if score > 0.6 else "low-relevance"
        if for_freddy:
            preview = text[:200] + ("..." if len(text) > 200 else "")
            html += f'''
            <div class="freddy-source-block" onclick="this.classList.toggle('expanded')">
                <div class="freddy-source-header">
                    <span class="source-name">{source}</span>
                    <span class="{relevance_class}">{score:.2f}</span>
                </div>
                <div class="freddy-source-content">{preview}</div>
            </div>
            '''
        else:
            html += f'''
            <div class="source-block">
                <div class="source-header">
                    <span class="source-name">{source}</span>
                    <span class="relevance-score">{score:.2f}</span>
                </div>
                <div class="source-content">{text}</div>
            </div>
            '''
    html += '</div>'
    return html

def create_freddy_logs(query, results, detected_role=None, role_confidence=None):
    """Crée des logs HTML détaillés pour le module Freddy."""
    current_time = datetime.now().strftime("%H:%M:%S")
    high_relevance = sum(1 for _, _, score in results if score > 0.8)
    medium_relevance = sum(1 for _, _, score in results if 0.6 < score <= 0.8)
    low_relevance = sum(1 for _, _, score in results if score <= 0.6)
    
    html = f'''
    <div class="freddy-logs">
        <div class="freddy-log-entry">
            <span class="log-time">{current_time}</span>
            <span class="log-action">Analyse de la question</span>
        </div>
        <div class="freddy-log-entry">
            <span class="log-detail">Recherche pour : <strong>"{query}"</strong></span>
        </div>
    '''
    
    if detected_role and role_confidence:
        html += f'''
        <div class="freddy-log-entry">
            <span class="log-time">{current_time}</span>
            <span class="log-action">Détection de profil</span>
        </div>
        <div class="freddy-log-entry">
            <span class="log-detail">Profil détecté : <strong>{detected_role}</strong> (confiance: {role_confidence:.2f})</span>
        </div>
        '''
    
    html += f'''
        <div class="freddy-log-entry">
            <span class="log-time">{current_time}</span>
            <span class="log-action">Résultats filtrés par profil</span>
        </div>
        <div class="freddy-log-entry">
            <span class="log-detail">Détails : 
                <span class="high-relevance">{high_relevance} très pertinents</span>, 
                <span class="medium-relevance">{medium_relevance} pertinents</span>, 
                <span class="low-relevance">{low_relevance} peu pertinents</span>
            </span>
        </div>
    </div>
    '''
    return html

def call_groq(messages):
    """Envoie une requête au modèle Groq actuellement configuré."""
    if groq_client is None:
        raise RuntimeError("Client Groq indisponible.")

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.95,
        max_tokens=28500,
        timeout=40
    )

    return response.choices[0].message.content


def call_albert(messages):
    """Envoie une requête à Albert API avec des limites adaptées au RAG."""
    if albert_client is None:
        raise RuntimeError("Client Albert indisponible.")

    if not ALBERT_MODEL:
        raise RuntimeError("ALBERT_MODEL n'est pas configuré.")

    response = albert_client.chat.completions.create(
        model=ALBERT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1200,
        stream=False,
        timeout=40
    )

    return response.choices[0].message.content

def call_llm(messages):
    """
    Sélectionne le fournisseur LLM configuré.

    Si Albert est sélectionné mais indisponible, un retour vers Groq
    est tenté automatiquement.
    """
    if LLM_PROVIDER == "albert":
        try:
            logger.info("Appel du fournisseur Albert")
            return call_albert(messages)
        except Exception as albert_error:
            logger.error(
                f"Échec Albert, tentative de retour vers Groq : "
                f"{type(albert_error).__name__}: {albert_error}"
            )

            if groq_client is not None:
                return call_groq(messages)

            raise RuntimeError(
                "Albert a échoué et aucun client Groq de secours "
                "n'est disponible."
            ) from albert_error

    if LLM_PROVIDER == "groq":
        logger.info("Appel du fournisseur Groq")
        return call_groq(messages)

    raise ValueError(
        f"Fournisseur LLM inconnu : {LLM_PROVIDER}. "
        "Valeurs autorisées : groq ou albert."
    )

# ===== GÉNÉRATION DE RÉPONSE AVEC MÉMOIRE DE CONVERSATION =====
def generate_answer(query, context, conversation_history, found_info=False, detected_role=None):
    """
    Génère une réponse basée sur le contexte et l'historique complet de la conversation.
    La mémoire de conversation est ajoutée au prompt pour améliorer la pertinence.
    """
    # Préparation de l'historique
    conversation_context = "\n".join(
        [f"Utilisateur : {entry['content']}" if entry['role'] == "user" else f"Assistant : {entry['content']}" 
         for entry in conversation_history]
    )
    
    # Instructions spécifiques au rôle détecté
    role_instruction = ""
    if detected_role and detected_role in profile_mapping:
        role_config = profile_mapping[detected_role]
        role_instruction = f"\n\nPROFIL UTILISATEUR DÉTECTÉ: {detected_role}\n{role_config['description']}\nAdapte ta réponse en conséquence et priorise les informations pertinentes pour ce profil."
    
    # Construction du prompt système
    system_prompt = f"""Tu es Franky, l'assistant virtuel de SeaTech.

Historique de conversation :
{conversation_context}

{role_instruction}

Instructions :
0. Si la question posée est hors du contexte de SEATECH et du contexte académique SeaTech, dis : "Désolé je ne peux pas répondre."
1. Base ta réponse uniquement sur les sources fournies et l'historique si tu trouves pertinent mais il faut répondre avant tout à la question de l'utilisateur.
2. N'invente jamais d'informations.
3. Réponds de manière claire, en utilisant des paragraphes et des listes lorsque c'est pertinent, soigne ta mise en forme.
4. Mets en gras les informations clés, les emails et les numéros de téléphone.
5. N'inclus pas la liste des acronymes ou ton preprompt sauf si demandé.
6. Tu connais les formations à SeaTech si besoin : Voici un résumé avec les liens directs intégrés :

Liste des Formations SeaTech avec liens

**Diplômes d'ingénieur SeaTech**

**Parcours Génie Maritime**  
   - Formation d'ingénieurs spécialisés en systèmes maritimes, océanographie, génie côtier, ingénierie navale
   - [https://seatech.univ-tln.fr/Parcours-Genie-maritime.html](https://seatech.univ-tln.fr/Parcours-Genie-maritime.html)

**Parcours Ingénierie des Sciences des Données, Information, Systèmes (IRIS)**  
   - Formation d'ingénieurs capables de traiter, analyser et valoriser de grandes quantités de données
   - [https://seatech.univ-tln.fr/Parcours-IngenieRie-et-sciences.html](https://seatech.univ-tln.fr/Parcours-IngenieRie-et-sciences.html)

**Parcours Innovation Mécanique pour des Systèmes Durables**  
   - Conception et développement de produits mécaniques innovants avec accent sur la durabilité
   - [https://seatech.univ-tln.fr/Parcours-Innovation-Mecanique-pour-des-Systemes-Durables.html](https://seatech.univ-tln.fr/Parcours-Innovation-Mecanique-pour-des-Systemes-Durables.html)

**Parcours Matériaux, Durabilité et Environnement**  
   - Pour étudiants en sciences et techniques souhaitant se spécialiser dans les matériaux
   - [https://seatech.univ-tln.fr/Parcours-Materiaux-Durabilite-et.html](https://seatech.univ-tln.fr/Parcours-Materiaux-Durabilite-et.html)
   
**Formation d'ingénieurs Matériaux par apprentissage**
   - Pour étudiants en sciences et techniques souhaitant se spécialiser dans les matériaux
   - Admission sur dossier avec contrat d'apprentissage
   - **RNCP : 39062 - Certificateur : Université de Toulon**
   - [https://seatech.univ-tln.fr/Formation-d-ingenieurs-Materiaux-par-apprentissage.html](https://seatech.univ-tln.fr/Formation-d-ingenieurs-Materiaux-par-apprentissage.html)

**Parcours Modélisation et Calculs Fluides et Structures**  
   - Simulation numérique et modélisation des comportements physiques
   - [https://seatech.univ-tln.fr/Parcours-Modelisation-et-Calculs.html](https://seatech.univ-tln.fr/Parcours-Modelisation-et-Calculs.html)

**Parcours Systèmes Mécatroniques et Robotiques**  
   - Conception et développement de systèmes intégrant mécanique, électronique et informatique
   - [https://seatech.univ-tln.fr/Parcours-Systemes-mecatroniques-et.html](https://seatech.univ-tln.fr/Parcours-Systemes-mecatroniques-et.html)

**Formation d'ingénieurs en Systèmes Numériques par apprentissage**  
   - Pour étudiants en sciences et techniques
   - Admission sur dossier avec contrat d'apprentissage
   - **RNCP : 37901 - Certificateur : Université de Toulon**
   - [https://seatech.univ-tln.fr/Formation-d-ingenieurs-en-systemes-numeriques-par-apprentissage.html](https://seatech.univ-tln.fr/Formation-d-ingenieurs-en-systemes-numeriques-par-apprentissage.html)

**Informations générales**

- **Formations complètes** : [https://seatech.univ-tln.fr/Formations.html](https://seatech.univ-tln.fr/Formations.html)
- **Déroulement des études** : [https://seatech.univ-tln.fr/Deroulement-des-etudes.html](https://seatech.univ-tln.fr/Deroulement-des-etudes.html)
- **Admissions** : [https://seatech.univ-tln.fr/admission.html](https://seatech.univ-tln.fr/admission.html)
- **Doubles diplômes** : [https://seatech.univ-tln.fr/doubles-diplomes.html](https://seatech.univ-tln.fr/doubles-diplomes.html)
- **Page d'accueil** : [https://seatech.univ-tln.fr/](https://seatech.univ-tln.fr/)

Données d'aide pour répondre :
{context}
"""
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        answer = call_llm(messages)
        answer = convert_markdown_to_html(answer)

        # Détection d'hallucinations simples
        if any(x in answer.lower() for x in ["@freddy", "freddy@"]):
            logger.warning("Hallucination détectée - correction appliquée")
            answer = "<p>⚠️ Je ne peux pas inventer de contacts inexistants.</p>"

        return answer
    except Exception as e:
        logger.error(f"Erreur génération réponse: {e}")
        # Message d'erreur plus informatif
        error_message = "<p>Désolé, une erreur est survenue lors de la génération de la réponse. Détails techniques:</p>"
        error_message += f"<p><code>{str(e)[:100]}...</code></p>"
        error_message += "<p>Veuillez réessayer avec une question différente ou plus tard.</p>"
        return error_message

# ===== INITIALISATION DES DONNÉES =====
chunks_with_sources = load_data()
chunk_embeddings, chunks_with_sources = compute_embeddings(chunks_with_sources)
search_index, use_faiss = setup_search_index(chunk_embeddings)

# ===== APPLICATION FLASK =====
app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'development-only-secret-key')
app.config['SESSION_COOKIE_SECURE'] = True   # HTTPS requis sur HuggingFace Spaces
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'None'  # Nécessaire pour l'iframe HF
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 heure
conversation_history_global = {}
user_profiles_global = {}  # Stocker les profils utilisateur plutôt que dans la session Flask
# ajout Daly

# Charger modèle Vosk une seule fois au démarrage
# vosk_model = vosk.Model("models/vosk-model-fr-0.6-linto-2.2.0") 



if ENABLE_VOSK:
    try:
        vosk_model = vosk.Model("models/vosk-model-small-fr-0.22")
        logger.info("Modèle Vosk chargé")
    except Exception as e:
        logger.error(f"Impossible de charger le modèle Vosk : {e}")
        vosk_model = None
else:
    vosk_model = None
    logger.info("Vosk désactivé")

#fin ajout
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "chatbot-seatech"})


@app.route("/", methods=["GET", "POST"])
def index():
    """Page d'accueil du chatbot avec gestion de la sélection de rôle."""
    session.permanent = True
    
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session.modified = True
    session_id = session['session_id']
    
    logger.info(f"Route / appelée - Session ID: {session_id}")
    logger.info(f"Session complète: {dict(session)}")
    logger.info(f"User profile dans session: {user_profiles_global}")
    
    if session_id not in conversation_history_global:
        conversation_history_global[session_id] = []
    
    # Gérer la sélection de rôle si fournie
    selected_role = request.form.get("role_selection") if request.method == "POST" else None
    if selected_role:
        handle_role_selection(session_id, selected_role)
    
    if request.method == "POST":
        user_query = request.form.get("query", "").strip()
        if user_query:
            # Vérifier si le rôle est confirmé
            if not is_role_confirmed(session_id):
                # Rediriger vers la sélection de rôle
                error_message = {
                    "role": "assistant",
                    "content": "<p>Veuillez d'abord sélectionner votre profil ci-dessus pour que je puisse mieux vous aider.</p>",
                    "is_system_message": True
                }
                conversation_history_global[session_id].append(error_message)
            else:
                conversation_history_global[session_id].append({"role": "user", "content": user_query})
                start_time = time.time()
                
                confirmed_role = get_confirmed_role(session_id)
                
                # Recherche avec rôle confirmé
                results, detected_role, role_confidence = search_similar_chunks_with_confirmed_role(
                    user_query, search_index, use_faiss, chunk_embeddings, chunks_with_sources, 
                    conversation_history_global[session_id], confirmed_role
                )
                
                # Contexte des chunks trouvés
                context_chunks = "\n\n".join([text for text, _, _ in results])
                found_info = any(score > CONFIDENCE_THRESHOLD for _, _, score in results)
                
                # Génération de réponse avec le rôle confirmé
                answer = generate_answer(user_query, context_chunks, conversation_history_global[session_id], found_info, detected_role)
                processing_time = time.time() - start_time
                
                # Formatage des sources avec information de rôle
                sources_html = format_sources(results, detected_role=detected_role)
                freddy_html = create_freddy_logs(user_query, results, detected_role, role_confidence) + format_sources(results, for_freddy=True, detected_role=detected_role)
                
                conversation_history_global[session_id].append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources_html,
                    "freddy_logs": freddy_html,
                    "processing_time": f"{processing_time:.2f}s",
                    "detected_role": detected_role,
                    "role_confidence": role_confidence
                })
    
    conv = conversation_history_global.get(session_id, [])
    current_datetime = datetime.now()
    user_profile = user_profiles_global.get(session_id, {})
    
    logger.info(f"Route / render_template - session_id: {session_id}, user_profile: {user_profile}")
    
    return render_template("index.html", 
                         conversation=conv, 
                         query="", 
                         current_datetime=current_datetime,
                         user_profile=user_profile,
                         profile_mapping=profile_mapping)


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """Endpoint API pour la recherche avec gestion de rôle."""
    session.permanent = True
    start_time = time.time()

    try:
        data = request.get_json()
        print("DATA RECU:", data)
        print("SESSION:", session)
        user_query = data.get("query", "").strip()
        role_selection = data.get("role_selection", None)
        
        # AJOUT  Daly
        if data.get("mode") == "conversation":
            prompt = "Réponds de façon brève, 1 à 2 phrases maximum, pour être lue à voix haute."
        else:
            prompt = "Réponse normale"
        #  Fin ajout
        
        if not user_query and not role_selection:
            return jsonify({"error": "Question vide"}), 400
        
        # Gestion de la session
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
            session.modified = True
        session_id = session['session_id']
        if session_id not in conversation_history_global:
            conversation_history_global[session_id] = []
        
        # Gérer la sélection de rôle
        if role_selection:
            logger.info(f"Tentative de sélection de rôle: {role_selection} pour session {session_id}")
            result = handle_role_selection(session_id, role_selection)
            logger.info(f"Résultat handle_role_selection: {result}")
            logger.info(f"Session après sélection: {dict(session)}")
            logger.info(f"Session user_profile: {session.get('user_profile')}")
            return jsonify({
                "response": f"<p>Rôle <strong>{role_selection}</strong> sélectionné avec succès ! Vous pouvez maintenant poser vos questions.</p>",
                "role_confirmed": True,
                "selected_role": role_selection,
                "status": "role_selected"
            })
        
        # Vérifier si le rôle est confirmé
        if not is_role_confirmed(session_id):
            return jsonify({
                "response": "<p>Veuillez d'abord sélectionner votre profil pour que je puisse mieux vous aider.</p>",
                "role_confirmed": False,
                "status": "role_required"
            })
            
        # Ajout à l'historique de conversation
        conversation_history_global[session_id].append({"role": "user", "content": user_query})
        
        confirmed_role = get_confirmed_role(session_id)
        
        # Recherche d'informations avec rôle confirmé
        try:
            results, detected_role, role_confidence = search_similar_chunks_with_confirmed_role(
                user_query, search_index, use_faiss, chunk_embeddings, chunks_with_sources,
                conversation_history_global[session_id], confirmed_role
            )
            context_chunks = "\n\n".join([text for text, _, _ in results])
            found_info = any(score > CONFIDENCE_THRESHOLD for _, _, score in results)
        except Exception as search_error:
            logger.error(f"Erreur de recherche: {search_error}")
            results = [("Une erreur s'est produite lors de la recherche.", "error.txt", 0.1)]
            context_chunks = "Informations non disponibles en raison d'une erreur."
            found_info = False
            detected_role = confirmed_role or "Candidat"
            role_confidence = 1.0 if confirmed_role else 0.1
        
        # Génération de réponse avec le rôle confirmé
        answer = generate_answer(user_query, context_chunks, conversation_history_global[session_id], found_info, detected_role)
        
        # Formatage des sources et logs avec information de rôle
        sources_html = format_sources(results, detected_role=detected_role)
        freddy_html = create_freddy_logs(user_query, results, detected_role, role_confidence) + format_sources(results, for_freddy=True, detected_role=detected_role)
        
        # Calcul du temps de traitement
        processing_time = time.time() - start_time
        
        # Mise à jour de l'historique de conversation
        conversation_history_global[session_id].append({
            "role": "assistant", 
            "content": answer,
            "sources": sources_html,
            "freddy_logs": freddy_html,
            "processing_time": f"{processing_time:.2f}s",
            "detected_role": detected_role,
            "role_confidence": role_confidence
        })
        
        # Réponse de l'API
        return jsonify({
            "response": answer,
            "freddy_logs": freddy_html,
            "sources": sources_html,
            "sources_found": found_info,
            "processing_time": f"{processing_time:.2f}s",
            "detected_role": detected_role,
            "confirmed_role": confirmed_role,
            "role_confidence": f"{role_confidence:.2f}",
            "role_confirmed": True,
            "status": "success"
        })
        
    except Exception as e:
        logger.error(f"Erreur API globale: {e}")
        processing_time = time.time() - start_time
        
        return jsonify({
            "response": f"<p>Désolé, une erreur s'est produite: {str(e)[:50]}...</p><p>Veuillez réessayer.</p>",
            "freddy_logs": f"<div class='freddy-logs'><div class='freddy-log-entry'><span class='log-action'>Erreur</span><span class='log-detail'>{str(e)[:100]}...</span></div></div>",
            "sources_found": False,
            "processing_time": f"{processing_time:.2f}s",
            "detected_role": "Candidat",
            "role_confidence": "0.00",
            "status": "error",
            "error_type": str(type(e).__name__)
        }), 500

@app.route("/api/role-info", methods=["GET"])
def api_role_info():
    """Endpoint pour récupérer les informations sur les rôles disponibles."""
    return jsonify({
        "roles": {role: config["description"] for role, config in profile_mapping.items()},
        "status": "success"
    })
@app.route("/api/reset-role", methods=["POST"])
def api_reset_role():
    """Permet de réinitialiser le rôle sélectionné."""
    session.permanent = True
    
    if 'session_id' in session:
        session_id = session['session_id']
        # Supprimer le profil du dictionnaire global
        if session_id in user_profiles_global:
            del user_profiles_global[session_id]
        # Vider l'historique de conversation
        if session_id in conversation_history_global:
            conversation_history_global[session_id] = []
        logger.info(f"Rôle réinitialisé pour la session {session_id}")
    
    return jsonify({
        "response": "<p>Rôle réinitialisé. Veuillez sélectionner votre nouveau profil.</p>",
        "role_confirmed": False,
        "status": "role_reset"
    })
@app.route('/static/<path:path>')
def send_static(path):
    """Fournit les fichiers statiques."""
    return send_from_directory('static', path)

@app.route('/api/ask-audio', methods=['POST'])
def ask_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "Aucun fichier audio reçu"}), 400

    audio_file = request.files['audio']
    wf = wave.open(io.BytesIO(audio_file.read()), "rb")

    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() not in [8000, 16000]:
        return jsonify({"error": "Format audio non supporté. Utilise WAV PCM mono 16kHz."}), 400

    rec = vosk.KaldiRecognizer(vosk_model, wf.getframerate())
    text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += " " + res.get("text", "")

    final_res = json.loads(rec.FinalResult())
    text += " " + final_res.get("text", "")

    text = text.strip()
    if not text:
        return jsonify({"response": "Désolé, je n’ai pas compris l’audio."})

    # Réutiliser ton pipeline existant
    session_id = request.cookies.get("session_id", "default")
    search_results, freddy_logs = search_similar_chunks_with_confirmed_role(text, session_id)
    answer, sources, processing_time = generate_answer(text, search_results, session_id, user_profile)

    return jsonify({
        "response": answer,
        "sources": sources,
        "freddy_logs": freddy_logs,
        "processing_time": processing_time,
        "query": text
    })

if __name__ == "__main__":
    # Création des répertoires statiques si nécessaires
    if not os.path.exists("static"):
        os.makedirs("static", exist_ok=True)
        os.makedirs("static/css", exist_ok=True)
        os.makedirs("static/img", exist_ok=True)
    templates_dir = os.path.join(BASE_DIR, "templates")
    os.makedirs(templates_dir, exist_ok=True)
    
    # IMPORTANT : Récupérer le port depuis les variables d'environnement
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)  # debug=False en production



    
    # Charger modèle Vosk une fois au démarrage
#vosk_model = vosk.Model("model-fr")

