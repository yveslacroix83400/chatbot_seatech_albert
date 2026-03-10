import os
import re
import json
import pickle
import time
import uuid
import vosk
import io
import wave
import logging
from datetime import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory, session
from dotenv import load_dotenv
load_dotenv() #Charge les données de l'environnement de développement dnas le fichier main.


import urllib.request
import zipfile

model_url = "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip"
model_path = "models/vosk-model-small-fr-0.22"

if not os.path.exists(model_path):
    print("Downloading speech model...")
    os.makedirs("models", exist_ok=True)
    urllib.request.urlretrieve(model_url, "model.zip")

    with zipfile.ZipFile("model.zip", 'r') as zip_ref:
        zip_ref.extractall("models")

    os.remove("model.zip")

# Importations ML et API
try:
    import torch
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_IMPORTS_SUCCESS = True
except ImportError:
    ML_IMPORTS_SUCCESS = False
    

try:
    from groq import Groq
    GROQ_IMPORT_SUCCESS = True
except ImportError:
    GROQ_IMPORT_SUCCESS = False

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
EMBEDDINGS_CACHE = os.path.join(CACHE_DIR, "chunk_embeddings.pkl")
CHUNKS_CACHE = os.path.join(CACHE_DIR, "chunks_with_sources.pkl")
QA_STORAGE = os.path.join(CACHE_DIR, "user_qa_memory.json")

# ===== CONFIGURATION =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

LLM_MODEL = "llama-3.3-70b-versatile"
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

if ML_IMPORTS_SUCCESS:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        #embedding_model = SentenceTransformer('intfloat/e5-large-v2', device='cpu')
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        embedding_model.max_seq_length = 256  # Réduire la longueur max
        logger.info(f"Modèle d'embedding chargé sur {device}")
    except Exception as e:
        logger.error(f"Erreur chargement modèle d'embedding: {e}")
        embedding_model = None
else:
    embedding_model = None
    logger.warning("Bibliothèques ML non disponibles")


def handle_role_selection(session_id, selected_role=None):
    """Gère la sélection de rôle utilisateur et initialise la conversation."""
    if session_id not in conversation_history_global:
        conversation_history_global[session_id] = []
    print(f"Session {session_id} - Rôle sélectionné : {selected_role}")
    # Si un rôle est sélectionné, l'enregistrer dans la session
    if selected_role and selected_role in profile_mapping:
        # Marquer la session comme permanente pour la persister
        session.permanent = True
        
        if 'user_profile' not in session:
            session['user_profile'] = {}
        session['user_profile']['role'] = selected_role
        session['user_profile']['confirmed'] = True
        
        # Force Flask à sauvegarder la session
        session.modified = True
        
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
    """Charge les fichiers dans DATA_DIR et découpe le contenu en chunks avec timestamp de dernière modification."""
    create_default_data()
    chunks_with_sources = []
    valid_extensions = ['.txt', '.md', '.html', '.csv','.json']
    
    # Charger un cache des timestamps si disponible
    timestamp_cache_path = os.path.join(CACHE_DIR, "file_timestamps.json")
    file_timestamps = {}
    reload_all = False
    
    if os.path.exists(timestamp_cache_path):
        try:
            with open(timestamp_cache_path, "r", encoding="utf-8") as f:
                file_timestamps = json.load(f)
        except Exception as e:
            logger.error(f"Erreur chargement cache timestamps: {e}")
            reload_all = True
    
    # Vérifier et charger les fichiers
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in valid_extensions):
            # Vérifier si le fichier a été modifié
            mtime = os.path.getmtime(file_path)
            
            if reload_all or filename not in file_timestamps or file_timestamps[filename] < mtime:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    # Découpage en chunks selon les paragraphes
                    raw_chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", content) if chunk.strip()]
                    for chunk in raw_chunks:
                        chunks_with_sources.append((chunk, filename))
                    logger.info(f"Fichier {filename} chargé : {len(raw_chunks)} chunks")
                    file_timestamps[filename] = mtime
                except Exception as e:
                    logger.error(f"Erreur chargement {filename}: {e}")
    
    # Sauvegarder les timestamps mis à jour
    try:
        with open(timestamp_cache_path, "w", encoding="utf-8") as f:
            json.dump(file_timestamps, f)
    except Exception as e:
        logger.error(f"Erreur sauvegarde cache timestamps: {e}")
    
    return chunks_with_sources

def compute_embeddings(chunks_with_sources):
    """Calcule ou charge les embeddings pour les chunks."""
    if not embedding_model:
        logger.warning("Modèle d'embedding non disponible")
        dummy_embeddings = np.zeros((len(chunks_with_sources), 384))
        return dummy_embeddings, chunks_with_sources
    if os.path.exists(EMBEDDINGS_CACHE) and os.path.exists(CHUNKS_CACHE):
        try:
            with open(EMBEDDINGS_CACHE, "rb") as f:
                cached_embeddings = pickle.load(f)
            with open(CHUNKS_CACHE, "rb") as f:
                cached_chunks = pickle.load(f)
            if len(cached_embeddings) == len(cached_chunks):
                logger.info(f"Cache chargé : {len(cached_embeddings)} embeddings")
                return cached_embeddings, cached_chunks
        except Exception as e:
            logger.error(f"Erreur chargement cache: {e}")
    try:
        chunk_texts = [chunk[0] for chunk in chunks_with_sources]
        embeddings = embedding_model.encode(chunk_texts, batch_size=8, convert_to_tensor=True)
        chunk_embeddings = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else np.array(embeddings)
        with open(EMBEDDINGS_CACHE, "wb") as f:
            pickle.dump(chunk_embeddings, f)
        with open(CHUNKS_CACHE, "wb") as f:
            pickle.dump(chunks_with_sources, f)
        logger.info(f"Embeddings calculés et sauvegardés : {len(chunk_embeddings)}")
        return chunk_embeddings, chunks_with_sources
    except Exception as e:
        logger.error(f"Erreur calcul embeddings: {e}")
        dummy_embeddings = np.zeros((len(chunks_with_sources), 384))
        return dummy_embeddings, chunks_with_sources

# ===== INDEXATION AVEC FAISS =====
def setup_search_index(embeddings):
    """
    Configure l'index FAISS.
    Utilise IndexIVFFlat pour un grand nombre d'embeddings, sinon IndexFlatL2.
    """
    if not ML_IMPORTS_SUCCESS or embeddings.size == 0:
        logger.warning("Index de recherche non disponible")
        return None, False
    try:
        d = embeddings.shape[1]
        if embeddings.shape[0] > 1000:
            quantizer = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(quantizer, d, 100, faiss.METRIC_L2)
            index.train(embeddings)
            index.add(embeddings)
            logger.info(f"Index FAISS IVFFlat créé avec {index.ntotal} vecteurs")
            return index, True
        else:
            index = faiss.IndexFlatL2(d)
            index.add(embeddings)
            logger.info(f"Index FAISS FlatL2 créé avec {index.ntotal} vecteurs")
            return index, True
    except Exception as e:
        logger.error(f"Erreur initialisation index: {e}")
        return None, False

def search_similar_chunks_with_confirmed_role(query, index, is_faiss, embeddings, chunks_data, conversation_history=None, confirmed_role=None, top_n=5):
    """Recherche avec rôle confirmé par l'utilisateur."""
    
    # Recherche de base (identique à l'ancienne fonction)
    if not embedding_model or not index:
        results = keyword_search(query, chunks_data, top_n)
    else:
        try:
            query_expanded = expand_acronyms_in_query(query)
            query_embedding = embedding_model.encode(query_expanded, convert_to_tensor=False)
            query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            
            if is_faiss:
                distances, indices = index.search(query_embedding, min(top_n * 2, len(chunks_data)))
                distances, indices = distances[0], indices[0]
                results = []
                for i, idx in enumerate(indices):
                    if 0 <= idx < len(chunks_data):
                        score = 1 / (1 + distances[i])
                        results.append((chunks_data[idx][0], chunks_data[idx][1], score))
            else:
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                top_indices = np.argsort(similarities)[-top_n*2:][::-1]
                results = [(chunks_data[idx][0], chunks_data[idx][1], similarities[idx])
                           for idx in top_indices if similarities[idx] > 0.1]
            
            results = sorted(results, key=lambda x: x[2], reverse=True)[:top_n*2]
        except Exception as e:
            logger.error(f"Erreur de recherche: {e}")
            results = keyword_search(query, chunks_data, top_n)
    
    # Utiliser le rôle confirmé ou détecter automatiquement
    if confirmed_role:
        detected_role = confirmed_role
        role_confidence = 1.0  # Confiance maximale car confirmé par l'utilisateur
    else:
        detected_role, role_confidence = detect_user_role(query, conversation_history)
    
    # Filtrer et réorganiser les résultats selon le rôle
    filtered_results = filter_chunks_by_role(results, detected_role, role_confidence)
    
    return filtered_results[:top_n], detected_role, role_confidence
# 3. Fonction pour vérifier si l'utilisateur a confirmé son rôle
def is_role_confirmed(session_id):
    """Vérifie si l'utilisateur a confirmé son rôle."""
    user_profile = session.get('user_profile', {})
    confirmed = user_profile.get('confirmed', False)
    logger.info(f"Vérification rôle pour session {session_id}: user_profile={user_profile}, confirmed={confirmed}")
    return confirmed

def get_confirmed_role(session_id):
    """Récupère le rôle confirmé de l'utilisateur."""
    user_profile = session.get('user_profile', {})
    return user_profile.get('role', None)
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
        if groq_client:
            response = groq_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                temperature=0.95,
                max_tokens=28500,
                # Ajout d'un timeout pour éviter les attentes trop longues
                timeout=40
            )
            answer = response.choices[0].message.content
            answer = convert_markdown_to_html(answer)
            # Détection d'hallucinations simples
            if any(x in answer.lower() for x in ["@freddy", "freddy@"]):
                logger.warning("Hallucination détectée - correction appliquée")
                answer = "<p>⚠️ Je ne peux pas inventer de contacts inexistants.</p>"
            return answer
        else:
            # Amélioration du message de fallback
            return "<p>Désolé, le service de génération de réponse n'est pas disponible actuellement. Veuillez réessayer plus tard ou contacter l'administrateur.</p>"
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
app.secret_key = 'seatech_chat_secret_key'
app.config['SESSION_COOKIE_SECURE'] = False  # Important pour développement/HTTP
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 heure
conversation_history_global = {}
# ajout Daly

# Charger modèle Vosk une seule fois au démarrage
# vosk_model = vosk.Model("models/vosk-model-fr-0.6-linto-2.2.0") 
vosk_model = vosk.Model("models/vosk-model-small-fr-0.22")

#fin ajout

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
    logger.info(f"User profile dans session: {session.get('user_profile')}")
    
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
    user_profile = session.get('user_profile', {})
    
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
    print("DATA RECU:", data)
    print("SESSION:", session)
    try:
        data = request.get_json()
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
@app.route("/api/reset-role", methods=["POST"])
def api_reset_role():
    """Permet de réinitialiser le rôle sélectionné."""
    session.permanent = True
    
    if 'user_profile' in session:
        session.pop('user_profile')
        session.modified = True
    
    if 'session_id' in session:
        session_id = session['session_id']
        if session_id in conversation_history_global:
            conversation_history_global[session_id] = []
    
    logger.info(f"Rôle réinitialisé pour la session {session.get('session_id')}")
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

