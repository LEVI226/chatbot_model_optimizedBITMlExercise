# streamlit_app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time
import os
import sys
from datetime import datetime
import logging
import traceback
from typing import Optional, Tuple, Dict, Any
import warnings

# Suppression des warnings non critiques
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration des logs optimisée
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Configuration de la page avec métadonnées SEO
st.set_page_config(
    page_title="🤖 ChatBot IA - Deep Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/votre-repo/chatbot',
        'Report a bug': 'https://github.com/votre-repo/chatbot/issues',
        'About': "ChatBot IA propulsé par DialoGPT et Transformers"
    }
)

# CSS personnalisé ultra-moderne avec animations avancées
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Variables CSS globales */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --info-color: #3b82f6;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --glass-bg: rgba(255, 255, 255, 0.1);
        --border-radius: 16px;
        --border-radius-sm: 8px;
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.1);
        --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.1);
        --shadow-lg: 0 10px 25px rgba(0, 0, 0, 0.15);
        --shadow-xl: 0 20px 40px rgba(0, 0, 0, 0.2);
        --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        --font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
    }
    
    /* Reset et base */
    * {
        box-sizing: border-box;
    }
    
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }
    
    /* Header principal avec effet glassmorphism */
    .main-header {
        background: var(--background-gradient);
        padding: 3rem 2rem;
        border-radius: var(--border-radius);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-xl);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        animation: shine 4s infinite;
        pointer-events: none;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .main-header h1 {
        font-family: var(--font-family);
        font-weight: 800;
        margin: 0;
        font-size: clamp(2rem, 5vw, 3rem);
        position: relative;
        z-index: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.02em;
    }
    
    .main-header p {
        font-family: var(--font-family);
        font-weight: 400;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-size: clamp(0.9rem, 2vw, 1.1rem);
        position: relative;
        z-index: 1;
        letter-spacing: 0.02em;
    }
    
    /* Sidebar moderne avec glassmorphism */
    .sidebar-content {
        font-family: var(--font-family);
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Cartes de statistiques avec micro-interactions */
    .stats-card {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        border: 1px solid rgba(226, 232, 240, 0.8);
        margin-bottom: 1rem;
        box-shadow: var(--shadow-md);
        text-align: center;
        transition: var(--transition);
        position: relative;
        overflow: hidden;
        cursor: pointer;
    }
    
    .stats-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.1), transparent);
        transition: left 0.5s;
    }
    
    .stats-card:hover {
        transform: translateY(-4px) scale(1.02);
        box-shadow: var(--shadow-xl);
        border-color: var(--primary-color);
    }
    
    .stats-card:hover::before {
        left: 100%;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: var(--background-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        font-family: var(--font-mono);
    }
    
    .stats-label {
        font-size: 0.875rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
    }
    
    /* Indicateurs de statut avec animations */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.75rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.875rem;
        margin-bottom: 1rem;
        transition: var(--transition);
    }
    
    .status-online {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .status-loading {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: currentColor;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.1); }
    }
    
    /* Boutons avec effets avancés */
    .stButton > button {
        background: var(--background-gradient);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        font-family: var(--font-family);
        font-weight: 600;
        font-size: 0.875rem;
        padding: 0.75rem 1.5rem;
        transition: var(--transition);
        box-shadow: var(--shadow-md);
        position: relative;
        overflow: hidden;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
        filter: brightness(1.1);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-md);
    }
    
    /* Carte de bienvenue avec design moderne */
    .welcome-card {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 24px;
        margin: 2rem 0;
        border: 1px solid rgba(226, 232, 240, 0.8);
        position: relative;
        overflow: hidden;
        box-shadow: var(--shadow-lg);
    }
    
    .welcome-card::before {
        content: '🤖';
        font-size: 5rem;
        position: absolute;
        top: -2.5rem;
        left: 50%;
        transform: translateX(-50%);
        background: white;
        padding: 1.5rem;
        border-radius: 50%;
        box-shadow: var(--shadow-lg);
        border: 4px solid #f8fafc;
    }
    
    .welcome-card h2 {
        color: var(--primary-color);
        margin: 3rem 0 1.5rem 0;
        font-family: var(--font-family);
        font-weight: 700;
        font-size: clamp(1.5rem, 4vw, 2rem);
    }
    
    .welcome-card p {
        font-size: clamp(1rem, 2vw, 1.125rem);
        color: #64748b;
        margin-bottom: 1.5rem;
        font-family: var(--font-family);
        line-height: 1.7;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Métriques de performance avec design tech */
    .perf-metric {
        background: var(--background-gradient);
        color: white;
        padding: 0.625rem 1.25rem;
        border-radius: 50px;
        font-size: 0.8rem;
        margin: 0.25rem;
        display: inline-block;
        font-weight: 600;
        box-shadow: var(--shadow-md);
        font-family: var(--font-mono);
        letter-spacing: 0.02em;
        transition: var(--transition);
    }
    
    .perf-metric:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Messages de chat avec design moderne */
    .chat-message {
        padding: 1.25rem;
        margin: 1rem 0;
        border-radius: var(--border-radius);
        font-family: var(--font-family);
        line-height: 1.6;
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    
    .user-message {
        background: var(--background-gradient);
        color: white;
        margin-left: 2rem;
        border-bottom-right-radius: 4px;
    }
    
    .assistant-message {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-right: 2rem;
        border-bottom-left-radius: 4px;
    }
    
    /* Animation de chargement moderne */
    .loading-container {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
        background: rgba(102, 126, 234, 0.1);
        border-radius: var(--border-radius);
        margin: 1rem 0;
    }
    
    .loading-dots {
        display: flex;
        gap: 0.25rem;
    }
    
    .loading-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--primary-color);
        animation: loading 1.4s infinite ease-in-out;
    }
    
    .loading-dot:nth-child(1) { animation-delay: -0.32s; }
    .loading-dot:nth-child(2) { animation-delay: -0.16s; }
    .loading-dot:nth-child(3) { animation-delay: 0s; }
    
    @keyframes loading {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }
    
    /* Messages de notification */
    .notification {
        padding: 1rem 1.25rem;
        border-radius: var(--border-radius);
        margin: 1rem 0;
        font-family: var(--font-family);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .notification-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .notification-error {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
    }
    
    .notification-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);
    }
    
    .notification-info {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Accordéons et expandeurs */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #f8fafc, #e2e8f0);
        border-radius: var(--border-radius-sm);
        font-weight: 600;
        color: var(--primary-color);
    }
    
    /* Tables et données */
    .stDataFrame {
        border-radius: var(--border-radius);
        overflow: hidden;
        box-shadow: var(--shadow-md);
    }
    
    /* Métriques Streamlit */
    [data-testid="metric-container"] {
        background: linear-gradient(145deg, #ffffff, #f8fafc);
        border: 1px solid rgba(226, 232, 240, 0.8);
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
    }
    
    /* Responsive design avancé */
    @media (max-width: 768px) {
        .main-header {
            padding: 2rem 1rem;
            margin-bottom: 1rem;
        }
        
        .main-header h1 { 
            font-size: 1.75rem; 
        }
        
        .main-header p { 
            font-size: 0.875rem; 
        }
        
        .welcome-card { 
            padding: 2rem 1rem;
            margin: 1rem 0;
        }
        
        .stats-card { 
            padding: 1rem; 
        }
        
        .user-message { 
            margin-left: 0; 
        }
        
        .assistant-message { 
            margin-right: 0; 
        }
        
        .chat-message {
            padding: 1rem;
        }
    }
    
    @media (max-width: 480px) {
        .main .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        
        .stats-number {
            font-size: 2rem;
        }
        
        .perf-metric {
            font-size: 0.75rem;
            padding: 0.5rem 1rem;
        }
    }
    
    /* Animations d'entrée */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Scrollbar personnalisée */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--secondary-color);
    }
    
    /* Focus states pour l'accessibilité */
    .stButton > button:focus,
    .stTextInput > div > div > input:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* Masquer les éléments Streamlit par défaut */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Style pour les code blocks */
    .stCodeBlock {
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-md);
    }
    
    /* Style pour les alertes */
    .stAlert {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--shadow-md);
    }
</style>
""", unsafe_allow_html=True)

# Configuration globale des modèles et paramètres
class ModelConfig:
    """Configuration centralisée des modèles et paramètres"""
    
    # Chemins des modèles par ordre de priorité
    MODEL_PATHS = [
        "./chatbot_model_optimized",
        "./chatbot_model", 
        "microsoft/DialoGPT-medium",
        "microsoft/DialoGPT-small"
    ]
    
    # Paramètres de génération optimisés
    GENERATION_CONFIG = {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "repetition_penalty": 1.2,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "length_penalty": 1.0,
        "do_sample": True
    }
    
    # Limites et seuils
    MAX_INPUT_LENGTH = 500
    MAX_CONTEXT_LENGTH = 800
    MAX_RESPONSE_TOKENS = 80
    AUTO_RESET_THRESHOLD = 10
    CACHE_TTL = 300  # 5 minutes
    
    # Informations des modèles
    MODEL_INFO = {
        "DialoGPT-medium": {
            "parameters": "345M",
            "layers": 24,
            "attention_heads": 16,
            "vocabulary_size": "50,257",
            "context_length": 1024,
            "architecture": "GPT-2 based",
            "description": "Modèle conversationnel de taille moyenne"
        },
        "DialoGPT-small": {
            "parameters": "117M", 
            "layers": 12,
            "attention_heads": 12,
            "vocabulary_size": "50,257",
            "context_length": 1024,
            "architecture": "GPT-2 based",
            "description": "Modèle conversationnel léger"
        }
    }

class ChatbotError(Exception):
    """Exception personnalisée pour le chatbot"""
    pass

class ModelManager:
    """Gestionnaire de modèles avec cache et fallback intelligent"""
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_config = {}
        self.is_loaded = False
        
    @st.cache_resource(show_spinner=False)
    def load_model(_self) -> Tuple[Optional[Any], Optional[Any], bool, Dict[str, Any]]:
        """Chargement du modèle avec mise en cache optimisée et fallback robuste"""
        start_time = time.time()
        
        try:
            for i, model_path in enumerate(ModelConfig.MODEL_PATHS):
                try:
                    logger.info(f"Tentative de chargement depuis: {model_path}")
                    
                    # Configuration selon le type de modèle
                    is_local = not model_path.startswith("microsoft/")
                    
                    # Configuration de chargement adaptative
                    load_config = {
                        "local_files_only": is_local,
                        "trust_remote_code": False,
                        "torch_dtype": torch.float32,
                        "low_cpu_mem_usage": True
                    }
                    
                    # Chargement du tokenizer avec gestion d'erreur
                    with st.spinner(f"🔄 Chargement du tokenizer ({i+1}/{len(ModelConfig.MODEL_PATHS)})..."):
                        tokenizer = AutoTokenizer.from_pretrained(
                            model_path,
                            local_files_only=is_local,
                            trust_remote_code=False
                        )
                    
                    # Chargement du modèle avec optimisations
                    with st.spinner(f"🔄 Chargement du modèle ({i+1}/{len(ModelConfig.MODEL_PATHS)})..."):
                        model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            **load_config,
                            device_map="cpu"
                        )
                    
                    # Configuration et optimisation du modèle
                    model.eval()
                    
                    # Configuration du pad_token avec vérification
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                        tokenizer.pad_token_id = tokenizer.eos_token_id
                    
                    # Calcul des métriques du modèle
                    load_time = time.time() - start_time
                    model_size = sum(p.numel() for p in model.parameters()) / 1e6
                    
                    logger.info(f"Modèle chargé avec succès en {load_time:.2f}s")
                    
                    # Message de succès avec détails
                    source = "Local" if is_local else "HuggingFace Hub"
                    model_name = model_path.split('/')[-1]
                    
                    st.success(f"✅ Modèle {model_name} chargé depuis {source} en {load_time:.1f}s")
                    
                    return tokenizer, model, True, {
                        "source": source,
                        "path": model_path,
                        "load_time": load_time,
                        "model_size": model_size,
                        "parameters": f"{model_size:.0f}M",
                        "model_name": model_name
                    }
                    
                except Exception as e:
                    logger.warning(f"Échec du chargement depuis {model_path}: {str(e)[:100]}...")
                    if i < len(ModelConfig.MODEL_PATHS) - 1:
                        st.warning(f"⚠️ {model_path} non disponible, tentative suivante...")
                    continue
            
            raise ChatbotError("Tous les chemins de modèle ont échoué")
            
        except Exception as e:
            logger.error(f"Erreur critique lors du chargement: {e}")
            st.error(f"❌ Impossible de charger le modèle: {str(e)[:200]}...")
            
            # Suggestions de dépannage
            st.info("""
            💡 **Suggestions de dépannage:**
            - Vérifiez votre connexion internet
            - Le modèle sera téléchargé automatiquement depuis HuggingFace
            - Essayez de redémarrer l'application
            - Contactez le support si le problème persiste
            """)
            
            return None, None, False, {}

class ConversationManager:
    """Gestionnaire de conversation avec optimisations mémoire"""
    
    def __init__(self):
        self.reset_conversation()
    
    def reset_conversation(self):
        """Reset complet de la conversation avec nettoyage mémoire"""
        st.session_state.messages = []
        st.session_state.chat_history_ids = None
        st.session_state.conversation_count = 0
        st.session_state.total_tokens_generated = 0
        st.session_state.total_response_time = 0
        st.session_state.error_count = 0
        
        # Nettoyage mémoire agressif
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def generate_response(self, user_input: str, tokenizer, model, chat_history_ids=None, max_attempts: int = 3) -> Tuple[str, Optional[torch.Tensor]]:
        """Génération de réponse optimisée avec retry logic et validation robuste"""
        
        for attempt in range(max_attempts):
            try:
                # Validation et nettoyage de l'entrée
                user_input = user_input.strip()
                if not user_input:
                    return "Pouvez-vous répéter votre message ?", chat_history_ids
                
                # Limitation de longueur pour éviter les débordements
                if len(user_input) > ModelConfig.MAX_INPUT_LENGTH:
                    user_input = user_input[:ModelConfig.MAX_INPUT_LENGTH] + "..."
                
                # Encodage de l'entrée utilisateur avec gestion d'erreur
                try:
                    encoded = tokenizer(
                        user_input + tokenizer.eos_token,
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        max_length=512
                    )
                    new_user_input_ids = encoded['input_ids']
                except Exception as e:
                    logger.error(f"Erreur d'encodage: {e}")
                    return "Désolé, je n'arrive pas à traiter votre message.", chat_history_ids
                
                # Gestion intelligente de l'historique avec optimisation mémoire
                if chat_history_ids is not None:
                    # Limiter drastiquement la taille pour éviter les débordements
                    if chat_history_ids.shape[-1] > ModelConfig.MAX_CONTEXT_LENGTH:
                        # Garder seulement les tokens les plus récents
                        chat_history_ids = chat_history_ids[:, -ModelConfig.MAX_CONTEXT_LENGTH:]
                    
                    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
                else:
                    bot_input_ids = new_user_input_ids
                
                # Vérification finale de la taille
                if bot_input_ids.shape[-1] > ModelConfig.MAX_CONTEXT_LENGTH:
                    bot_input_ids = bot_input_ids[:, -ModelConfig.MAX_CONTEXT_LENGTH:]
                
                # Configuration de génération optimisée
                generation_config = {
                    **ModelConfig.GENERATION_CONFIG,
                    "max_length": min(bot_input_ids.shape[-1] + ModelConfig.MAX_RESPONSE_TOKENS, 900),
                    "min_length": bot_input_ids.shape[-1] + 5,
                    "num_return_sequences": 1,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }
                
                # Génération avec gestion mémoire
                with torch.no_grad():
                    try:
                        chat_history_ids = model.generate(
                            bot_input_ids,
                            **generation_config
                        )
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            # Réduction drastique en cas de problème mémoire
                            generation_config["max_length"] = min(bot_input_ids.shape[-1] + 30, 600)
                            chat_history_ids = model.generate(bot_input_ids, **generation_config)
                        else:
                            raise e
                
                # Extraction et nettoyage de la réponse
                response = tokenizer.decode(
                    chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                    skip_special_tokens=True
                ).strip()
                
                # Post-traitement amélioré de la réponse
                if not response or len(response) < 2:
                    fallback_responses = [
                        "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?",
                        "Intéressant ! Pouvez-vous m'en dire plus ?",
                        "Hmm, pouvez-vous être plus spécifique ?",
                        "Je réfléchis... Pouvez-vous répéter différemment ?"
                    ]
                    response = fallback_responses[attempt % len(fallback_responses)]
                
                # Nettoyage des répétitions et artefacts
                response = self._clean_response(response)
                
                # Nettoyage mémoire agressif
                del bot_input_ids, encoded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                return response, chat_history_ids
                
            except Exception as e:
                logger.warning(f"Tentative {attempt + 1} échouée: {str(e)[:100]}...")
                if attempt == max_attempts - 1:
                    error_responses = [
                        "Désolé, je rencontre des difficultés techniques. Pouvez-vous réessayer ?",
                        "Je suis temporairement indisponible. Veuillez patienter un moment.",
                        "Erreur technique. Essayez de reformuler votre question."
                    ]
                    return error_responses[0], chat_history_ids
                
                # Pause progressive entre les tentatives
                time.sleep(0.5 * (attempt + 1))
                
                # Nettoyage mémoire entre les tentatives
                gc.collect()
        
        return "Erreur technique persistante.", chat_history_ids
    
    def _clean_response(self, response: str) -> str:
        """Nettoyage et amélioration de la réponse générée"""
        if not response:
            return response
        
        # Suppression des répétitions en fin de phrase
        sentences = response.split('.')
        if len(sentences) > 2:
            # Vérifier les répétitions
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and sentence not in cleaned_sentences:
                    cleaned_sentences.append(sentence)
            
            if cleaned_sentences:
                response = '. '.join(cleaned_sentences)
                if not response.endswith('.') and not response.endswith('!') and not response.endswith('?'):
                    response += '.'
        
        # Suppression des caractères indésirables
        response = response.replace('  ', ' ').strip()
        
        # Limitation de longueur
        if len(response) > 300:
            # Couper à la dernière phrase complète
            last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_punct > 100:
                response = response[:last_punct + 1]
            else:
                response = response[:300] + "..."
        
        return response
    
    def get_conversation_stats(self) -> Dict[str, float]:
        """Calcul des statistiques de conversation"""
        if not st.session_state.messages:
            return {
                "avg_response_time": 0,
                "avg_message_length": 0,
                "user_messages": 0,
                "bot_messages": 0,
                "avg_user_length": 0,
                "avg_bot_length": 0
            }
        
        user_messages = [msg for msg in st.session_state.messages if msg["role"] == "user"]
        bot_messages = [msg for msg in st.session_state.messages if msg["role"] == "assistant"]
        
        avg_user_length = sum(len(msg["content"]) for msg in user_messages) / len(user_messages) if user_messages else 0
        avg_bot_length = sum(len(msg["content"]) for msg in bot_messages) / len(bot_messages) if bot_messages else 0
        
        return {
            "avg_response_time": getattr(st.session_state, 'total_response_time', 0) / max(len(bot_messages), 1),
            "avg_user_length": avg_user_length,
            "avg_bot_length": avg_bot_length,
            "user_messages": len(user_messages),
            "bot_messages": len(bot_messages)
        }

class UIManager:
    """Gestionnaire d'interface utilisateur avec composants réutilisables"""
    
    @staticmethod
    def render_header():
        """Rendu du header principal avec animation"""
        st.markdown("""
        <div class="main-header fade-in-up">
            <h1>🤖 ChatBot IA - Deep Learning</h1>
            <p>Propulsé par DialoGPT • Transformers • Dataset DailyDialog</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_status_indicator(status: str, message: str):
        """Rendu d'un indicateur de statut"""
        status_classes = {
            "online": "status-online",
            "loading": "status-loading", 
            "error": "status-error"
        }
        
        icons = {
            "online": "🟢",
            "loading": "🟡",
            "error": "🔴"
        }
        
        st.markdown(f"""
        <div class="status-indicator {status_classes.get(status, 'status-online')}">
            <span class="status-dot"></span>
            {icons.get(status, "🟢")} {message}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_stats_card(title: str, value: str, subtitle: str = ""):
        """Rendu d'une carte de statistique"""
        st.markdown(f"""
        <div class="stats-card">
            <div class="stats-number">{value}</div>
            <div class="stats-label">{title}</div>
            {f'<div style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem;">{subtitle}</div>' if subtitle else ''}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_perf_metric(label: str, value: str):
        """Rendu d'une métrique de performance"""
        st.markdown(f"""
        <div class="perf-metric">{label}: {value}</div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_welcome_card():
        """Rendu de la carte de bienvenue"""
        st.markdown("""
        <div class="welcome-card fade-in-up">
            <h2>👋 Bienvenue dans votre ChatBot IA !</h2>
            <p>
                Je suis votre assistant IA basé sur DialoGPT, entraîné sur des conversations naturelles.<br>
                Commencez une conversation en tapant un message ci-dessous !
            </p>
            <p style="font-size: 0.9rem; color: #64748b;">
                💡 Utilisez les suggestions dans la barre latérale pour découvrir mes capacités
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_notification(type_: str, message: str):
        """Rendu d'une notification"""
        st.markdown(f"""
        <div class="notification notification-{type_}">
            {message}
        </div>
        """, unsafe_allow_html=True)

def response_generator(response_text: str, speed: float = 0.04):
    """Générateur pour l'effet de streaming amélioré avec pauses naturelles"""
    if not response_text:
        yield "..."
        return
    
    words = response_text.split()
    
    for i, word in enumerate(words):
        yield word + " "
        
        # Pauses naturelles selon la ponctuation
        if word.endswith(('.', '!', '?')):
            time.sleep(speed * 4)  # Pause longue
        elif word.endswith((',', ';', ':')):
            time.sleep(speed * 2)  # Pause moyenne
        else:
            time.sleep(speed)  # Pause normale

@st.cache_data(ttl=ModelConfig.CACHE_TTL, show_spinner=False)
def get_model_info(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Informations détaillées sur le modèle avec cache"""
    model_name = model_config.get("model_name", "DialoGPT-medium")
    return ModelConfig.MODEL_INFO.get(model_name, ModelConfig.MODEL_INFO["DialoGPT-medium"])

def main():
    """Fonction principale avec gestion d'erreur globale et architecture modulaire"""
    try:
        # Initialisation des gestionnaires
        model_manager = ModelManager()
        conversation_manager = ConversationManager()
        ui_manager = UIManager()
        
        # Rendu du header
        ui_manager.render_header()
        
        # Initialisation robuste de l'état de session
        session_defaults = {
            "messages": [],
            "chat_history_ids": None,
            "conversation_count": 0,
            "total_tokens_generated": 0,
            "total_response_time": 0,
            "model_loaded": False,
            "error_count": 0,
            "last_activity": time.time()
        }
        
        for key, default_value in session_defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        # Chargement du modèle avec gestion d'état améliorée
        if not st.session_state.model_loaded:
            with st.spinner("🚀 Initialisation du système IA..."):
                tokenizer, model, model_loaded, model_config = model_manager.load_model()
                
                if model_loaded:
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model = model
                    st.session_state.model_config = model_config
                    st.session_state.model_loaded = True
                    st.session_state.model_info = get_model_info(model_config)
                    st.session_state.error_count = 0
                else:
                    st.session_state.error_count += 1
        
        # Gestion des erreurs de chargement
        if not st.session_state.model_loaded:
            ui_manager.render_notification("error", "❌ Impossible de charger le modèle. Vérifiez la configuration.")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Réessayer le chargement", use_container_width=True):
                    st.session_state.model_loaded = False
                    st.cache_resource.clear()
                    st.rerun()
            
            with col2:
                if st.button("📋 Voir les détails", use_container_width=True):
                    st.info("""
                    **Causes possibles:**
                    - Connexion internet instable
                    - Modèle en cours de téléchargement
                    - Ressources insuffisantes
                    
                    **Solutions:**
                    - Attendez quelques minutes
                    - Rafraîchissez la page
                    - Vérifiez votre connexion
                    """)
            return
        
        # Sidebar avec statistiques et contrôles améliorés
        with st.sidebar:
            st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
            
            # Status du système avec indicateur animé
            st.markdown("## 🔌 Statut Système")
            ui_manager.render_status_indicator("online", "En ligne")
            
            # Informations du modèle avec métriques
            if "model_config" in st.session_state:
                config = st.session_state.model_config
                st.markdown("### 📊 Modèle Actuel")
                st.markdown(f"""
                **🏷️ Modèle:** {config.get('model_name', 'DialoGPT')}  
                **📍 Source:** {config['source']}  
                **⚡ Chargement:** {config['load_time']:.1f}s  
                **🧠 Paramètres:** {config['parameters']}  
                **💾 Taille:** {config['model_size']:.0f}M params
                """)
            
            st.markdown("## 📈 Statistiques Temps Réel")
            
            # Cartes de statistiques avec hover
            stats = conversation_manager.get_conversation_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                ui_manager.render_stats_card(
                    "Messages Total", 
                    str(len(st.session_state.messages)),
                    f"{stats['user_messages']} utilisateur"
                )
            
            with col2:
                ui_manager.render_stats_card(
                    "Tours de Parole", 
                    str(st.session_state.conversation_count),
                    f"{stats['bot_messages']} réponses"
                )
            
            # Métriques de performance détaillées
            if st.session_state.total_tokens_generated > 0:
                st.markdown("### 🎯 Métriques Performance")
                
                col1, col2 = st.columns(2)
                with col1:
                    ui_manager.render_perf_metric("Tokens", str(st.session_state.total_tokens_generated))
                
                with col2:
                    avg_time = stats["avg_response_time"]
                    ui_manager.render_perf_metric("Temps moy", f"{avg_time:.1f}s")
                
                # Statistiques additionnelles
                if stats["user_messages"] > 0:
                    st.markdown(f"""
                    **📝 Messages utilisateur:** {stats["user_messages"]}  
                    **🤖 Réponses IA:** {stats["bot_messages"]}  
                    **📏 Longueur moy. user:** {stats["avg_user_length"]:.0f} chars  
                    **📏 Longueur moy. bot:** {stats["avg_bot_length"]:.0f} chars
                    """)
            
            st.markdown("## ⚙️ Contrôles")
            
            # Boutons de contrôle avec confirmations
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Reset Chat", use_container_width=True):
                    conversation_manager.reset_conversation()
                    ui_manager.render_notification("success", "💬 Conversation réinitialisée!")
                    time.sleep(1)
                    st.rerun()
            
            with col2:
                if st.button("🔄 Restart IA", use_container_width=True):
                    st.session_state.model_loaded = False
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    ui_manager.render_notification("success", "🤖 IA redémarrée!")
                    time.sleep(1)
                    st.rerun()
            
            # Paramètres avancés avec explications
            with st.expander("🔧 Paramètres Avancés"):
                st.markdown(f"""
                **🔄 Auto-reset:** Après {ModelConfig.AUTO_RESET_THRESHOLD} échanges  
                **📏 Max réponse:** {ModelConfig.MAX_RESPONSE_TOKENS} tokens  
                **🌡️ Temperature:** {ModelConfig.GENERATION_CONFIG['temperature']} (créativité)  
                **🔁 Anti-répétition:** {ModelConfig.GENERATION_CONFIG['repetition_penalty']}  
                **🎯 Top-k:** {ModelConfig.GENERATION_CONFIG['top_k']} | **Top-p:** {ModelConfig.GENERATION_CONFIG['top_p']}  
                **💾 Contexte max:** {ModelConfig.MAX_CONTEXT_LENGTH} tokens
                """)
                
                # Options de débogage
                if st.checkbox("🐛 Mode Debug"):
                    debug_info = {
                        "model_loaded": st.session_state.model_loaded,
                        "messages_count": len(st.session_state.messages),
                        "conversation_count": st.session_state.conversation_count,
                        "error_count": st.session_state.error_count,
                        "memory_usage": f"{gc.get_count()}",
                        "last_activity": datetime.fromtimestamp(st.session_state.last_activity).strftime("%H:%M:%S")
                    }
                    st.json(debug_info)
            
            # Informations techniques du modèle
            st.markdown("## ℹ️ Architecture Modèle")
            if "model_info" in st.session_state:
                info = st.session_state.model_info
                st.markdown(f"""
                **🏗️ Type:** {info['architecture']}  
                **📊 Paramètres:** {info['parameters']}  
                **🔧 Couches:** {info['layers']}  
                **👁️ Têtes attention:** {info['attention_heads']}  
                **📚 Vocabulaire:** {info['vocabulary_size']}  
                **💾 Contexte:** {info['context_length']} tokens
                """)
            
            # Exemples de questions par catégorie
            st.markdown("## 💡 Suggestions de Conversation")
            
            example_categories = {
                "🎯 Questions Générales": [
                    "Hello, how are you today?",
                    "What's your favorite hobby?", 
                    "Tell me about yourself",
                    "How was your day?"
                ],
                "🎭 Divertissement": [
                    "Tell me a joke",
                    "What's a fun fact?",
                    "Share an interesting story",
                    "What makes you happy?"
                ],
                "🤔 Conversations Profondes": [
                    "What do you think about AI?",
                    "What motivates you?",
                    "What's your opinion on technology?",
                    "How do you see the future?"
                ],
                "🌟 Créativité": [
                    "Write a short poem",
                    "Describe a perfect day",
                    "What would you do if you were human?",
                    "Tell me about your dreams"
                ]
            }
            
            for category, questions in example_categories.items():
                with st.expander(category):
                    for question in questions:
                        if st.button(f"💬 {question}", key=f"example_{hash(question)}", use_container_width=True):
                            # Ajouter à l'historique
                            st.session_state.messages.append({"role": "user", "content": question})
                            
                            # Générer la réponse
                            start_time = time.time()
                            with st.spinner("🤔 Génération de la réponse..."):
                                bot_response, new_chat_history_ids = conversation_manager.generate_response(
                                    question, 
                                    st.session_state.tokenizer, 
                                    st.session_state.model, 
                                    st.session_state.chat_history_ids
                                )
                            
                            generation_time = time.time() - start_time
                            
                            # Mise à jour de l'état
                            st.session_state.messages.append({"role": "assistant", "content": bot_response})
                            st.session_state.chat_history_ids = new_chat_history_ids
                            st.session_state.conversation_count += 1
                            st.session_state.total_tokens_generated += len(bot_response.split())
                            st.session_state.total_response_time += generation_time
                            st.session_state.last_activity = time.time()
                            
                            # Reset automatique après seuil
                            if st.session_state.conversation_count >= ModelConfig.AUTO_RESET_THRESHOLD:
                                st.session_state.chat_history_ids = None
                                st.session_state.conversation_count = 0
                            
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Affichage de l'historique avec les éléments de chat natifs
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Input de chat natif Streamlit avec validation
        if prompt := st.chat_input("💭 Tapez votre message ici... (max 500 caractères)"):
            # Validation robuste de l'entrée
            if len(prompt.strip()) == 0:
                st.warning("⚠️ Veuillez saisir un message non vide.")
                return
            
            if len(prompt) > ModelConfig.MAX_INPUT_LENGTH:
                st.warning(f"⚠️ Message trop long (max {ModelConfig.MAX_INPUT_LENGTH} caractères).")
                return
            
            # Filtrage de contenu basique
            if any(word in prompt.lower() for word in ['spam', 'test' * 10]):
                st.warning("⚠️ Message détecté comme spam.")
                return
            
            # Affichage du message utilisateur
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Génération et affichage de la réponse
            with st.chat_message("assistant"):
                start_time = time.time()
                
                with st.spinner("🤔 Réflexion en cours..."):
                    bot_response, new_chat_history_ids = conversation_manager.generate_response(
                        prompt, 
                        st.session_state.tokenizer, 
                        st.session_state.model, 
                        st.session_state.chat_history_ids
                    )
                
                generation_time = time.time() - start_time
                
                # Affichage avec streaming amélioré
                if bot_response and len(bot_response.strip()) > 0:
                    response = st.write_stream(response_generator(bot_response))
                else:
                    response = "Désolé, je n'ai pas pu générer une réponse appropriée."
                    st.write(response)
                
                # Métrique de performance (optionnel)
                with st.expander("📈 Détails de génération"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("⏱️ Temps", f"{generation_time:.2f}s")
                    with col2:
                        st.metric("📝 Mots", len(response.split()))
                    with col3:
                        st.metric("📏 Caractères", len(response))
            
            # Mise à jour de l'état avec gestion d'erreur
            try:
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.session_state.chat_history_ids = new_chat_history_ids
                st.session_state.conversation_count += 1
                st.session_state.total_tokens_generated += len(response.split())
                st.session_state.total_response_time += generation_time
                st.session_state.last_activity = time.time()
                
                # Reset automatique avec notification
                if st.session_state.conversation_count >= ModelConfig.AUTO_RESET_THRESHOLD:
                    st.session_state.chat_history_ids = None
                    st.session_state.conversation_count = 0
                    st.info("🔄 Contexte automatiquement réinitialisé pour optimiser les performances.")
                    
            except Exception as e:
                logger.error(f"Erreur lors de la mise à jour de l'état: {e}")
                st.error("Erreur lors de la sauvegarde de la conversation.")
        
        # Message de bienvenue si pas d'historique
        if not st.session_state.messages:
            ui_manager.render_welcome_card()
        
        # Footer avec informations techniques complètes
        with st.expander("🔧 Informations Techniques Détaillées"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🏗️ Architecture Modèle**
                - Base: GPT-2 architecture
                - Spécialisation: Dialogue
                - Entraînement: DailyDialog dataset
                - Optimisation: Conversation naturelle
                - Tokenizer: GPT-2 BPE
                - Attention: Multi-head self-attention
                """)
            
            with col2:
                st.markdown(f"""
                **⚙️ Paramètres Génération**
                - Temperature: {ModelConfig.GENERATION_CONFIG['temperature']} (équilibre créativité/cohérence)
                - Top-k: {ModelConfig.GENERATION_CONFIG['top_k']} (diversité contrôlée)
                - Top-p: {ModelConfig.GENERATION_CONFIG['top_p']} (nucleus sampling)
                - Repetition penalty: {ModelConfig.GENERATION_CONFIG['repetition_penalty']} (anti-répétition)
                - No repeat n-gram: {ModelConfig.GENERATION_CONFIG['no_repeat_ngram_size']} (évite répétitions)
                - Early stopping: {ModelConfig.GENERATION_CONFIG['early_stopping']}
                """)
            
            with col3:
                st.markdown("""
                **📊 Performance & Optimisation**
                - Device: CPU optimized
                - Precision: Float32
                - Memory: Auto garbage collection
                - Streaming: Real-time word display
                - Cache: Streamlit @cache_resource
                - Fallback: Multi-model support
                """)
            
            # Informations système détaillées
            st.markdown("**💻 Environnement Système**")
            
            try:
                system_info = {
                    "PyTorch Version": torch.__version__,
                    "Device": "CPU (Streamlit Cloud optimized)",
                    "Python Version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "Memory Management": "Auto garbage collection + torch cache clearing",
                    "Streaming": "Real-time word-by-word streaming",
                    "Cache Strategy": "Model cached with @cache_resource, data with @cache_data",
                    "Error Handling": "Multi-attempt generation with fallback responses",
                    "Context Management": f
