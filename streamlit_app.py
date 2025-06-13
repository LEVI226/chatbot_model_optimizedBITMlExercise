# streamlit_app.py

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import time
import os
from datetime import datetime
import logging

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration de la page
st.set_page_config(
    page_title="🤖 ChatBot IA - Deep Learning",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé moderne et épuré
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Variables CSS */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --background-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --border-radius: 12px;
        --shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Header principal */
    .main-header {
        background: var(--background-gradient);
        padding: 2rem;
        border-radius: var(--border-radius);
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
    }
    
    .main-header h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        margin: 0;
        font-size: 2.2rem;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1rem;
    }
    
    /* Sidebar styling */
    .sidebar-content {
        font-family: 'Inter', sans-serif;
    }
    
    .stats-card {
        background: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
        box-shadow: var(--shadow);
        text-align: center;
    }
    
    .stats-number {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .stats-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.25rem;
    }
    
    /* Status indicators */
    .status-online {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-loading {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--background-gradient);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Welcome message */
    .welcome-card {
        text-align: center;
        padding: 3rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .welcome-card h2 {
        color: #667eea;
        margin-bottom: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .welcome-card p {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 1.5rem;
        font-family: 'Inter', sans-serif;
    }
    
    /* Performance metrics */
    .perf-metric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.8rem; }
        .main-header p { font-size: 0.9rem; }
        .welcome-card { padding: 2rem 1rem; }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    """Chargement du modèle avec mise en cache optimisée et fallback"""
    start_time = time.time()
    
    try:
        # Priorité aux modèles locaux optimisés
        model_paths = [
            "./chatbot_model_optimized",
            "./chatbot_model",
            "microsoft/DialoGPT-medium"  # Fallback HuggingFace
        ]
        
        for i, model_path in enumerate(model_paths):
            try:
                logger.info(f"Tentative de chargement depuis: {model_path}")
                
                # Configuration selon le type de modèle
                is_local = not model_path.startswith("microsoft/")
                load_config = {
                    "local_files_only": is_local,
                    "trust_remote_code": False
                }
                
                # Chargement du tokenizer
                with st.spinner(f"🔄 Chargement du tokenizer ({i+1}/{len(model_paths)})..."):
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        **load_config
                    )
                
                # Chargement du modèle
                with st.spinner(f"🔄 Chargement du modèle ({i+1}/{len(model_paths)})..."):
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        **load_config
                    )
                
                # Configuration du modèle
                model.eval()
                
                # Configuration du pad_token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                load_time = time.time() - start_time
                model_size = sum(p.numel() for p in model.parameters()) / 1e6  # Millions de paramètres
                
                logger.info(f"Modèle chargé avec succès en {load_time:.2f}s")
                
                # Message de succès avec détails
                source = "Local" if is_local else "HuggingFace Hub"
                st.success(f"✅ Modèle chargé depuis {source} en {load_time:.1f}s")
                
                return tokenizer, model, True, {
                    "source": source,
                    "path": model_path,
                    "load_time": load_time,
                    "model_size": model_size,
                    "parameters": f"{model_size:.0f}M"
                }
                
            except Exception as e:
                logger.warning(f"Échec du chargement depuis {model_path}: {e}")
                if i < len(model_paths) - 1:
                    st.warning(f"⚠️ {model_path} non disponible, tentative suivante...")
                continue
        
        raise Exception("Tous les chemins de modèle ont échoué")
        
    except Exception as e:
        logger.error(f"Erreur critique lors du chargement: {e}")
        st.error(f"❌ Impossible de charger le modèle: {e}")
        return None, None, False, {}

@st.cache_data(ttl=300, show_spinner=False)  # Cache 5 minutes
def get_model_info(model_config):
    """Informations détaillées sur le modèle"""
    return {
        "architecture": "DialoGPT-medium",
        "parameters": model_config.get("parameters", "345M"),
        "layers": 24,
        "attention_heads": 16,
        "vocabulary_size": "50,257",
        "context_length": 1024
    }

def generate_response(user_input, tokenizer, model, chat_history_ids=None, max_attempts=3):
    """Génération de réponse optimisée avec retry logic"""
    
    for attempt in range(max_attempts):
        try:
            # Nettoyage de l'entrée
            user_input = user_input.strip()
            if not user_input:
                return "Pouvez-vous répéter votre message ?", chat_history_ids
            
            # Encodage de l'entrée utilisateur
            encoded = tokenizer(
                user_input + tokenizer.eos_token,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            new_user_input_ids = encoded['input_ids']
            
            # Gestion intelligente de l'historique
            if chat_history_ids is not None:
                # Limiter la taille de l'historique pour éviter les débordements
                if chat_history_ids.shape[-1] > 700:
                    # Garder seulement les 500 derniers tokens
                    chat_history_ids = chat_history_ids[:, -500:]
                
                bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            else:
                bot_input_ids = new_user_input_ids
            
            # Génération avec paramètres optimisés
            generation_config = {
                "max_length": min(bot_input_ids.shape[-1] + 100, 1000),
                "num_return_sequences": 1,
                "temperature": 0.8,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 3,
                "early_stopping": True
            }
            
            with torch.no_grad():
                chat_history_ids = model.generate(
                    bot_input_ids,
                    **generation_config
                )
            
            # Extraction de la réponse
            response = tokenizer.decode(
                chat_history_ids[:, bot_input_ids.shape[-1]:][0], 
                skip_special_tokens=True
            ).strip()
            
            # Post-traitement de la réponse
            if not response or len(response) < 2:
                response = "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?"
            
            # Nettoyage des répétitions en fin de phrase
            sentences = response.split('.')
            if len(sentences) > 1 and sentences[-2] == sentences[-1]:
                response = '.'.join(sentences[:-1]) + '.'
            
            # Nettoyage mémoire
            del bot_input_ids, encoded
            gc.collect()
            
            return response, chat_history_ids
            
        except Exception as e:
            logger.warning(f"Tentative {attempt + 1} échouée: {e}")
            if attempt == max_attempts - 1:
                return "Désolé, je rencontre des difficultés techniques. Pouvez-vous réessayer ?", chat_history_ids
            time.sleep(0.5)  # Pause courte avant retry
    
    return "Erreur technique persistante.", chat_history_ids

def response_generator(response_text, speed=0.03):
    """Générateur pour l'effet de streaming amélioré"""
    if not response_text:
        yield "..."
        return
    
    words = response_text.split()
    for i, word in enumerate(words):
        yield word + " "
        # Pause plus longue après la ponctuation
        if word.endswith(('.', '!', '?', ':')):
            time.sleep(speed * 3)
        else:
            time.sleep(speed)

def reset_conversation():
    """Reset complet de la conversation"""
    st.session_state.messages = []
    st.session_state.chat_history_ids = None
    st.session_state.conversation_count = 0
    st.session_state.total_tokens_generated = 0
    gc.collect()  # Nettoyage mémoire

def main():
    # Header principal avec animation
    st.markdown("""
    <div class="main-header">
        <h1>🤖 ChatBot IA - Deep Learning</h1>
        <p>Propulsé par DialoGPT-medium • Transformers • Dataset DailyDialog</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialisation de l'état de session
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chat_history_ids" not in st.session_state:
        st.session_state.chat_history_ids = None
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    if "total_tokens_generated" not in st.session_state:
        st.session_state.total_tokens_generated = 0
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    
    # Chargement du modèle avec gestion d'état
    if not st.session_state.model_loaded:
        with st.spinner("🚀 Initialisation du système IA..."):
            tokenizer, model, model_loaded, model_config = load_model()
            
            if model_loaded:
                st.session_state.tokenizer = tokenizer
                st.session_state.model = model
                st.session_state.model_config = model_config
                st.session_state.model_loaded = True
                st.session_state.model_info = get_model_info(model_config)
    
    if not st.session_state.model_loaded:
        st.error("❌ Impossible de charger le modèle. Vérifiez la configuration.")
        st.info("💡 Le modèle sera téléchargé automatiquement depuis HuggingFace si les fichiers locaux ne sont pas disponibles.")
        return
    
    # Sidebar avec statistiques et contrôles
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Status du système
        st.markdown("## 🔌 Statut Système")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("🟢")
        with col2:
            st.markdown('<span class="status-online">En ligne</span>', unsafe_allow_html=True)
        
        # Informations du modèle
        if "model_config" in st.session_state:
            config = st.session_state.model_config
            st.markdown(f"""
            **🏷️ Source:** {config['source']}  
            **⚡ Chargement:** {config['load_time']:.1f}s  
            **🧠 Paramètres:** {config['parameters']}
            """)
        
        st.markdown("## 📊 Statistiques en Temps Réel")
        
        # Cartes de statistiques améliorées
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(st.session_state.messages)}</div>
                <div class="stats-label">Messages</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{st.session_state.conversation_count}</div>
                <div class="stats-label">Tours</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Métriques de performance
        if hasattr(st.session_state, 'total_tokens_generated'):
            st.markdown(f"""
            <div class="perf-metric">Tokens générés: {st.session_state.total_tokens_generated}</div>
            """, unsafe_allow_html=True)
        
        st.markdown("## ⚙️ Contrôles")
        
        # Boutons de contrôle
        if st.button("🗑️ Nouvelle Conversation", use_container_width=True):
            reset_conversation()
            st.success("💬 Conversation réinitialisée!")
            time.sleep(1)
            st.rerun()
        
        if st.button("🔄 Redémarrer IA", use_container_width=True):
            st.session_state.model_loaded = False
            st.cache_resource.clear()
            st.success("🤖 IA redémarrée!")
            time.sleep(1)
            st.rerun()
        
        # Paramètres avancés
        with st.expander("🔧 Paramètres Avancés"):
            st.markdown("""
            **Réinitialisation automatique:** Après 10 échanges  
            **Longueur max réponse:** 100 tokens  
            **Temperature:** 0.8 (créativité)  
            **Répétition:** Penalty 1.15
            """)
        
        st.markdown("## ℹ️ Informations Techniques")
        if "model_info" in st.session_state:
            info = st.session_state.model_info
            st.markdown(f"""
            **🏗️ Architecture:** {info['architecture']}  
            **📊 Paramètres:** {info['parameters']}  
            **🔧 Couches:** {info['layers']}  
            **👁️ Têtes attention:** {info['attention_heads']}  
            **📚 Vocabulaire:** {info['vocabulary_size']}  
            **💾 Contexte:** {info['context_length']} tokens
            """)
        
        # Exemples de questions avec catégories
        st.markdown("## 💡 Suggestions de Conversation")
        
        example_categories = {
            "🎯 Questions générales": [
                "Hello, how are you today?",
                "What's your favorite hobby?",
                "Tell me about yourself"
            ],
            "🎭 Divertissement": [
                "Tell me a joke",
                "What's a fun fact?",
                "Share an interesting story"
            ],
            "🤔 Conversations profondes": [
                "What do you think about AI?",
                "How was your day?",
                "What motivates you?"
            ]
        }
        
        for category, questions in example_categories.items():
            with st.expander(category):
                for question in questions:
                    if st.button(f"💬 {question}", key=f"example_{hash(question)}", use_container_width=True):
                        # Ajouter à l'historique et traiter
                        st.session_state.messages.append({"role": "user", "content": question})
                        
                        # Générer la réponse
                        with st.spinner("🤔 Génération de la réponse..."):
                            bot_response, new_chat_history_ids = generate_response(
                                question, 
                                st.session_state.tokenizer, 
                                st.session_state.model, 
                                st.session_state.chat_history_ids
                            )
                        
                        st.session_state.messages.append({"role": "assistant", "content": bot_response})
                        st.session_state.chat_history_ids = new_chat_history_ids
                        st.session_state.conversation_count += 1
                        st.session_state.total_tokens_generated += len(bot_response.split())
                        
                        # Reset automatique après 10 échanges
                        if st.session_state.conversation_count >= 10:
                            st.session_state.chat_history_ids = None
                            st.session_state.conversation_count = 0
                        
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Affichage de l'historique avec les éléments de chat natifs
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input de chat natif Streamlit
    if prompt := st.chat_input("💭 Tapez votre message ici..."):
        # Validation de l'entrée
        if len(prompt.strip()) == 0:
            st.warning("⚠️ Veuillez saisir un message non vide.")
            return
        
        if len(prompt) > 500:
            st.warning("⚠️ Message trop long (max 500 caractères).")
            return
        
        # Affichage du message utilisateur
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Génération et affichage de la réponse
        with st.chat_message("assistant"):
            start_time = time.time()
            
            with st.spinner("🤔 Réflexion en cours..."):
                bot_response, new_chat_history_ids = generate_response(
                    prompt, 
                    st.session_state.tokenizer, 
                    st.session_state.model, 
                    st.session_state.chat_history_ids
                )
            
            generation_time = time.time() - start_time
            
            # Affichage avec streaming
            response = st.write_stream(response_generator(bot_response))
            
            # Métrique de performance (optionnel, masqué par défaut)
            if st.button("📈 Détails génération", key="perf_details"):
                st.caption(f"⏱️ Généré en {generation_time:.2f}s • {len(bot_response.split())} mots")
        
        # Mise à jour de l'état
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.chat_history_ids = new_chat_history_ids
        st.session_state.conversation_count += 1
        st.session_state.total_tokens_generated += len(response.split())
        
        # Reset automatique après 10 échanges avec notification
        if st.session_state.conversation_count >= 10:
            st.session_state.chat_history_ids = None
            st.session_state.conversation_count = 0
            st.info("🔄 Contexte automatiquement réinitialisé pour optimiser les performances.")
    
    # Message de bienvenue si pas d'historique
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-card">
            <h2>👋 Bienvenue dans votre ChatBot IA !</h2>
            <p>
                Je suis votre assistant IA basé sur DialoGPT-medium, entraîné sur des conversations naturelles.<br>
                Commencez une conversation en tapant un message ci-dessous !
            </p>
            <p style="font-size: 0.9rem; color: #888;">
                💡 Utilisez les suggestions dans la barre latérale pour découvrir mes capacités
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer avec informations techniques détaillées
    with st.expander("🔧 Informations Techniques Complètes"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🏗️ Architecture Modèle**
            - Modèle: DialoGPT-medium
            - Paramètres: 345M
            - Couches: 24 layers
            - Têtes d'attention: 16
            - Embedding: 1024
            - Vocabulaire: 50,257 tokens
            """)
        
        with col2:
            st.markdown("""
            **⚙️ Paramètres Génération**
            - Temperature: 0.8
            - Top-k: 50
            - Top-p: 0.95
            - Max length: 1000 tokens
            - Repetition penalty: 1.15
            - No repeat n-gram: 3
            """)
        
        with col3:
            st.markdown("""
            **📊 Dataset & Performance**
            - Source: DailyDialog
            - Dialogues: 13,118
            - Tours moyens: 7.9
            - Domaines: Conversations quotidiennes
            - Optimisé: CPU Streamlit Cloud
            - Auto-reset: 10 échanges
            """)
        
        # Informations système
        st.markdown("**💻 Informations Système**")
        system_info = {
            "PyTorch Version": torch.__version__,
            "Device": "CPU (Streamlit Cloud optimized)",
            "Memory Management": "Auto garbage collection",
            "Streaming": "Real-time word streaming",
            "Cache": "Model cached with Streamlit @cache_resource"
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")

if __name__ == "__main__":
    main()
