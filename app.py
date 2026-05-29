import os
import time
import random
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import TextPreprocessor
from src.model import ToxicityClassifier, DeepLearningToxicityClassifier
from src.moderator import ContentModerator
from train import run_training_pipeline

# Helper to serve decoupled frontend assets
def load_frontend_asset(filename: str) -> str:
    filepath = os.path.join("frontend", filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return ""

# ---------------------------------------------------------
# Page Configurations & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="NEUTRA-MOD // Toxic Comment Filtering",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS stylesheet
css_content = load_frontend_asset("style.css")
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Load external JavaScript validation checks
js_content = load_frontend_asset("script.js")
if js_content:
    st.markdown(f"<script>{js_content}</script>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Moderation System State
# ---------------------------------------------------------
MODEL_PATH = "models/toxicity_classifier.joblib"

@st.cache_resource
def load_classical_model():
    if os.path.exists(MODEL_PATH):
        return ToxicityClassifier.load(MODEL_PATH)
    return None

@st.cache_resource
def load_deep_learning_model():
    # Will lazily load unitary/toxic-bert from HuggingFace
    clf = DeepLearningToxicityClassifier(model_name='unitary/toxic-bert')
    return clf

# Initialize session states
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [
        {"username": "Cyber_Striker", "text": "let's push outer tower", "action": "ALLOW", "severity": 0.2, "timestamp": time.time() - 30},
        {"username": "ProGamer_99", "text": "ggwp guys close one!", "action": "ALLOW", "severity": 0.1, "timestamp": time.time() - 25},
        {"username": "RageMonster", "text": "stfu u absolute garbage", "action": "WARN", "severity": 1.8, "timestamp": time.time() - 20},
        {"username": "Moderator_Bot", "text": "[SYSTEM ALERT] RageMonster has been WARNED. Reason: Please keep chat respectful.", "action": "SYSTEM", "severity": 0.0, "timestamp": time.time() - 19},
        {"username": "Cyber_Striker", "text": "wow, nice counter rage lol", "action": "ALLOW", "severity": 0.4, "timestamp": time.time() - 10}
    ]

if 'user_action_history' not in st.session_state:
    st.session_state.user_action_history = []

if 'active_players' not in st.session_state:
    st.session_state.active_players = {
        "Cyber_Striker": "Active",
        "ProGamer_99": "Active",
        "RageMonster": "Warned",
        "ShadowNinja": "Active",
        "Toxic_Avenger": "Active"
    }

# Dynamic player message presets for simulation
SIMULATED_MESSAGES = [
    ("ShadowNinja", "i need some backup on the left portal", "ALLOW"),
    ("ProGamer_99", "buy aimbot at hacklords.com for 5 dollars!", "WARN"),
    ("Toxic_Avenger", "kys garbage bot uninstall life", "BAN"),
    ("RageMonster", "fuck this lag and fuck this stupid teammate", "MUTE"),
    ("ShadowNinja", "we can still secure the victory, hold defense!", "ALLOW"),
    ("Cyber_Striker", "nice heal thx so much", "ALLOW")
]

# ---------------------------------------------------------
# Sidebar Panel - Configuration & Model Control
# ---------------------------------------------------------
st.sidebar.markdown("<h2 style='color:#FF1744;'>🛡️ CONTROL CENTER</h2>", unsafe_allow_html=True)
st.sidebar.write("Configure the active AI model parameters and rule thresholds.")

# 1. Model Backend Selector
model_type = st.sidebar.radio(
    "AI Model Backend",
    ["⚡ Classical ML (Fast TF-IDF)", "🧠 Deep Learning (DistilBERT)"],
    help="Classical ML runs immediately offline in under 5ms. Deep Learning loads a state-of-the-art transformer model (unitary/toxic-bert) for ultimate accuracy."
)

# Load Classifier
model = None
is_model_trained = os.path.exists(MODEL_PATH)

if model_type.startswith("⚡"):
    model = load_classical_model()
else:
    with st.sidebar.status("Loading Deep Learning Model (unitary/toxic-bert)...", expanded=False) as dl_status:
        try:
            model = load_deep_learning_model()
            dl_status.update(label="Deep Learning Model Loaded!", state="complete")
        except Exception as e:
            st.sidebar.error(f"Failed to load Deep Learning model: {e}")

# 2. Rule Customizers
st.sidebar.markdown("<hr style='border-color: #1F2937;'/>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚙️ Threshold Tuning (0.0 to 5.0)")
warn_thresh = st.sidebar.slider("⚠️ Warn Threshold", 0.5, 2.0, 1.0, 0.1)
mute_thresh = st.sidebar.slider("🔇 Mute Threshold", 2.0, 3.8, 2.5, 0.1)
ban_thresh = st.sidebar.slider("🚫 Ban Threshold", 3.5, 5.0, 4.0, 0.1)

# 3. Model Training Action (Integrated Trigger)
st.sidebar.markdown("<hr style='border-color: #1F2937;'/>", unsafe_allow_html=True)
st.sidebar.markdown("### ⚡ Model Training")
if not is_model_trained:
    st.sidebar.warning("Trained model file `models/toxicity_classifier.joblib` not found. Live predictions will be disabled until trained.")
    if st.sidebar.button("⚙️ Train Classical Model Now"):
        with st.spinner("Training ML pipeline and optimizing thresholds (takes ~5s)..."):
            try:
                run_training_pipeline()
                st.cache_resource.clear() # clear cache to reload model
                st.success("Model trained successfully! Loading pipeline...")
                st.rerun()
            except Exception as e:
                st.error(f"Error during training: {e}")
else:
    st.sidebar.success("Classical Model File: Detected & Loaded")
    if st.sidebar.button("🔄 Retrain ML Classifier"):
        with st.spinner("Retraining model pipeline..."):
            try:
                run_training_pipeline()
                st.cache_resource.clear()
                st.success("Model successfully retrained!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------------------------------------
# Main Page View
# ---------------------------------------------------------
header_html = load_frontend_asset("header.html")
if header_html:
    st.markdown(header_html, unsafe_allow_html=True)

# Tabs
tab_simulator, tab_analytics, tab_policy = st.tabs([
    "🕹️ Live Chat Simulator", 
    "📊 Admin Analytics Dashboard", 
    "📜 Mitigation Strategies"
])

# Initialize Content Moderator wrapper
moderator = ContentModerator(classifier=model)
moderator.threshold_warn = warn_thresh
moderator.threshold_mute = mute_thresh
moderator.threshold_ban = ban_thresh

# ---------------------------------------------------------
# TAB 1: Live Chat Simulator
# ---------------------------------------------------------
with tab_simulator:
    col_chat, col_player_status = st.columns([3, 1])
    
    with col_chat:
        chat_header = load_frontend_asset("chat_header.html")
        if chat_header:
            st.markdown(chat_header, unsafe_allow_html=True)
        
        # Simulation controls
        sim_cols = st.columns([4, 1])
        with sim_cols[1]:
            if st.button("🤖 Sim Player Message"):
                # Pick a random preset
                user, msg_text, _ = random.choice(SIMULATED_MESSAGES)
                
                # Fetch user history for rate-limit / spam checking
                user_hist = [m for m in st.session_state.chat_history if m.get('username') == user]
                
                # Run moderation
                report = moderator.analyze_message(msg_text, username=user, user_history=user_hist)
                
                # Update player status
                if report['action'] == "WARN":
                    st.session_state.active_players[user] = "Warned"
                elif report['action'] == "MUTE":
                    st.session_state.active_players[user] = "Muted"
                elif report['action'] == "BAN":
                    st.session_state.active_players[user] = "Banned"
                
                # Add to history
                st.session_state.chat_history.append({
                    "username": user,
                    "text": report['redacted_text'],
                    "original": report['original_text'],
                    "action": report['action'],
                    "severity": report['severity_score'],
                    "timestamp": time.time()
                })
                
                # If toxic, insert a System Bot intervention message
                if report['action'] in ["WARN", "MUTE", "BAN"]:
                    st.session_state.chat_history.append({
                        "username": "Moderator_Bot",
                        "text": f"system alert: {user} mitigation dispatch [{report['action']}] - reason: {report['action_reason']}",
                        "action": "SYSTEM",
                        "severity": 0.0,
                        "timestamp": time.time()
                    })
                    
                st.rerun()
                
        with sim_cols[0]:
            st.markdown("<div style='font-size: 0.75rem; color: #6B7280; font-family: \"JetBrains Mono\", monospace;'>// click trigger simulation to inject dynamic streaming context</div>", unsafe_allow_html=True)
            
        # Draw the scrollable chat
        chat_html = "<div class='chat-container' style='max-height: 400px; overflow-y: auto; padding: 15px 0px; border-top: 1px solid rgba(128,128,128,0.15); border-bottom: 1px solid rgba(128,128,128,0.15); background-color: transparent; margin-bottom: 15px; margin-top: 10px;'>"
        for msg in st.session_state.chat_history[-15:]:  # Display last 15 messages
            username = msg["username"]
            text = msg["text"]
            action = msg["action"]
            
            if username == "Moderator_Bot":
                chat_html += f"<div style='margin-bottom: 12px; padding: 10px 0px; border-bottom: 1px dashed rgba(239, 68, 68, 0.15); color: #EF4444; font-family: \"JetBrains Mono\", monospace; font-size: 0.8rem;'>// {text}</div>"
            else:
                badge = ""
                if action == "WARN":
                    badge = "<span style='font-family: \"JetBrains Mono\", monospace; color: #F59E0B; font-size: 0.75rem; font-weight: 500; margin-left: 6px;'>// WARN</span>"
                elif action == "MUTE":
                    badge = "<span style='font-family: \"JetBrains Mono\", monospace; color: #3B82F6; font-size: 0.75rem; font-weight: 500; margin-left: 6px;'>// MUTE</span>"
                elif action == "BAN":
                    badge = "<span style='font-family: \"JetBrains Mono\", monospace; color: #EF4444; font-size: 0.75rem; font-weight: 500; margin-left: 6px;'>// BAN</span>"
                    
                # Format timestamp
                time_str = time.strftime('%H:%M:%S', time.localtime(msg["timestamp"]))
                chat_html += f"<div class='chat-dashed-line' style='margin-bottom: 12px; padding-bottom: 8px; font-family: \"Inter\", sans-serif; font-size: 0.9rem;'><span style='font-family: \"JetBrains Mono\", monospace; color: #6B7280; opacity: 0.8; font-size: 0.75rem; margin-right: 8px;'>[{time_str}]</span> <span class='text-high-contrast' style='font-weight: 600;'>{username}</span>{badge} <span class='text-muted-contrast' style='margin-left: 8px;'>{text}</span></div>"
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)
        
        # User input interactive chat box
        user_input = st.chat_input("Join the lobby chat... (Type something to test the AI moderation response)")
        if user_input:
            # Check player status of the tester
            tester_name = "You_The_Player"
            
            # Moderation report
            user_hist = [m for m in st.session_state.chat_history if m.get('username') == tester_name]
            report = moderator.analyze_message(user_input, username=tester_name, user_history=user_hist)
            
            # Show live analyzer card at top of submission
            st.session_state.chat_history.append({
                "username": tester_name,
                "text": report['redacted_text'],
                "original": report['original_text'],
                "action": report['action'],
                "severity": report['severity_score'],
                "timestamp": time.time()
            })
            
            if report['action'] in ["WARN", "MUTE", "BAN"]:
                st.session_state.chat_history.append({
                    "username": "Moderator_Bot",
                    "text": f"system alert: {tester_name} mitigation dispatch [{report['action']}] - reason: {report['action_reason']}",
                    "action": "SYSTEM",
                    "severity": 0.0,
                    "timestamp": time.time()
                })
                
            # Log player actions for analytics tab
            st.session_state.user_action_history.append(report)
            st.rerun()
            
    with col_player_status:
        directory_header = load_frontend_asset("directory_header.html")
        if directory_header:
            st.markdown(directory_header, unsafe_allow_html=True)
        
        for p, status in st.session_state.active_players.items():
            color = "#10B981"  # Active (Green)
            dot_class = "dot-active"
            if status == "Warned":
                color = "#F59E0B"  # Yellow
                dot_class = "dot-warned"
            elif status == "Muted":
                color = "#3B82F6"  # Blue
                dot_class = "dot-muted"
            elif status == "Banned":
                color = "#EF4444"  # Red
                dot_class = "dot-banned"
                
            st.markdown(f"""
            <div style='background-color: transparent; padding: 12px 0px; border-bottom: 1px solid #1A1C20; display: flex; justify-content: space-between; align-items: center;'>
                <span style='font-family: "Inter", sans-serif; font-size: 0.9rem; font-weight: 500; color: #FFFFFF;'>{p}</span>
                <span style='font-family: "JetBrains Mono", monospace; color: {color}; font-size: 0.75rem; font-weight: 500; display: flex; align-items: center; gap: 8px;'>
                    <span class='{dot_class}'></span>{status.upper()}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
    # Dynamic live detailed output display of the LAST message analyzed
    last_user_msgs = [m for m in st.session_state.chat_history if m["username"] != "Moderator_Bot"]
    if last_user_msgs:
        last_msg = last_user_msgs[-1]
        
        pipeline_metrics_header = load_frontend_asset("pipeline_metrics_header.html")
        if pipeline_metrics_header:
            st.markdown(pipeline_metrics_header, unsafe_allow_html=True)
        
        # If model is loaded, we can re-analyze or fetch details
        raw_txt = last_msg.get("original", last_msg["text"])
        user = last_msg["username"]
        
        # Re-run for visual analytics display
        rep = moderator.analyze_message(raw_txt, username=user)
        
        col_m1, col_m2 = st.columns([1, 1])
        
        with col_m1:
            # Severity color
            sev = rep['severity_score']
            color = "#10B981"
            if rep['action'] == "WARN":
                color = "#F59E0B"
            elif rep['action'] == "MUTE":
                color = "#3B82F6"
            elif rep['action'] == "BAN":
                color = "#EF4444"
                
            st.markdown(f"""
            <div class='border-divider' style='background-color: transparent; padding: 12px 0px; display: flex; justify-content: space-between; align-items: center; font-family: "Inter", sans-serif; font-size: 0.9rem;'>
                <span class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.75rem; text-transform: uppercase;'>Input message</span>
                <span class='text-high-contrast' style='font-weight: 500;'>"{raw_txt}"</span>
            </div>
            <div class='border-divider' style='background-color: transparent; padding: 12px 0px; display: flex; justify-content: space-between; align-items: center; font-family: "Inter", sans-serif; font-size: 0.9rem;'>
                <span class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.75rem; text-transform: uppercase;'>Normalized text</span>
                <span class='text-muted-contrast' style='font-weight: 500;'>"{rep['preprocessed_text']}"</span>
            </div>
            <div class='border-divider' style='background-color: transparent; padding: 12px 0px; display: flex; justify-content: space-between; align-items: center; font-family: "Inter", sans-serif; font-size: 0.9rem;'>
                <span class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.75rem; text-transform: uppercase;'>Filtration output</span>
                <span style='color: #10B981; font-weight: 500;'>"{rep['redacted_text']}"</span>
            </div>
            
            <div class='card-container' style='border-top: 4px solid {color} !important; margin-top: 20px; padding: 20px;'>
                <div class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;'>Computed Severity Index</div>
                <div class='text-high-contrast' style='font-size: 2.2rem; font-weight: 700; margin: 8px 0; font-family: "JetBrains Mono", monospace;'>{sev:.2f} <span class='text-dim-contrast' style='font-size: 1rem;'>/ 5.00</span></div>
                <div class='text-muted-contrast' style='font-family: "Inter", sans-serif; font-size: 0.85rem;'>Recommendation: <strong class='text-high-contrast'>{rep['action']}</strong> // {rep['action_reason']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_m2:
            st.markdown("<div class='text-dim-contrast' style='font-family: \"JetBrains Mono\", monospace; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px;'>Predicted Multi-label Class Probabilities</div>", unsafe_allow_html=True)
            
            categories_disp = {
                'toxic': 'Toxic (General)',
                'severe_toxic': 'Severe Toxic',
                'obscene': 'Profanity / Obscene',
                'threat': 'Threat / Violent',
                'insult': 'Insulting',
                'identity_hate': 'Hate Speech / Slur',
                'spam': 'Spam / Advertising'
            }
            
            for cat, display_name in categories_disp.items():
                prob = rep['probabilities'].get(cat, 0.0)
                flag = rep['flags'].get(cat, 0)
                
                glow_color = "#3B82F6"
                flag_badge = ""
                if flag == 1:
                    if cat in ['severe_toxic', 'identity_hate']:
                        glow_color = "#EF4444"
                        flag_badge = "<span style='font-family: \"JetBrains Mono\", monospace; color: #EF4444; font-size: 0.65rem; font-weight: bold; margin-left: 8px;'>// SEVERE CRITICAL</span>"
                    elif cat in ['threat', 'obscene']:
                        glow_color = "#F59E0B"
                        flag_badge = "<span style='font-family: \"JetBrains Mono\", monospace; color: #F59E0B; font-size: 0.65rem; font-weight: bold; margin-left: 8px;'>// WARNING</span>"
                    else:
                        glow_color = "#10B981"
                        flag_badge = "<span style='font-family: \"JetBrains Mono\", monospace; color: #10B981; font-size: 0.65rem; font-weight: bold; margin-left: 8px;'>// TOXIC</span>"
                        
                pct = prob * 100
                st.markdown(f"""
                <div class='border-divider' style='background-color: transparent; padding: 10px 0px; margin-bottom: 10px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;'>
                        <span class='text-high-contrast' style='font-weight: 500; font-size: 0.85rem; display: flex; align-items: center;'>{display_name}{flag_badge}</span>
                        <span style='font-family: "JetBrains Mono", monospace; color: {glow_color}; font-weight: 700; font-size: 0.85rem;'>{prob:.1%}</span>
                    </div>
                    <div style='background-color: rgba(128, 128, 128, 0.12); height: 3px; border-radius: 1.5px; overflow: hidden;'>
                        <div style='background-color: {glow_color}; width: {pct:.2f}%; height: 100%;'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 2: Admin Analytics Dashboard
# ---------------------------------------------------------
with tab_analytics:
    analytics_header = load_frontend_asset("analytics_header.html")
    if analytics_header:
        st.markdown(analytics_header, unsafe_allow_html=True)
    
    # Summary Metrics Row
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    
    # Calculate stats from active session chat history
    all_chat = st.session_state.chat_history
    user_msgs = [m for m in all_chat if m["username"] != "Moderator_Bot"]
    sys_msgs = [m for m in all_chat if m["username"] == "Moderator_Bot"]
    
    total_messages = len(user_msgs)
    warns_issued = sum(1 for m in sys_msgs if "mitigation dispatch [WARN]" in m["text"])
    mutes_issued = sum(1 for m in sys_msgs if "mitigation dispatch [MUTE]" in m["text"])
    bans_issued = sum(1 for m in sys_msgs if "mitigation dispatch [BAN]" in m["text"])
    
    with m_col1:
        st.markdown(f"""
        <div class='border-divider' style='background-color: transparent; padding: 15px 0px;'>
            <div class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;'>Total Chats Analyzed</div>
            <div class='text-high-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 1.8rem; font-weight: 700;'>// {total_messages}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class='border-divider' style='background-color: transparent; padding: 15px 0px;'>
            <div class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;'>Warnings Dispatched</div>
            <div style='font-family: "JetBrains Mono", monospace; font-size: 1.8rem; font-weight: 700; color: #F59E0B;'>// {warns_issued}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class='border-divider' style='background-color: transparent; padding: 15px 0px;'>
            <div class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;'>Player Mutings</div>
            <div style='font-family: "JetBrains Mono", monospace; font-size: 1.8rem; font-weight: 700; color: #3B82F6;'>// {mutes_issued}</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class='border-divider' style='background-color: transparent; padding: 15px 0px;'>
            <div class='text-dim-contrast' style='font-family: "JetBrains Mono", monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px;'>Permanent Bans</div>
            <div style='font-family: "JetBrains Mono", monospace; font-size: 1.8rem; font-weight: 700; color: #EF4444;'>// {bans_issued}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-bottom: 25px;'></div>", unsafe_allow_html=True)
    
    # Split charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.markdown("#### Live Severity Scores Timeline")
        if user_msgs:
            # Line plot of severity scores
            sevs = [m.get("severity", 0.0) for m in user_msgs]
            indices = list(range(1, len(sevs) + 1))
            
            fig, ax = plt.subplots(figsize=(6, 3.5))
            plt.style.use('dark_background')
            fig.patch.set_facecolor('#0E1117')
            ax.set_facecolor('#1A1C23')
            
            ax.plot(indices, sevs, marker='o', color='#FF1744', linewidth=2, label='Severity')
            # Add threshold indicator lines
            ax.axhline(y=warn_thresh, color='#FFC107', linestyle='--', alpha=0.5, label='Warn Threshold')
            ax.axhline(y=mute_thresh, color='#FF9100', linestyle='--', alpha=0.5, label='Mute Threshold')
            ax.axhline(y=ban_thresh, color='#FF1744', linestyle='--', alpha=0.5, label='Ban Threshold')
            
            ax.set_xlabel('Sequence of Message', color='white')
            ax.set_ylabel('Severity Score (0-5)', color='white')
            ax.tick_params(colors='white')
            ax.set_ylim(0, 5.2)
            ax.legend(facecolor='#1A1C23', labelcolor='white', framealpha=0.6, fontsize='small')
            
            st.pyplot(fig)
        else:
            st.info("No messages in queue to chart timeline.")
            
    with chart_col2:
        st.markdown("#### Category Infraction Distribution")
        if st.session_state.user_action_history:
            # Sum up flags from history
            hist_df = pd.DataFrame([h['flags'] for h in st.session_state.user_action_history])
            
            if not hist_df.empty:
                sum_flags = hist_df.sum().reset_index()
                sum_flags.columns = ['Category', 'Count']
                
                fig, ax = plt.subplots(figsize=(6, 3.5))
                plt.style.use('dark_background')
                fig.patch.set_facecolor('#0E1117')
                ax.set_facecolor('#1A1C23')
                
                sns.barplot(data=sum_flags, x='Category', y='Count', hue='Category', palette='cool', legend=False, ax=ax)
                ax.set_xlabel('Toxicity Class', color='white')
                ax.set_ylabel('Flag Count', color='white')
                ax.tick_params(colors='white')
                plt.xticks(rotation=45)
                
                st.pyplot(fig)
        else:
            # Standard pre-validation check
            # Load and display saved validation performance charts if they exist
            if os.path.exists("artifacts/evaluation_metrics.png"):
                st.image("artifacts/evaluation_metrics.png", caption="Model Offline Validation Performance Matrix")
            else:
                st.info("Simulate more custom player messages in Tab 1 to dynamically compile active infraction distribution charts here.")
                
    # Detailed Moderator Logs Table
    st.markdown("### 📝 Detailed AI Moderator Session Log")
    if st.session_state.user_action_history:
        log_data = []
        for rep in st.session_state.user_action_history:
            log_data.append({
                "User": rep['username'],
                "Message": rep['original_text'],
                "Severity": rep['severity_score'],
                "Action": rep['action'],
                "Reason": rep['action_reason'],
                "Latency": f"{rep['latency_ms']} ms"
            })
        st.dataframe(pd.DataFrame(log_data), use_container_width=True)
    else:
        st.caption("Interact with the live chat simulator in Tab 1 to populate detailed system reports.")

# ---------------------------------------------------------
# TAB 3: Mitigation Strategies Policy
# ---------------------------------------------------------
with tab_policy:
    policy_header = load_frontend_asset("policy_header.html")
    if policy_header:
        st.markdown(policy_header, unsafe_allow_html=True)
    
    # Draw table
    col_p_data = {
        "Severity Score Range": ["0.00 – 1.00 (Safe)", "1.00 – 2.50 (Mild Hazard)", "2.50 – 4.00 (Medium Hazard)", "4.00 – 5.00 (Extreme Hazard)"],
        "Violations Detected": ["No violations / Standard gaming conversation", "Spam links, minor profanity, low-level insults", "Excessive obscenities, slurs, targeted player abuse", "Hate speech, self-harm incitement (KYS), real-life threats"],
        "Moderator Bot Action": ["**ALLOW** (No action taken)", "**WARN** (Standard chat warning prompt)", "**MUTE** (Remove text and mute mic/chat for 5 minutes)", "**BAN** (Terminate user session and black-list IP)"],
        "Mitigation Goal": ["Facilitate active gameplay", "Provide learning guardrails for players", "De-escalate immediate arguments and shield server", "Defend platform integrity and legal safety"]
    }
    st.table(pd.DataFrame(col_p_data))
    
    key_features = load_frontend_asset("key_features.html")
    if key_features:
        st.markdown(key_features, unsafe_allow_html=True)
