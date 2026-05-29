# ==========================================
# Author: Ansh Jaiswal
# Neutra-Mod Server Chat & Filtration Log Simulator
# ==========================================
import os
import time
import random
import pandas as pd
from src.preprocessing import TextPreprocessor
from src.moderator import ContentModerator
from src.model import ToxicityClassifier

def run_log_generation():
    print("==================================================")
    print("📋 GENERATING SAMPLE CHAT LOGS AND DETECTION LOGS")
    print("==================================================\n")
    
    # 1. Load trained classifier if available, else use a placeholder classifier for fallback
    # to guarantee the script always runs without crashes
    model_path = "models/toxicity_classifier.joblib"
    clf = None
    if os.path.exists(model_path):
        try:
            clf = ToxicityClassifier.load(model_path)
            print("Loaded trained model for log generation.")
        except Exception as e:
            print(f"Error loading model, using mock predictions: {e}")
    else:
        print("Trained model not found. Generating logs using simulated scoring.")
        
    moderator = ContentModerator(classifier=clf)
    
    # 2. Define a list of players
    players = [
        "Sniper_Pro", "Raging_Bull", "Shadow_Ninja", "LootGoblin", "GamerGirl_V",
        "Warlock_Elite", "Cyber_Cop", "Chatty_Cathy", "NoobSlayer", "AdBot_999"
    ]
    
    # 3. Define chat messages representing a standard evening stream
    chat_database = [
        # Clean gaming comments
        ("lets take the dragon first guys", False),
        ("gg wp, great match!", False),
        ("nice play sniper!", False),
        ("can someone share some ammo?", False),
        ("lagging so hard right now, ping is 300ms", False),
        ("sorry, that was my fault", False),
        ("no problem, we got this next round", False),
        ("who wants to team up for ranked?", False),
        ("sweet shot shadow!", False),
        ("gg guys, well played", False),
        ("i'm going to base to heal up", False),
        ("place a ward in the brush", False),
        ("that was an amazing ultimate!", False),
        ("anyone got a mic?", False),
        ("glhf", False),
        
        # Spam
        ("FREE GOLD! CLICK HERE ---> WWW.GOLDSELLER4FREE.XYZ NO BAN FAST DELIVERY", True),
        ("Visit discountgamingkeys.org for 90% off all retail releases! Promo: SAVE90", True),
        ("boost your rank to global elite cheap! discord.gg/cheatboosters", True),
        ("Earn 100 dollars a day playing this game! click http://workfromhome.click", True),
        
        # Obscene / Insults
        ("stfu u absolute garbage noob", False),
        ("uninstall the game you blind retard", False),
        ("my teammates are complete assholes", False),
        ("you are a brainless piece of shit", False),
        ("fuck this laggy garbage game", False),
        ("get good trash kid", False),
        ("stop stealing my kills you dick", False),
        
        # Severe / Threats / Hate
        ("kys get off our server you subhuman immigrant", False),
        ("i will find where you live and slit your throat you cheat", False),
        ("hope your whole family dies in a fire", False),
        ("fucking racial slurs, report this guy", False),
        ("go kill yourself, you are useless", False),
        ("i am going to track your IP address and swat your house", False)
    ]
    
    raw_logs = []
    detection_logs = []
    
    current_time = time.time()
    
    print("Simulating 60 gaming chat messages and running through moderator...")
    
    # Generate 60 log messages
    for i in range(60):
        # Determine player and message type
        player = random.choice(players)
        
        # AdBot always sends advertising spam
        if player == "AdBot_999":
            msg_text = random.choice([m[0] for m in chat_database if m[1] is True])
        else:
            # Other players send a mix (mostly clean, some toxic)
            if random.random() < 0.7:
                msg_text = random.choice([m[0] for m in chat_database if m[1] is False and "stfu" not in m[0] and "kys" not in m[0] and "fuck" not in m[0] and "slit" not in m[0]])
            else:
                msg_text = random.choice([m[0] for m in chat_database if "stfu" in m[0] or "kys" in m[0] or "fuck" in m[0] or "slit" in m[0] or m[1] is True])
                
        # Space messages out by a few seconds
        timestamp = current_time - (60 - i) * 8.5
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
        
        # 1. Append to raw chat logs
        raw_logs.append({
            "timestamp": time_str,
            "username": player,
            "message": msg_text
        })
        
        # 2. Append to moderation detection logs
        # Simulate user history for rate limits
        hist = [{"timestamp": current_time - (60 - k) * 8.5} for k in range(max(0, i-3), i)]
        
        # Run moderator
        report = moderator.analyze_message(msg_text, username=player, user_history=hist)
        
        # Compile active prediction categories for cleaner logging
        active_cats = [k for k, v in report['flags'].items() if v == 1 and k != 'spam']
        active_cats_str = ", ".join(active_cats) if active_cats else "None"
        
        detection_logs.append({
            "timestamp": time_str,
            "username": player,
            "message": msg_text,
            "preprocessed_message": report['preprocessed_text'],
            "is_spam": "Yes" if report['is_spam'] else "No",
            "detected_toxicity_categories": active_cats_str,
            "severity_score": report['severity_score'],
            "moderation_action": report['action'],
            "action_reason": report['action_reason'],
            "redacted_message": report['redacted_text'],
            "latency_ms": report['latency_ms']
        })
        
    # Write to CSV
    os.makedirs("logs", exist_ok=True)
    
    raw_df = pd.DataFrame(raw_logs)
    raw_df.to_csv("logs/sample_chat_logs.csv", index=False)
    print("Saved raw chat feed to logs/sample_chat_logs.csv")
    
    detect_df = pd.DataFrame(detection_logs)
    detect_df.to_csv("logs/moderation_detection_logs.csv", index=False)
    print("Saved detailed moderator detection feed to logs/moderation_detection_logs.csv")
    
    print("\n==============================================")
    print("📋 LOG GENERATION SUCCESSFULLY COMPLETED")
    print("==============================================")

if __name__ == "__main__":
    run_log_generation()
