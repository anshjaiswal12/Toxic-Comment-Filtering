# ==========================================
# Author: Ansh Jaiswal
# Neutra-Mod ContentModerator & SpamDetector Engines
# ==========================================
import re
import time
from typing import Dict, List, Tuple, Any
from src.preprocessing import TextPreprocessor

class SpamDetector:
    def __init__(self):
        # Match common link structures
        self.url_pattern = re.compile(
            r'(https?://\S+|www\.\S+|\b[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\/\S*)'
        )
        # Check for character repeating 4+ times (e.g., "soooooloo")
        self.char_repeat_pattern = re.compile(r'([a-zA-Z0-9])\1{4,}')
        
    def check_spam(self, text: str, user_history: List[Dict[str, Any]] = None) -> Tuple[bool, float, str]:
        """
        Analyze text for structural spam signatures.
        Returns:
            is_spam: bool
            spam_score: float (0.0 to 1.0)
            reason: str
        """
        if not text or not isinstance(text, str):
            return False, 0.0, ""
            
        score = 0.0
        reasons = []
        
        # 1. URL/Link detection (High threat spam in gaming chat)
        if self.url_pattern.search(text):
            score += 0.8
            reasons.append("Contains links/advertisements")
            
        # 2. Excessive Caps Lock (Shouting/Spamming)
        letters = [c for c in text if c.isalpha()]
        if len(letters) >= 6:
            caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
            if caps_ratio > 0.85:
                score += 0.4
                reasons.append("Excessive capitalization")
                
        # 3. Excessive Character Repetition (e.g., "heeeeeeeeeeeeeyyyyy")
        if self.char_repeat_pattern.search(text):
            score += 0.3
            reasons.append("Character repetition")
            
        # 4. Excessive Word Repetition (e.g., "buy gold buy gold buy gold")
        words = text.lower().split()
        if len(words) >= 6:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:
                score += 0.5
                reasons.append("Repetitive word phrases")
                
        # 5. Rate limit / Message frequency spam (Requires history)
        if user_history and len(user_history) >= 3:
            current_time = time.time()
            # Get last 3 message times
            timestamps = [msg.get('timestamp', 0) for msg in user_history[-3:]]
            if len(timestamps) == 3:
                time_diff = current_time - timestamps[0]
                if time_diff < 2.5: # 3 messages in under 2.5 seconds
                    score += 0.6
                    reasons.append("Rapid message rate")
                    
        # Final decision
        is_spam = score >= 0.5
        reason = ", ".join(reasons) if reasons else "Normal"
        
        return is_spam, min(score, 1.0), reason


class ContentModerator:
    def __init__(self, classifier=None):
        self.preprocessor = TextPreprocessor()
        self.spam_detector = SpamDetector()
        self.classifier = classifier
        
        # Base list of bad words for local character replacement/redaction
        self.profane_words = [
            "fuck", "shit", "asshole", "bitch", "cunt", "nigger", "faggot", 
            "dick", "pussy", "retard", "whore", "slut", "bastard", "kys", 
            "stfu", "dumbass", "motherfuck"
        ]
        # Precompile regex for bad words
        self.profane_patterns = [
            re.compile(rf"\b{word}[a-z]*\b", re.IGNORECASE) for word in self.profane_words
        ]
        
        # Action threshold configuration
        self.threshold_warn = 1.0
        self.threshold_mute = 2.5
        self.threshold_ban = 4.0
        
    def set_classifier(self, classifier):
        self.classifier = classifier
        
    def redact_message(self, text: str, labels_flagged: Dict[str, int]) -> str:
        """Mask profane words in the text if toxicity is detected."""
        # If no toxicity categories are flagged (except spam), keep text original
        toxic_flagged = sum(v for k, v in labels_flagged.items() if k != 'spam')
        if toxic_flagged == 0:
            return text
            
        redacted = text
        for pattern in self.profane_patterns:
            # Replace profane word with asterisks corresponding to word length
            def replacer(match):
                return "*" * len(match.group(0))
            redacted = pattern.sub(replacer, redacted)
            
        # Additional cleanup of known bad-words found inside the preprocessor
        return redacted
        
    def calculate_severity(self, probs: Dict[str, float], is_spam: bool, spam_score: float) -> float:
        """
        Calculate a 0 to 5 severity score.
        Formula aggregates probabilities of active toxicity classes with custom weights.
        """
        # Class weights
        weights = {
            'severe_toxic': 2.0,
            'identity_hate': 2.0, # Hate speech
            'threat': 1.8,
            'toxic': 1.0,
            'insult': 1.0,
            'obscene': 1.0 # Profanity
        }
        
        weighted_sum = 0.0
        max_possible = 0.0
        
        # Sum active probability weights
        for col, prob in probs.items():
            w = weights.get(col, 1.0)
            weighted_sum += prob * w
            max_possible += w
            
        # Add spam contribution
        if is_spam:
            weighted_sum += spam_score * 1.5
            max_possible += 1.5
            
        if max_possible == 0:
            return 0.0
            
        # Scale to 0 - 5 range
        severity = (weighted_sum / max_possible) * 5.0
        
        # Amplify if extreme flags are high confidence
        highest_severe = max(probs.get('severe_toxic', 0.0), probs.get('identity_hate', 0.0), probs.get('threat', 0.0))
        if highest_severe > 0.65:
            # Boost score to represent high hazard level
            severity = max(severity, 3.8 + (highest_severe - 0.65) * 3.4)
            
        return float(np_clip(severity, 0.0, 5.0))
        
    def get_moderation_action(self, severity_score: float) -> Tuple[str, str]:
        """Determine moderation action based on thresholds."""
        if severity_score >= self.threshold_ban:
            return "BAN", "Auto-ban triggered: Extreme toxicity / hate speech detected."
        elif severity_score >= self.threshold_mute:
            return "MUTE", "Player muted: Profanity/Insults exceeding acceptable threshold."
        elif severity_score >= self.threshold_warn:
            return "WARN", "Warning: Please keep chat respectful."
        else:
            return "ALLOW", "Safe comment."
            
    def analyze_message(self, text: str, username: str, user_history: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fully analyze a chat message.
        Returns detailed report containing clean/redacted text, probabilities, severity, and moderation actions.
        """
        start_time = time.time()
        
        # 1. Structural Spam Check (checks original text spacing/cases)
        is_spam, spam_score, spam_reason = self.spam_detector.check_spam(text, user_history)
        
        # 2. Text Preprocessing for Toxicity Classifier
        preprocessed_text = self.preprocessor.preprocess(text)
        
        # 3. Toxicity Classification
        probs = {cat: 0.0 for cat in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']}
        preds = {cat: 0 for cat in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']}
        
        if self.classifier and preprocessed_text:
            # Predict probabilities and binary labels
            prob_dicts = self.classifier.predict_proba([preprocessed_text])
            pred_dicts = self.classifier.predict([preprocessed_text])
            if prob_dicts:
                probs = prob_dicts[0]
                preds = pred_dicts[0]
                
        # Include spam in dynamic results
        probs['spam'] = spam_score
        preds['spam'] = 1 if is_spam else 0
        
        # 4. Calculate Severity Score
        # Extract toxicity classes for severity math
        toxic_probs = {k: v for k, v in probs.items() if k != 'spam'}
        severity_score = self.calculate_severity(toxic_probs, is_spam, spam_score)
        
        # 5. Determine Action and Redaction
        action, action_reason = self.get_moderation_action(severity_score)
        redacted_text = self.redact_message(text, preds)
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            'username': username,
            'original_text': text,
            'preprocessed_text': preprocessed_text,
            'redacted_text': redacted_text,
            'probabilities': probs,
            'flags': preds,
            'severity_score': round(severity_score, 2),
            'action': action,
            'action_reason': action_reason,
            'is_spam': is_spam,
            'spam_reason': spam_reason,
            'latency_ms': round(elapsed_ms, 2)
        }

def np_clip(val, min_val, max_val):
    """Local helper in case numpy is not loaded yet."""
    return max(min(val, max_val), min_val)
