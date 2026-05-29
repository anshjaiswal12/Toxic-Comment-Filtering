import re
import emoji

# Standard contractions dictionary
CONTRACTIONS = {
    "i'm": "i am", "im": "i am",
    "you're": "you are", "youre": "you are",
    "he's": "he is", "hes": "he is",
    "she's": "she is", "shes": "she is",
    "it's": "it is", "its": "it is",
    "we're": "we are", "were": "we are",
    "they're": "they are", "theyre": "they are",
    "i've": "i have", "ive": "i have",
    "you've": "you have", "youve": "you have",
    "we've": "we have", "weve": "we have",
    "they've": "they have", "theyve": "they have",
    "i'd": "i would", "id": "i would",
    "you'd": "you would", "youd": "you would",
    "he'd": "he would", "hed": "he would",
    "she'd": "she would", "shed": "she would",
    "we'd": "we would", "wed": "we would",
    "they'd": "they would", "theyd": "they would",
    "i'll": "i will", "ill": "i will",
    "you'll": "you will", "youll": "you will",
    "he'll": "he will", "hell": "he will",
    "she'll": "she will", "shell": "she will",
    "we'll": "we will", "well": "we will",
    "they'll": "they will", "theyll": "they will",
    "isn't": "is not", "isnt": "is not",
    "aren't": "are not", "arent": "are not",
    "wasn't": "was not", "wasnt": "was not",
    "weren't": "were not", "werent": "were not",
    "hasn't": "has not", "hasnt": "has not",
    "haven't": "have not", "havent": "have not",
    "hadn't": "had not", "hadnt": "had not",
    "won't": "will not", "wont": "will not",
    "wouldn't": "would not", "wouldnt": "would not",
    "don't": "do not", "dont": "do not",
    "doesn't": "does not", "doesnt": "does not",
    "didn't": "did not", "didnt": "did not",
    "can't": "cannot", "cant": "cannot",
    "cannot": "cannot",
    "couldn't": "could not", "couldnt": "could not",
    "shouldn't": "should not", "shouldnt": "should not",
    "mightn't": "might not", "mightnt": "might not",
    "mustn't": "must not", "mustnt": "must not"
}

# Gaming slang dictionary
GAMING_SLANG = {
    "kys": "kill yourself",
    "stfu": "shut the fuck up",
    "wtf": "what the fuck",
    "ffs": "for fucks sake",
    "ez": "easy",
    "lmao": "laughing my ass off",
    "lmfao": "laughing my fucking ass off",
    "rofl": "rolling on floor laughing",
    "noob": "newbie",
    "noobs": "newbies",
    "gg": "good game",
    "gtfo": "get the fuck out",
    "omg": "oh my god",
    "w8": "wait",
    "gr8": "great",
    "plz": "please",
    "pls": "please",
    "thx": "thanks",
    "ty": "thank you",
    "u": "you",
    "r": "are",
    "afk": "away from keyboard",
    "brb": "be right back",
    "smh": "shaking my head",
    "glhf": "good luck have fun",
    "bg": "bad game",
    "wp": "well played",
    "toxicity": "toxic behavior",
    "hacker": "cheater",
    "aimbot": "cheating software"
}

class TextPreprocessor:
    def __init__(self):
        # Precompile regex for speed
        # Match words for contractions and slang
        self.word_re = re.compile(r"\b[a-zA-Z']+\b")
        
    def expand_contractions(self, text: str) -> str:
        """Expands standard English contractions."""
        words = text.split()
        expanded_words = []
        for word in words:
            # Check lowercase version
            word_lower = word.lower()
            # Clean punctuation from ends of word for checking
            cleaned_word = re.sub(r"^[^\w']+|[^\w']+$", "", word_lower)
            if cleaned_word in CONTRACTIONS:
                # Replace keeping surrounding characters if possible
                replaced = CONTRACTIONS[cleaned_word]
                expanded_words.append(word_lower.replace(cleaned_word, replaced))
            else:
                expanded_words.append(word)
        return " ".join(expanded_words)
        
    def replace_gaming_slang(self, text: str) -> str:
        """Replaces common gaming slang and abbreviations."""
        words = text.split()
        cleaned_words = []
        for word in words:
            word_lower = word.lower()
            cleaned_word = re.sub(r"^[^\w']+|[^\w']+$", "", word_lower)
            if cleaned_word in GAMING_SLANG:
                replaced = GAMING_SLANG[cleaned_word]
                cleaned_words.append(word_lower.replace(cleaned_word, replaced))
            else:
                cleaned_words.append(word)
        return " ".join(cleaned_words)
        
    def demojize_text(self, text: str) -> str:
        """Converts emojis into text descriptions (e.g., 🔥 -> :fire:)."""
        # We replace the colons from demojize with spaces to make it clean text
        demojized = emoji.demojize(text, delimiters=(" :", ": "))
        # Replace multiple spaces with a single space
        return re.sub(r"\s+", " ", demojized).strip()
        
    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline: demojize, casing, contractions, slang, character cleaning."""
        if not isinstance(text, str):
            return ""
            
        # 1. Demojize emojis first so we capture their semantic meaning
        text = self.demojize_text(text)
        
        # 2. Lowercase the text
        text = text.lower()
        
        # 3. Expand contractions
        text = self.expand_contractions(text)
        
        # 4. Replace gaming abbreviations
        text = self.replace_gaming_slang(text)
        
        # 5. Clean up special characters but keep letters, numbers, spaces, and exclamation/question marks
        text = re.sub(r"[^a-z0-9\s!?']", " ", text)
        
        # 6. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text

if __name__ == "__main__":
    preprocessor = TextPreprocessor()
    test_chat = "stfu you absolute noob!! 🖕🔥 i'm gonna hack u, kys ez game"
    print("Original:", test_chat)
    print("Preprocessed:", preprocessor.preprocess(test_chat))
