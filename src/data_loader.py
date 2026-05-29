import os
import random
import pandas as pd
import numpy as np

# A dictionary of clean and toxic gaming comment templates to generate a robust mock dataset for offline fallback
SYNTHETIC_TEMPLATES = {
    "clean": [
        "gg wp everyone",
        "nice shot man!",
        "can anyone help me with this raid?",
        "let's group up near mid lane",
        "i need some healing please",
        "great teamplay there guys, keep it up!",
        "thanks for the carry!",
        "is the new update out yet?",
        "what build are you running on your mage?",
        "glhf!",
        "we can still win this, don't give up",
        "sorry, my ping is super high right now",
        "i'll buy the next item",
        "good try guys, close game",
        "who wants to queue together next round?",
        "this game's graphics are absolutely stunning",
        "can we defend the base first?",
        "i'm placing a ward here",
        "gg guys, well played",
        "gg, that was a really fun match!"
    ],
    "toxic": [
        "you are absolutely trash at this game",
        "uninstall the game right now you garbage",
        "this team is so bad i can't even play",
        "garbage player, go back to tutorials",
        "stop throwing the match you idiot",
        "you're the worst player i have ever seen",
        "literally brainless team, waste of my time",
        "clown play, what are you doing?",
        "get good kid, you're awful",
        "you play like a blind bot",
        "this is why you are stuck in bronze",
        "report our jungler for being useless",
        "my teammates are actual monkeys",
        "stop stealing my farm you piece of trash"
    ],
    "severe_toxic": [
        "kys absolute garbage trash player",
        "go kill yourself, no one wants you here",
        "i hope you die in a painful fire",
        "hope your pc explodes and burns your house down",
        "kill yourself, you are a waste of oxygen",
        "hope you get a terminal disease",
        "drink bleach you worthless piece of trash",
        "go hang yourself, uninstall life",
        "i'm gonna end your life next time we match",
        "literally slit your wrists you bot"
    ],
    "obscene": [
        "fuck off and leave me alone",
        "shut the fuck up stfu you b",
        "what the fuck are you even doing?",
        "you piece of shit teammates",
        "this is fucking ridiculous",
        "bullshit match, bad matchmaking",
        "go fuck yourself, no one cares",
        "stfu stfu stfu you motherfucker",
        "fucking retard player, fuck this",
        "he's a cheating asshole"
    ],
    "threat": [
        "i will find where you live and kill you",
        "i am going to track your ip and murder you in real life",
        "i will hunt you down and slit your throat",
        "i know your location, i'm coming for you",
        "gonna swat your house and get you killed",
        "i will hack your account and leak all your personal info",
        "i'll murder your whole family if you throw again",
        "i'm going to track your IP address and end you"
    ],
    "insult": [
        "you are a complete idiot",
        "you are so stupid and brain-dead",
        "you're an absolute loser, get a life",
        "retarded teammate, useless bot",
        "you have a double digit IQ, absolute moron",
        "you're a clown, delete the game",
        "dumb player, why do you exist?",
        "you are so pathetic, go cry to your mom"
    ],
    "identity_hate": [
        "go back to your own country you immigrant",
        "racial slurs, get out of our server",
        "he's a gay faggot, report him",
        "fucking immigrant trash ruin this game",
        "go wash dishes women shouldn't be playing gaming",
        "i hate people of your race, hope you all die",
        "fucking nigger hacking on our server",
        "you people are subhuman scum"
    ]
}

def generate_synthetic_dataset(num_samples=2500) -> pd.DataFrame:
    """Generate a highly representative multi-label synthetic dataset for offline training."""
    print(f"Generating rich, representative synthetic dataset of {num_samples} samples...")
    data = []
    
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
    for i in range(num_samples):
        # 1. Decide category mixture
        # We want to represent a mix of clean (50%) and toxic (50%) comments
        is_toxic = random.random() < 0.50
        
        comment_text = ""
        labels = {cat: 0 for cat in categories}
        
        if not is_toxic:
            # Clean comment
            comment_text = random.choice(SYNTHETIC_TEMPLATES["clean"])
            # Maybe add minor typos/repeats
            if random.random() < 0.2:
                comment_text += " !!!"
        else:
            # Toxic comment - select 1 to 3 active categories to make it multi-label
            active_cats = []
            
            # Choose a primary toxic category
            primary_cat = random.choice(categories)
            active_cats.append(primary_cat)
            
            # Maybe add a secondary category
            if random.random() < 0.4 and primary_cat in ['toxic', 'obscene', 'insult']:
                secondary_cat = random.choice([c for c in categories if c != primary_cat])
                active_cats.append(secondary_cat)
                
            # If severe_toxic, almost always flag 'toxic' and 'obscene' or 'insult' as well
            if 'severe_toxic' in active_cats:
                if 'toxic' not in active_cats: active_cats.append('toxic')
                if 'obscene' not in active_cats and random.random() < 0.7: active_cats.append('obscene')
                
            if 'identity_hate' in active_cats or 'threat' in active_cats:
                if 'toxic' not in active_cats: active_cats.append('toxic')
                
            # Build text by combining templates
            texts = []
            for cat in active_cats:
                texts.append(random.choice(SYNTHETIC_TEMPLATES[cat]))
                labels[cat] = 1
                
            # Shuffle sentences to make it feel natural
            random.shuffle(texts)
            comment_text = " ".join(texts)
            
            # Add typical spelling variations/gaming typos
            if random.random() < 0.3:
                # e.g. replace you with u, are with r
                comment_text = comment_text.replace("you ", "u ").replace("are ", "r ")
            if random.random() < 0.15:
                # Add emojis
                comment_text += " 🖕💀🤡"
                
        data.append({
            "comment_text": comment_text,
            **labels
        })
        
    df = pd.DataFrame(data)
    # Ensure there are no duplicate comments to avoid leakages
    df = df.drop_duplicates(subset=["comment_text"])
    return df

def load_toxicity_dataset() -> Tuple[pd.DataFrame, bool]:
    """
    Tries to download the Jigsaw Toxic Comment dataset from Hugging Face.
    If it fails, automatically generates a rich synthetic gaming-chat dataset.
    """
    from typing import Tuple
    
    print("Attempting to load Jigsaw Toxic Comment Dataset...")
    hf_repos = [
        "thesofakillers/jigsaw-toxic-comment-classification-challenge",
        "julian-r/jigsaw-toxic-comment-classification",
        "OxK/jigsaw-toxic-comment-classification-challenge"
    ]
    
    for repo in hf_repos:
        try:
            print(f"Trying Hugging Face repository: '{repo}'...")
            from datasets import load_dataset
            dataset = load_dataset(repo, split="train")
            df = pd.DataFrame(dataset)
            
            # Check standard columns
            required_cols = ["comment_text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            if all(col in df.columns for col in required_cols):
                print(f"Jigsaw dataset successfully loaded from Hugging Face ('{repo}')! Row count: {len(df)}")
                return df[required_cols], False # False indicates it is NOT synthetic
        except Exception as e:
            print(f"Could not load from '{repo}' (reason: {e}).")
        
    # Offline fallback
    print("Falling back to generating high-fidelity gaming-chat synthetic training dataset...")
    df = generate_synthetic_dataset(num_samples=4000)
    return df, True # True indicates it is synthetic
