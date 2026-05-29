Project Title: Toxic Comment Filtering
Project Overview:
Develop and demonstrate a multi-label toxicity classifier for a gaming platform’s live chat. The system flags messages as spam, insult, profanity, hate speech, or non-toxic, assigns a 0-5 severity score, and outputs recommended moderation actions.

Scope:

Train multi-label classification model for toxic content detection.
Handle short-form, informal, and abbreviation-heavy chat text.
Build a moderation alert system based on severity level.
Step-by-Step Guide:

Use the Jigsaw Toxic Comment Dataset.
Preprocess for casing, emoji removal, and contractions.
Train classifier (BERT/DistilBERT or traditional ML).
Create a scoring system: e.g., Toxicity Score 0–5.
Test on a live chat simulation or dummy chat feed.
Project Guidelines:

Precision > 80% for “Severe Toxic” class.
Include multi-label capability (comments can belong to multiple categories).
Clearly define thresholds for moderation alerts.
Required Deliverables:

Trained toxicity classifier and test output.
Visual log of detected messages by type (generated within the notebook).
Summary of mitigation strategies (mute, warning, auto-ban).
Submissions Required:

Python project code + trained model file.
Jupyter notebook demo (and optional Streamlit script, if built).
Sample chat logs + detection logs.
Zip file submission with documentation and result visualizations.
