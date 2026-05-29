# 🛡️ NEUTRA-MOD: High-Fidelity Toxic Comment Filtering & Moderation Alert System

**Author**: Ansh Jaiswal

NEUTRA-MOD is a state-of-the-art, multi-label text classifier and automated moderation alert system designed for a gaming platform's live chat feed. The system Normalizes gaming-specific dialects (abbreviations, slurs, emojis), identifies structural spamming, computes a unified **0 to 5 severity hazard score**, and enforces dynamic moderation actions (**ALLOW**, **WARN**, **MUTE**, **BAN**).

---

## 🌟 Key Features

1. **Context-Aware Text Preprocessor**: Normalizes gaming slang (*stfu*, *kys*, *ez*, *noob*) and translates raw emoji context (e.g. `🖕` $\rightarrow$ `:middle_finger:`) into high-signal textual features.
2. **Multi-Label Toxicity Classifier**: Flags text across 6 hazard profiles (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`).
3. **Precision Optimization Guardrails**: Automatically tunes decision boundaries to guarantee **Precision > 80%** specifically for the `severe_toxic` class, preventing false-positive bans in competitive environments.
4. **Structural Spam Detector**: Parses raw comments for links, advertisements, character spamming (flooding), and shouting (excessive capitalization).
5. **Unified Severity Scoring (0-5)**: Mathematical aggregation of weighted probabilities.
6. **Tiered Mitigation Policies**: Bridges prediction scores directly to operational moderation triggers.
7. **Premium Streamlit Lobby Simulator**: A beautiful, fully interactive cyberpunk game lobby interface showing live analysis, bot interventions, and real-time administrator visualization charts.

---

## ⚙️ Directory Structure

```directory
Newton Project AI/
├── src/
│   ├── __init__.py           # Package marker
│   ├── preprocessing.py      # Normalization, emojis, contractions, and slang abbreviations
│   ├── data_loader.py        # HuggingFace datasets loader with offline synthetic fallback
│   ├── model.py              # Classical TF-IDF & Deep Learning wrapper classes
│   └── moderator.py          # Spam checking, severity scores, and message redactions
├── models/
│   └── toxicity_classifier.joblib  # Serialized trained model pipeline
├── logs/
│   ├── sample_chat_logs.csv        # Simulated live gaming raw chat feed
│   └── moderation_detection_logs.csv # Detailed moderator analysis outputs
├── artifacts/
│   ├── evaluation_metrics.png      # Validation performance bar plots
│   └── label_correlations.png      # Infraction heatmap analysis
├── app.py                     # Premium Streamlit Lobby App
├── train.py                   # Automated ML training script
├── generate_logs.py           # Simulated server logging script
├── toxicity_moderation_demo.ipynb # Comprehensive Jupyter Notebook Demo
├── Instructions.md            # Original project requirements
└── README.md                  # Detailed Documentation (This file)
```

---

## 🛠️ Step-by-Step Setup & Usage

This project is built using Python 3.14 on Arch Linux. We manage dependencies and environments using `uv` for lightning-fast speeds.

### 1. Initialize Virtual Environment & Activate

If not already inside a virtual environment, activate it:
```bash
source .venv/bin/activate
```

### 2. Run the Automated Training Pipeline

Before using the dashboard, run the training pipeline to pre-train the model, calibrate thresholds, and output validation performance visualizations:
```bash
python train.py
```
*Note: The script automatically attempts to fetch the Jigsaw dataset from Hugging Face. If offline or slow, it immediately falls back to a high-fidelity 4,000-sample gaming-specific synthetic dataset to guarantee a fast and successful run.*

### 3. Generate Simulated Chat Logs

Produce the raw chat logs and detailed detection logs required by the deliverables:
```bash
python generate_logs.py
```
This generates:
- `logs/sample_chat_logs.csv`: Standard streaming game chat records.
- `logs/moderation_detection_logs.csv`: Full AI-moderator analysis logs with preprocessed outputs, spam status, class flags, severity scores, actions, redacted strings, and model latency records.

### 4. Deploy the Streamlit Lobby Dashboard

Launch the premium interactive gaming simulator:
```bash
streamlit run app.py
```
Open the provided URL (typically `http://localhost:8501`) in your browser to experience the dashboard:
- Join the chat lobby and type messages to test immediate AI feedback.
- Simulate random players in the channel and see Moderator Bot interventions.
- Inspect the Admin Dashboard showing timeline charts and category breakdowns.
- Adjust sliders in the control panel to modify active moderation thresholds.

### 5. Review Jupyter Notebook Documentation

Open the Jupyter Notebook to review step-by-step pipeline execution, validation charts, and code walk-throughs:
```bash
jupyter notebook toxicity_moderation_demo.ipynb
```

---

## 🔬 System Architecture & Mathematics

### A. Severity Score Math
The Unified Severity Score is computed as a weighted average of model probabilities, normalized, and scaled to a $0 - 5$ range:

$$\text{Severity} = \text{clip}\left(\frac{\sum (P_i \times w_i) + (S_{\text{spam}} \times 1.5)}{\sum w_i + 1.5} \times 5.0, \; 0.0, \; 5.0\right)$$

*Weights:*
- `severe_toxic` / `identity_hate`: $2.0$
- `threat`: $1.8$
- `toxic` / `obscene` / `insult`: $1.0$
- `spam`: $1.5$

*Hazard Boost:* If high-hazard flags (`severe_toxic`, `identity_hate`, `threat`) exceed a $0.65$ probability threshold, the score is amplified upwards to immediately trigger muting/banning.

### B. Tiered Mitigation Policies

| Severity Score Range | Violations Detected | Moderation Action | Intervention Goal |
| :--- | :--- | :--- | :--- |
| **0.00 – 1.00** | Harmless gameplay talk | **ALLOW** | Maintain uninterrupted gameplay flow |
| **1.00 – 2.50** | Spamming links, low insults | **WARN** | Enforce alert guardrails and warnings |
| **2.50 – 4.00** | Severe insults, obscene slurs | **MUTE** | Redact profanity, mute user chat for 5m |
| **4.00 – 5.00** | Hate speech, self-harm, threats | **BAN** | Terminate player session, block IP |
