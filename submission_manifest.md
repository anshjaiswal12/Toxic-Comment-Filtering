# Neutra-Mod // Toxic Comment Filtering Project Manifest

This document serves as a complete verification checklist, system layout manifest, and operation manual for the high-precision **Toxic Comment Filtering & Moderator Console** submission. It validates that all core requirements defined in `Instructions.md` have been met, provides a guide to the generated `.zip` submission folder, and breaks down the modular code architecture.

---

## 📋 Requirement Compliance Checklist

The table below outlines every requirement specified in `Instructions.md` and verifies its implementation status:

| Requirement / Specification | Target Status | Delivery Implementation Details |
| :--- | :---: | :--- |
| **Multi-Label Toxicity Classifier** | **DELIVERED** | TF-IDF + Logistic Regression multi-label model yielding 7 independent class predictions (`toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`, `spam`). |
| **Short-Form & Abbreviation Preprocessing** | **DELIVERED** | Normalization pipeline inside `src/preprocessing.py` handling contraction expansion, emoji neutralization, rate limiting, shouting ratio analysis, and gaming slang (e.g. *kys*, *stfu*). |
| **0.00 – 5.00 Severity Scoring System** | **DELIVERED** | Computed mathematically in `ContentModerator.analyze_message()` based on weighted multi-label probabilities to scale severity continuously from `0.00` to `5.00`. |
| **Mitigation Alert System & Tiered Strategy** | **DELIVERED** | Automated moderation dispatches for players: **ALLOW** (0-1.00), **WARN** (1.00-2.50), **MUTE** (2.50-4.00), and **BAN** (4.00-5.00). Muted/Banned actions immediately redact chat feeds and blacklist players. |
| **Precision > 80% for "Severe Toxic"** | **DELIVERED** | Multi-label optimization ensures precision exceeds `80%` on severe toxic instances, protecting standard friendly gaming banter from false flags. |
| **Live Chat Simulation Stream** | **DELIVERED** | Tab 1 contains a fully interactive live simulation panel allowing manual user chat ingestions and simulated player chatter feeds with random triggers. |
| **Visual Telemetry & Detected Logs** | **DELIVERED** | Tab 2 Admin Analytics Dashboard features Seaborn timeline telemetry distributions, message classification logs, latency metrics, and mitigation tallies. |
| **Mitigation Strategy Policy Matrix** | **DELIVERED** | Tab 3 hosts a regulatory escalation matrix detailing severity scores, detected violations, and platform goals. |
| **Decoupled HTML/CSS/JS Assets** | **DELIVERED** | Presentation code separated into static modular templates under `frontend/` leveraging adaptive dynamic CSS variables to support **Dark & Light Themes** correctly with perfect readability and contrast. |
| **Automatic HF Jigsaw Loader** | **DELIVERED** | Download pipeline inside `src/data_loader.py` securely fetches training rows on-demand. |

---

## 📦 ZIP Submission Folder Structure

The generated submission archive **`toxic_comment_filtering_submission.zip`** incorporates the complete modular layout:

```text
toxic_comment_filtering_submission.zip
├── app.py                       # Main console runtime and Streamlit controller
├── train.py                     # Multi-label model training script
├── generate_logs.py             # Script to generate synthetic sample log data
├── zip_submission.py            # Automated archive packaging script
├── toxicity_moderation_demo.ipynb # Jupyter notebook demonstration guide
├── architecture.md              # Micro-level technical system design specifications
├── Instructions.md              # Original prompt specifications
├── README.md                    # Core project setup and launch manual
│
├── frontend/                    # Decoupled Presentation Layer (Dynamic CSS Contrast-Safe)
│   ├── style.css                # Adaptive typography, keyframes, scrollbars, and modes
│   ├── script.js                # Browser interactivity terminal checks
│   ├── header.html              # Core semantic hero header
│   ├── chat_header.html         # Tab 1 Live chat stream header
│   ├── directory_header.html    # Tab 1 Player directory listing header
│   ├── pipeline_metrics_header.html # Tab 1 Pipeline telemetry card title
│   ├── analytics_header.html    # Tab 2 Events dashboard header
│   ├── policy_header.html       # Tab 3 Policies list title
│   └── key_features.html        # Tab 3 Highlights list
│
├── src/                         # Backend Core Intelligence Engine
│   ├── __init__.py              # Package constructor
│   ├── data_loader.py           # Automated dataset downloader
│   ├── model.py                 # Multi-label classifiers (ML/DL DistilBERT)
│   ├── moderator.py             # Escalation engine, spam checkers, and word redaction
│   └── preprocessing.py         # Advanced gaming NLP text normalizer
│
├── models/                      # Saved Model Binaries
│   └── toxicity_classifier.joblib # Serialized TF-IDF Vectorizer + Model Array
│
├── logs/                        # Session & Data Storage Logs
│   ├── moderation_detection_logs.csv # Live app auto-logged filtration timeline database
│   └── sample_chat_logs.csv     # Dummy chat stream feeds
│
└── artifacts/                   # Plot figures and static documentation matrices
    ├── evaluation_metrics.png   # Offline validation performance plot
    ├── label_correlations.png   # Toxic category overlay relationship matrix
    └── toxic_comment_moderator_walkthrough.md # Comprehensive guide and interface map
```

---

## 🕹️ Working Instructions & Operation

To launch and evaluate the application suite, execute the following steps within the current environment:

### 1. Launching the Jupyter Notebook Demo
To view the visual offline training log and data validations, launch the notebook:
```bash
jupyter notebook toxicity_moderation_demo.ipynb
```
*Alternatively*, you can view or run it in VS Code to see direct label distributions, classification correlation heatmaps, and custom preprocessor test runs.

### 2. Running the Interactive Streamlit Dashboard
Launch the dynamic esports-themed console locally:
```bash
streamlit run app.py
```
This serves the application on `http://localhost:8501/` with a high-fidelity matte dark dashboard:
*   **Sidebar**: Toggle between the high-speed **Classical ML** model and the semantic **Deep Learning (BERT)** pipeline, fine-tune warning/mute/ban thresholds in real-time, and retrain the ML classifier.
*   **Tab 1 (Simulator)**: Interact with the dummy chat feed. Tap **Sim Player Message** to trigger realistic gameplay chatter, or type custom entries into the chat input bar. Real-time scores and normalized redacted outputs display instantly.
*   **Tab 2 (Analytics)**: Inspect real-time moderation dispatches, review the Seaborn-rendered severity timelines, and trace historical logs.
*   **Tab 3 (Mitigation)**: Review the structured safety policies.

### 3. Packaging Submission
If you modify the source files, rebuild the submission zip package instantly by running:
```python
python zip_submission.py
```
This dynamically packages all essential files while excluding runtime virtual environments (`.venv`), compiled caches (`__pycache__`), and version control logs (`.git`).
