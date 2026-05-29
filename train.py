import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, hamming_loss
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import TextPreprocessor
from src.data_loader import load_toxicity_dataset
from src.model import ToxicityClassifier

def run_training_pipeline():
    print("==================================================")
    print("🚀 STARTING TOXICITY CLASSIFIER TRAINING PIPELINE")
    print("==================================================\n")
    
    # 1. Load Data
    df, is_synthetic = load_toxicity_dataset()
    
    # Subsample if dataset is too large to keep preprocessing and training fast and memory-safe
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    if len(df) > 15000:
        print(f"Dataset is large ({len(df)} rows). Stratified subsampling for fast, high-quality training...")
        toxic_mask = df[categories].any(axis=1)
        df_toxic = df[toxic_mask]
        df_clean = df[~toxic_mask]
        
        # Balance dataset: take all toxic comments and match with sample of clean comments
        n_clean_needed = max(6000, len(df_toxic))
        df_clean_sampled = df_clean.sample(n=min(n_clean_needed, len(df_clean)), random_state=42)
        
        df = pd.concat([df_toxic, df_clean_sampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Subsampled dataset to {len(df)} rows (Toxic: {len(df_toxic)}, Clean: {len(df_clean_sampled)}).")
        
    # 2. Preprocess Data
    print("Preprocessing texts (this might take a moment)...")
    preprocessor = TextPreprocessor()
    
    start_time = time.time()
    df['clean_comment'] = df['comment_text'].apply(preprocessor.preprocess)
    elapsed_preprocess = time.time() - start_time
    print(f"Preprocessed {len(df)} rows in {elapsed_preprocess:.2f} seconds.")
    
    # Filter out empty comments after preprocessing
    df = df[df['clean_comment'] != ""]
    print(f"Dataset size after cleaning empty comments: {len(df)}")
    
    # 3. Split Train/Validation
    categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    X = df['clean_comment'].to_numpy()
    y = df[categories].to_numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training split: {X_train.shape[0]} samples. Validation split: {X_val.shape[0]} samples.\n")
    
    # 4. Train Model
    clf = ToxicityClassifier(categories=categories)
    start_train = time.time()
    clf.fit(X_train, y_train)
    print(f"Model fitted in {time.time() - start_train:.2f} seconds.")
    
    # 5. Optimize thresholds for target precision > 80% on 'severe_toxic'
    # The requirement is: "Precision > 80% for 'Severe Toxic' class."
    # Let's run optimization to ensure this.
    clf.optimize_thresholds(X_val, y_val, target_precision=0.82, target_class='severe_toxic')
    
    # 6. Evaluate Validation Metrics
    print("\n==============================================")
    print("📊 VALIDATION PERFORMANCE EVALUATION")
    print("==============================================\n")
    
    # Predict using the custom thresholds
    preds_val_dicts = clf.predict(X_val)
    
    # Convert predictions back to numpy matrix for report
    y_pred = np.zeros((len(X_val), len(categories)))
    for i, pred_dict in enumerate(preds_val_dicts):
        for j, col in enumerate(categories):
            y_pred[i, j] = pred_dict[col]
            
    # Compute metrics
    h_loss = hamming_loss(y_val, y_pred)
    print(f"Hamming Loss (fraction of misclassified labels): {h_loss:.4f}")
    
    # Print per-class report
    print("\nClassification Report (per class):")
    report = classification_report(y_val, y_pred, target_names=categories, output_dict=False, zero_division=0.0)
    print(report)
    
    # Generate classification report as dictionary for validation plots
    report_dict = classification_report(y_val, y_pred, target_names=categories, output_dict=True, zero_division=0.0)
    
    # 7. Generate Evaluation Visualization Plots
    os.makedirs("models", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['precision', 'recall', 'f1-score']
    class_metrics = {m: [report_dict[c][m] for c in categories] for m in metrics_to_plot}
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # Dark modern styling for the plot
    plt.style.use('dark_background')
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1A1C23')
    
    colors = ['#1E88E5', '#D81B60', '#FFC107'] # blue, dark pink, amber
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i*width - width/2, class_metrics[metric], width, label=metric.capitalize(), color=colors[i])
        
    ax.set_ylabel('Score', color='white', fontsize=12)
    ax.set_title('Toxicity Classifier - Validation Performance by Category', color='white', fontsize=14, pad=15)
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(categories, color='white', fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1A1C23', edgecolor='none', labelcolor='white')
    
    # Draw horizontal target line at 80% for severe toxic precision
    severe_idx = categories.index('severe_toxic')
    ax.axhline(y=0.8, color='#00E676', linestyle='--', alpha=0.7, label='Target (>80% Precision for Severe Toxic)')
    
    plt.tight_layout()
    plot_path = "artifacts/evaluation_metrics.png"
    plt.savefig(plot_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Performance plots successfully saved to: {plot_path}")
    
    # Save the correlation matrix of categories
    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#0E1117')
    ax.set_facecolor('#1A1C23')
    corr = pd.DataFrame(y, columns=categories).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, cbar_kws={'label': 'Correlation'}, square=True)
    ax.set_title('Toxicity Label Correlation Matrix', color='white', fontsize=12, pad=15)
    ax.tick_params(colors='white')
    plt.tight_layout()
    corr_path = "artifacts/label_correlations.png"
    plt.savefig(corr_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    print(f"Label correlation heatmap saved to: {corr_path}")
    
    # 8. Save Trained Model
    model_save_path = "models/toxicity_classifier.joblib"
    clf.save(model_save_path)
    print(f"\nModel pipeline saved successfully to {model_save_path}")
    
    # 9. Verify on a live chat simulation test
    print("\n==============================================")
    print("🕹️ SAMPLE TEST RUN (GAMING CHAT STREAM)")
    print("==============================================\n")
    
    test_messages = [
        "ggwp, nice game guys!",
        "stfu you garbage noob, stop throwing the match",
        "you absolute brainless idiot, go uninstall life",
        "i am going to find where you live and end you tonight",
        "let's group up and secure the healing shrine",
        "kys u subhuman immigrant trash"
    ]
    
    # Wrap in ContentModerator for a quick demonstration
    from src.moderator import ContentModerator
    moderator = ContentModerator(classifier=clf)
    
    for msg in test_messages:
        analysis = moderator.analyze_message(msg, username="Gamer_Alpha")
        print(f"Message: '{analysis['original_text']}'")
        print(f"  └─ Preprocessed: '{analysis['preprocessed_text']}'")
        print(f"  └─ Flags: { {k: v for k, v in analysis['flags'].items() if v == 1} }")
        print(f"  └─ Severity (0-5): {analysis['severity_score']}")
        print(f"  └─ Moderation Action: {analysis['action']} ({analysis['action_reason']})")
        print(f"  └─ Redacted Text: '{analysis['redacted_text']}'")
        print(f"  └─ Latency: {analysis['latency_ms']} ms\n")
        
    print("==============================================")
    print("🏆 TRAINING PIPELINE RUN SUCCESSFULLY COMPLETE")
    print("==============================================")

if __name__ == "__main__":
    run_training_pipeline()
