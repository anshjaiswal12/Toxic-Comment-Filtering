# ==========================================
# Author: Ansh Jaiswal
# Neutra-Mod Zip Submission Automation Script
# ==========================================
import os
import zipfile

def create_zip_submission():
    print("==================================================")
    print("📦 BUILDING SUBMISSION ZIP FILE")
    print("==================================================\n")
    
    zip_name = "toxic_comment_filtering_submission.zip"
    
    # Files and folders to include
    include_paths = [
        "src",
        "models",
        "logs",
        "frontend",
        "artifacts",
        "app.py",
        "train.py",
        "generate_logs.py",
        "zip_submission.py",
        "toxicity_moderation_demo.ipynb",
        "README.md",
        "architecture.md",
        "submission_manifest.md"
    ]
    
    # Exclude directories
    exclude_extensions = [".pyc", ".pyo", ".git"]
    exclude_folders = [".venv", "__pycache__", ".git", ".ipynb_checkpoints"]
    
    print(f"Creating zip file: {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in include_paths:
            if not os.path.exists(path):
                print(f"⚠️ Warning: Path '{path}' does not exist yet. It will be skipped.")
                continue
                
            if os.path.isfile(path):
                zipf.write(path, path)
                print(f"  Added file: {path}")
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    # Filter out excluded directories
                    dirs[:] = [d for d in dirs if d not in exclude_folders]
                    
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Check extensions
                        if any(file.endswith(ext) for ext in exclude_extensions):
                            continue
                            
                        # Compute relative path for zip mapping
                        arcname = os.path.relpath(file_path, os.path.join(path, ".."))
                        zipf.write(file_path, arcname)
                print(f"  Added directory: {path}/")
                
    print(f"\n🏆 Submission file '{zip_name}' built successfully!")
    print("==================================================")

if __name__ == "__main__":
    create_zip_submission()
