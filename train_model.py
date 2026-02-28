"""
Model Verification Module - DeepFace Upgrade
With Deep Learning (ArcFace/Vector DB), we NO LONGER NEED TO TRAIN a custom model!
The models are pre-trained. "Registration" is simply saving a 512-d vector.

This script now just verifies that the embeddings database exists and is healthy,
acting as a placeholder to keep the `main.py` menu intact without throwing errors.
"""

import os

# Ensure relative paths resolve from the script's own directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def train_model():
    print("\n" + "-"*60)
    print("🔄 VERIFYING DEEP LEARNING EMBEDDINGS...")
    print("-" * 60)
    
    db_path = os.path.join(_SCRIPT_DIR, 'dataset', 'embeddings')
    
    if not os.path.exists(db_path):
        print("❌ Error: Embeddings database not found!")
        print("   Please register at least one student first.")
        return
        
    embeddings = [f for f in os.listdir(db_path) if f.endswith('.npy')]
    
    if not embeddings:
         print("⚠️ Database is empty. Please register students.")
         return
         
    print(f"\n[✅] SUCCESS! The system is ready for instant recognition.")
    print(f"  • Found {len(embeddings)} registered student embeddings.")
    print(f"  • Using ArcFace (Pre-trained Deep Learning Model).")
    print("\nNote: Unlike LBPH, DeepFace does not require 'training'. Adding a student")
    print("      is instant! You can proceed directly to Start Attendance.\n")

if __name__ == "__main__":
    os.chdir(_SCRIPT_DIR)
    train_model()
