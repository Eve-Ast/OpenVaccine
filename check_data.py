import os
import pandas as pd
from src import config  # Importe votre fichier de configuration

def verify_setup():
    print("=== VÉRIFICATION DES FICHIERS DE DONNÉES ===")
    
    # Chemins issus de config.py ou par défaut
    data_dir = getattr(config, 'DATA_DIR', './data')
    train_path = os.path.join(data_dir, 'train.json.zip')
    test_path = os.path.join(data_dir, 'test.json.zip')
    bpps_dir = os.path.join(data_dir, 'bpps')

    # Vérification des fichiers JSON
    for path, name in [(train_path, 'train.json'), (test_path, 'test.json')]:
        if os.path.exists(path):
            df = pd.read_json(path, lines=True)
            print(f"✅ {name} trouvé : {len(df)} lignes carregées.")
        else:
            print(f"❌ {name} INTROUVABLE au chemin : {path}")

    # Vérification du dossier BPPS
    if os.path.exists(bpps_dir):
        files = os.listdir(bpps_dir)
        print(f"✅ Dossier bpps/ trouvé : {len(files)} matrices .npy présentes.")
    else:
        print(f"⚠️ Dossier bpps/ introuvable au chemin : {bpps_dir} (Attention si vos modèles l'utilisent !)")

if __name__ == "__main__":
    verify_setup()