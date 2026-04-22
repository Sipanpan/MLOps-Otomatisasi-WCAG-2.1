import os
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def extract_palettes():
    processed_dir = 'data/processed'
    models_dir = 'models' # Folder untuk menyimpan hasil/model
    os.makedirs(models_dir, exist_ok=True)
    
    palettes_data = {}
    
    image_files = [f for f in os.listdir(processed_dir) if f.endswith('.jpg')]
    print(f"Mengekstrak palet warna dari {len(image_files)} gambar...")
    
    for i, filename in enumerate(image_files):
        filepath = os.path.join(processed_dir, filename)
        
        # Buka gambar yang sudah di-resize (150x150)
        img = Image.open(filepath)
        img_array = np.array(img)
        
        # --- TAHAP 4: REDUKSI DIMENSI (FLATTENING) ---
        # Mengubah matriks 3D (150, 150, 3) menjadi list 2D (22500, 3)
        pixels = img_array.reshape(-1, 3)
        
        # --- TAHAP 5: NORMALISASI DATA (SCALING) ---
        # Mengubah rentang piksel 0-255 menjadi 0.0 - 1.0 agar K-Means lebih cepat konvergen
        pixels_scaled = pixels / 255.0
        
        # Trik Optimasi: Ambil 3000 piksel acak saja per gambar agar komputasi kilat
        np.random.seed(42)
        subset_indices = np.random.choice(pixels_scaled.shape[0], 3000, replace=False)
        pixels_subset = pixels_scaled[subset_indices]
        
        # --- TAHAP 6: ALGORITMA K-MEANS ---
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels_subset)
        
        # Kembalikan nilai ke 0-255 (format standar RGB) untuk disimpan
        dominant_colors = (kmeans.cluster_centers_ * 255).astype(int)
        
        # Simpan ke dictionary
        palettes_data[filename] = dominant_colors.tolist()
        
        if (i+1) % 20 == 0:
            print(f"Progres: {i+1}/{len(image_files)} gambar diproses...")

    # Simpan hasil ekstraksi (Knowledge Base)
    output_file = os.path.join(models_dir, 'extracted_palettes.json')
    with open(output_file, 'w') as f:
        json.dump(palettes_data, f, indent=4)
        
    print(f"\nSelesai! Palet warna dari {len(image_files)} gambar berhasil disimpan di {output_file}")

if __name__ == "__main__":
    extract_palettes()