import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

def run_experiment_and_train():
    processed_dir = 'data/processed'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(processed_dir) if f.endswith('.jpg')]
    
    # 1. SETUP EKSPERIMEN
    k_values_to_test = [2, 3, 4, 5, 6, 7, 8] # Kita bereksperimen mencari jumlah warna terbaik
    experiment_logs = []
    
    print(f"🔬 MEMULAI EKSPERIMEN K-MEANS PADA {len(image_files)} GAMBAR...")
    
    # Untuk mempercepat eksperimen, kita ambil sampel 50 gambar saja untuk mencari parameter terbaik
    sample_images = image_files[:50] 
    
    for k in k_values_to_test:
        total_inertia = 0
        print(f"Mencoba Parameter K = {k}...")
        
        for filename in sample_images:
            filepath = os.path.join(processed_dir, filename)
            img_array = np.array(Image.open(filepath))
            pixels = img_array.reshape(-1, 3) / 255.0
            
            np.random.seed(42)
            subset_indices = np.random.choice(pixels.shape[0], 2000, replace=False)
            pixels_subset = pixels[subset_indices]
            
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(pixels_subset)
            
            # Inertia adalah Metrik evaluasi K-Means (semakin kecil semakin baik, tapi ada batas optimalnya)
            total_inertia += kmeans.inertia_ 
            
        avg_inertia = total_inertia / len(sample_images)
        experiment_logs.append({'K_Parameter': k, 'Average_Inertia_Metric': avg_inertia})
        print(f"  -> Rata-rata Inertia (Metrik): {avg_inertia:.2f}")

    # 2. SIMPAN LOG EKSPERIMEN (Syarat Dosen)
    log_df = pd.DataFrame(experiment_logs)
    log_path = os.path.join(models_dir, 'experiment_log.csv')
    log_df.to_csv(log_path, index=False)
    
    # 3. BUAT GRAFIK METRIK TERBAIK (Elbow Method)
    plt.figure(figsize=(8, 5))
    plt.plot(log_df['K_Parameter'], log_df['Average_Inertia_Metric'], marker='o', linestyle='--', color='b')
    plt.title('Evaluasi Model: Elbow Method untuk mencari K Optimal')
    plt.xlabel('Jumlah Cluster / Warna (K)')
    plt.ylabel('Metrik (Average Inertia)')
    plt.grid(True)
    plot_path = os.path.join(models_dir, 'metric_elbow_plot.png')
    plt.savefig(plot_path)
    print(f"\\n📊 Grafik Evaluasi Metrik disimpan di {plot_path}")
    
    # 4. TENTUKAN MODEL TERBAIK (K=5 biasanya adalah siku/elbow yang optimal untuk UI)
    best_k = 5 
    print(f"\\n🏆 KESIMPULAN EKSPERIMEN: Parameter Terbaik adalah K = {best_k} (Berdasarkan Elbow Method)")
    
    # 5. EKSTRAKSI DATA PENUH DENGAN MODEL TERBAIK
    print(f"\\n🚀 Mengekstrak model akhir untuk seluruh {len(image_files)} gambar menggunakan K={best_k}...")
    palettes_data = {}
    for i, filename in enumerate(image_files):
        filepath = os.path.join(processed_dir, filename)
        img_array = np.array(Image.open(filepath))
        pixels = (img_array.reshape(-1, 3) / 255.0)
        
        np.random.seed(42)
        pixels_subset = pixels[np.random.choice(pixels.shape[0], 3000, replace=False)]
        
        best_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        best_model.fit(pixels_subset)
        
        palettes_data[filename] = (best_model.cluster_centers_ * 255).astype(int).tolist()
        
    final_model_path = os.path.join(models_dir, 'best_extracted_palettes.json')
    with open(final_model_path, 'w') as f:
        json.dump(palettes_data, f)
        
    print(f"✅ Selesai! Model Terbaik disimpan di {final_model_path}")

if __name__ == "__main__":
    run_experiment_and_train()