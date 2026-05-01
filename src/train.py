import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
import mlflow
import mlflow.sklearn

def run_mlflow_experiment():
    # 0. KUNCI ALAMAT PENYIMPANAN KE FOLDER MLRUNS
    mlflow.set_tracking_uri("file:./mlruns")
    
    # 0. SAPU BERSIH SESI YANG NYANGKUT
    while mlflow.active_run():
        mlflow.end_run()

    processed_dir = 'data/processed'
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(processed_dir) if f.endswith('.jpg')]
    sample_images = image_files[:50] 
    
    mlflow.set_experiment("UI_Color_Accessibility_KMeans")
    k_values = [2, 3, 4, 5, 6, 7, 8]
    inertia_list = []

    print("🔬 Memulai Eksperimen dengan MLflow...")

    # BUNGKUS SEMUA DALAM 1 PARENT RUN AGAR RAPI
    with mlflow.start_run(run_name="Eksperimen_Utama"):
        
        # 1. LOOP EKSPERIMEN 
        for k in k_values:
            with mlflow.start_run(run_name=f"KMeans_K_{k}", nested=True):
                total_inertia = 0
                for filename in sample_images:
                    filepath = os.path.join(processed_dir, filename)
                    img = Image.open(filepath)
                    pixels = np.array(img).reshape(-1, 3) / 255.0
                    
                    np.random.seed(42)
                    pixels_subset = pixels[np.random.choice(pixels.shape[0], 2000, replace=False)]
                    
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    model.fit(pixels_subset)
                    total_inertia += model.inertia_
                
                avg_inertia = total_inertia / len(sample_images)
                inertia_list.append(avg_inertia)
                
                mlflow.log_param("k_value", k)
                mlflow.log_metric("average_inertia", avg_inertia)
                print(f"Selesai K={k} | Inertia: {avg_inertia:.2f}")

        # 2. VISUALISASI ELBOW METHOD
        plt.figure(figsize=(8, 5))
        plt.plot(k_values, inertia_list, marker='o', color='b')
        plt.title('Elbow Method Evaluation')
        plt.xlabel('K')
        plt.ylabel('Inertia')
        plt.grid(True)
        plot_path = os.path.join(models_dir, "elbow_plot_mlflow.png")
        plt.savefig(plot_path)
        
        mlflow.log_artifact(plot_path)

        # 3. TRAINING MODEL TERBAIK (FINAL)
        best_k = 5
        print(f"\n🏆 Membangun Model Terbaik (K={best_k})...")
        
        with mlflow.start_run(run_name="Final_Model_Extraction", nested=True):
            palettes_data = {}
            for filename in image_files:
                filepath = os.path.join(processed_dir, filename)
                img = Image.open(filepath)
                pixels = np.array(img).reshape(-1, 3) / 255.0
                
                model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                model.fit(pixels) 
                
                palettes_data[filename] = (model.cluster_centers_ * 255).astype(int).tolist()
            
            final_model_path = os.path.join(models_dir, 'best_extracted_palettes.json')
            with open(final_model_path, 'w') as f:
                json.dump(palettes_data, f)
                
            mlflow.log_param("final_k", best_k)
            mlflow.log_artifact(final_model_path)
            print(f"✅ Berhasil! Semua log dan model telah dicatat oleh MLflow.")

if __name__ == "__main__":
    run_mlflow_experiment()