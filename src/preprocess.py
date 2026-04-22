import os
import numpy as np
from PIL import Image

def preprocess_images():
    raw_dir = 'data/raw'
    processed_dir = 'data/processed' # Folder baru untuk hasil bersih
    os.makedirs(processed_dir, exist_ok=True)
    
    # Konfigurasi Preprocessing
    TARGET_SIZE = (150, 150) # Resizing untuk efisiensi
    
    print(f"Memulai Preprocessing Data di folder {raw_dir}...")
    valid_count = 0
    
    for filename in os.listdir(raw_dir):
        if not filename.endswith('.jpg'):
            continue
            
        filepath = os.path.join(raw_dir, filename)
        
        try:
            # 1. Buka Gambar (Otomatis handle file corrupt jika ada)
            with Image.open(filepath) as img:
                
                # 2. Color Space Conversion: Pastikan murni RGB (Buang Alpha/Transparan)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 3. Resizing: Perkecil gambar untuk mempercepat komputasi K-Means
                img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                # 4. Data Cleaning: Cek apakah gambar terlalu polos (blank)
                # Ambil variansi warna, jika terlalu kecil berarti gambar monoton (rusak/kosong)
                img_array = np.array(img_resized)
                if np.var(img_array) < 10.0:
                    print(f"Skipping {filename}: Gambar terlalu monoton/blank.")
                    continue
                
                # 5. Simpan gambar yang sudah bersih dan terstandarisasi
                save_path = os.path.join(processed_dir, filename)
                img_resized.save(save_path)
                valid_count += 1
                
        except Exception as e:
            print(f"Data Cleaning - File Rusak {filename}: {e}")

    print(f"\nPreprocessing Selesai! {valid_count} gambar valid tersimpan di {processed_dir}")
    print("Siap untuk proses Flattening dan Scaling di tahap K-Means.")

if __name__ == "__main__":
    preprocess_images()