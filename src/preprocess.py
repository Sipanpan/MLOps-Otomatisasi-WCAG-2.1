import os
import hashlib # Tambahan untuk hashing
import numpy as np
from PIL import Image

def preprocess_images():
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    
    TARGET_SIZE = (150, 150)
    
    print(f"Memulai Preprocessing & Cleaning Data di {raw_dir}...")
    valid_count = 0
    duplicate_count = 0
    blank_count = 0
    
    # KUNCI UTAMA: Keranjang untuk mengingat sidik jari gambar yang sudah diproses
    seen_hashes = set() 
    
    for filename in os.listdir(raw_dir):
        if not filename.endswith('.jpg'):
            continue
            
        filepath = os.path.join(raw_dir, filename)
        
        try:
            # 1. HAPUS DUPLIKAT (Data Cleaning)
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
                
            # Jika sidik jarinya sudah ada di keranjang, lewati file ini!
            if file_hash in seen_hashes:
                print(f"🗑️ Skipping {filename}: Terdeteksi sebagai DUPLIKAT.")
                duplicate_count += 1
                continue
                
            # Jika aman, masukkan sidik jari ke keranjang
            seen_hashes.add(file_hash)
            
            # 2. PROSES GAMBAR (Normalisasi & Hapus Blank)
            with Image.open(filepath) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                
                img_array = np.array(img_resized)
                if np.var(img_array) < 10.0:
                    print(f"🌫️ Skipping {filename}: Gambar terlalu monoton/blank.")
                    blank_count += 1
                    continue
                
                # Simpan gambar yang lolos semua seleksi
                save_path = os.path.join(processed_dir, filename)
                img_resized.save(save_path)
                valid_count += 1
                
        except Exception as e:
            print(f"Data Cleaning - File Rusak {filename}: {e}")

    print("\n" + "="*40)
    print("📊 LAPORAN HASIL PREPROCESSING")
    print("="*40)
    print(f"✅ Data Valid Tersimpan : {valid_count} gambar")
    print(f"❌ Data Duplikat Dihapus: {duplicate_count} gambar")
    print(f"❌ Data Blank Dihapus   : {blank_count} gambar")
    print("="*40)

if __name__ == "__main__":
    preprocess_images()