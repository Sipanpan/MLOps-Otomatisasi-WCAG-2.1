import os
from datasets import load_dataset
from PIL import Image

def download_data():
    # 1. Pastikan folder tujuan tersedia
    os.makedirs('data/raw', exist_ok=True)
    
    # Buat folder cache lokal di dalam proyek agar terhindar dari limitasi Windows
    os.makedirs('data/hf_cache', exist_ok=True)
    
    print("Mulai mengunduh dataset UI dari Hugging Face...")
    
    # 2. Tambahkan parameter cache_dir
    dataset = load_dataset(
        "Voxel51/WaveUI-25k", 
        split="train[:50]", 
        cache_dir="data/hf_cache" # Mengalihkan penyimpanan sementara ke sini
    )
    
    print(f"Berhasil mengunduh {len(dataset)} gambar. Mulai menyimpan ke data/raw/...")
    
    # 3. Ekstrak dan simpan gambar
    for i, item in enumerate(dataset):
        try:
            image = item['image']
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            file_path = f"data/raw/ui_sample_{i+1}.jpg"
            image.save(file_path)
            
        except Exception as e:
            print(f"Gagal memproses gambar {i+1}: {e}")
            
    print("Selesai! 50 sampel gambar UI berhasil disimpan.")

if __name__ == "__main__":
    download_data()