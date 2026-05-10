import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def extract_single_palette(image_source, n_colors=5):
    """
    Mengekstrak n_colors warna dominan dari satu gambar menggunakan algoritma K-Means.
    
    :param image_source: Bisa berupa path string (lokasi file) atau objek PIL Image (dari Streamlit)
    :param n_colors: Jumlah warna yang ingin diekstrak (Standar UI = 5)
    :return: List berisi 5 warna RGB [R, G, B] dalam rentang 0-255
    """
    # 1. Buka gambar
    if isinstance(image_source, str):
        img = Image.open(image_source)
    else:
        img = image_source
        
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    # 2. Resize gambar untuk mempercepat komputasi K-Means 
    # (Mengecilkan gambar tidak akan mengubah warna dominan secara signifikan)
    img = img.resize((150, 150), Image.Resampling.LANCZOS)
    
    # 3. Bentuk ulang (reshape) matriks piksel menjadi flat list of 3D vectors [R, G, B]
    img_array = np.array(img)
    pixels = img_array.reshape(-1, 3)
    
    # 4. Jalankan K-Means
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    
    # 5. Ambil warna pusat cluster (centroid) yang merepresentasikan warna dominan
    colors = kmeans.cluster_centers_
    
    # Konversi hasil array float ke dalam bentuk integer 0-255 dan jadikan list
    palette = colors.astype(int).tolist()
    
    return palette

# --- BLOK PENGUJIAN ---
if __name__ == "__main__":
    import os
    
    # Kita tes menggunakan salah satu data sampel yang ada di foldermu
    test_img = "data/raw/ui_sample_1.jpg" 
    
    if os.path.exists(test_img):
        print(f"Mengekstrak 5 warna dominan dari {test_img}...")
        palet_ekstrak = extract_single_palette(test_img, n_colors=5)
        
        print("\nHasil Ekstraksi K-Means:")
        for i, warna in enumerate(palet_ekstrak):
            print(f"Warna {i+1}: RGB {warna}")
    else:
        print(f"File {test_img} tidak ditemukan. Pastikan path gambar benar.")