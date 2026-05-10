import streamlit as st
from PIL import Image
import numpy as np

# Mengimpor modul buatan kita sendiri dari folder src
from src.inference import extract_single_palette
from src.autoheal import WCAGAutoHeal

# Konfigurasi Halaman Web
st.set_page_config(page_title="Auto-Heal WCAG 2.1 UI", layout="wide")

st.title("🎨 K-Means & Auto-Heal untuk Aksesibilitas UI")
st.markdown("**Proyek Data Mining & Metodologi Penelitian | Muhammad Affan Al Ghifari**")
st.write("Sistem machine learning untuk mengekstrak palet warna dari antarmuka web dan melakukan koreksi otomatis (Auto-Heal) agar lulus standar aksesibilitas WCAG 2.1.")

# 1. Fitur Unggah Gambar
uploaded_file = st.file_uploader("Unggah Screenshot Website (JPG/PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Tampilkan gambar yang diunggah
    image = Image.open(uploaded_file)
    st.image(image, caption="Antarmuka Asli", use_container_width=True)

    # Tombol untuk mulai memproses
    if st.button("Mulai Ekstrak & Analisis Warna", use_container_width=True):
        
        # --- TAHAP 1: EKSTRAKSI K-MEANS ---
        with st.spinner("Mengekstrak 5 warna dominan dengan K-Means..."):
            palette = extract_single_palette(image, n_colors=5)
        
        st.subheader("1️⃣ Palet Warna Dominan (Hasil K-Means)")
        
        # Tampilkan kotak warna asli menggunakan HTML/CSS di Streamlit
        cols = st.columns(5)
        for i, col in enumerate(cols):
            hex_color = '#%02x%02x%02x' % tuple(palette[i])
            rgb_text = f"rgb{tuple(palette[i])}"
            with col:
                st.markdown(f"""
                <div style="background-color:{hex_color}; height:80px; border-radius:10px; border:1px solid #ddd; margin-bottom:10px;"></div>
                """, unsafe_allow_html=True)
                
                st.caption(f"**Warna {i+1}**")
                # PEMISAHAN COPY HEX & RGB
                st.code(hex_color, language="plaintext")
                st.code(rgb_text, language="plaintext")
        
        st.divider()
        
        # --- TAHAP 2: EVALUASI & AUTO-HEAL ---
        st.subheader("2️⃣ Evaluasi & Auto-Heal (Standar WCAG 2.1)")
        st.info("💡 **Sistem mengasumsikan Warna 1 sebagai Background dominan.** Keempat warna lainnya akan diuji rasio kontrasnya terhadap Warna 1.")
        
        bg_color = palette[0]
        fg_colors = palette[1:]
        healer = WCAGAutoHeal(target_ratio=4.5)
        
        for i, fg_color in enumerate(fg_colors):
            st.markdown(f"#### Pengujian Teks Warna {i+2} pada Background Warna 1")
            
            # Hitung Rasio Asli
            rasio_asli = healer.calculate_contrast_ratio(bg_color, fg_color)
            lulus_asli = rasio_asli >= 4.5
            
            # Jalankan Algoritma Auto-Heal
            healed_color, success = healer.heal_color(bg_color, fg_color)
            rasio_baru = healer.calculate_contrast_ratio(bg_color, healed_color)
            
            # Konversi RGB ke HEX
            hex_bg = '#%02x%02x%02x' % tuple(bg_color)
            hex_fg = '#%02x%02x%02x' % tuple(fg_color)
            hex_healed = '#%02x%02x%02x' % tuple(healed_color)
            rgb_healed = f"rgb{tuple(healed_color)}"
            
            # Tampilan Perbandingan (Before - After)
            col_before, col_arrow, col_after = st.columns([4, 1, 4])
            
            with col_before:
                st.markdown("**Desain Asli (Sebelum)**")
                status_asli_text = "✅ Lulus WCAG" if lulus_asli else "❌ Gagal WCAG"
                border_color = "green" if lulus_asli else "red"
                st.markdown(f"""
                <div style="background-color:{hex_bg}; padding:30px; border-radius:10px; border:2px solid {border_color}; text-align:center;">
                    <span style="color:{hex_fg}; font-size:24px; font-weight:bold;">Contoh Teks Elemen {i+2}</span>
                </div>
                <p style="text-align:center; font-size:16px; margin-top:5px;">Rasio Kontras: <b>{rasio_asli:.2f}:1</b> | {status_asli_text}</p>
                """, unsafe_allow_html=True)
            
            with col_arrow:
                st.markdown("<h1 style='text-align:center; margin-top:35px; color:#aaa;'>➡️</h1>", unsafe_allow_html=True)
                
            with col_after:
                st.markdown("**Desain Auto-Heal (Sesudah)**")
                st.markdown(f"""
                <div style="background-color:{hex_bg}; padding:30px; border-radius:10px; border:2px solid green; text-align:center;">
                    <span style="color:{hex_healed}; font-size:24px; font-weight:bold;">Contoh Teks Elemen {i+2}</span>
                </div>
                <p style="text-align:center; font-size:16px; margin-top:5px; margin-bottom:5px;">Rasio Kontras: <b>{rasio_baru:.2f}:1</b> | ✅ Lulus WCAG</p>
                """, unsafe_allow_html=True)
                
                # PEMISAHAN COPY HEX & RGB UNTUK HASIL HEALED
                if not lulus_asli:
                    st.caption("Warna Baru (Koreksi):")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.code(hex_healed, language="plaintext")
                    with c2:
                        st.code(rgb_healed, language="plaintext")
                else:
                    st.info("Warna asli sudah lulus standar, tidak ada perubahan.")
            
            st.write("---")