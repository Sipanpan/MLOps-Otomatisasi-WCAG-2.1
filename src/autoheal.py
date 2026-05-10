import numpy as np
import colorsys
import matplotlib.colors as mcolors

class WCAGAutoHeal:
    def __init__(self, target_ratio=4.5):
        """
        Inisialisasi modul Auto-Heal untuk aksesibilitas warna.
        :param target_ratio: Rasio minimal (WCAG 2.1 AA untuk teks normal adalah 4.5:1)
        """
        self.target_ratio = target_ratio

    def calculate_relative_luminance(self, rgb):
        """
        Menghitung relative luminance menurut standar WCAG 2.1
        :param rgb: list atau tuple warna RGB dalam rentang 0-255
        """
        # Normalisasi ke 0.0 - 1.0
        r, g, b = [x / 255.0 for x in rgb]

        # Konversi sRGB ke linear RGB
        r = r / 12.92 if r <= 0.03928 else ((r + 0.055) / 1.055) ** 2.4
        g = g / 12.92 if g <= 0.03928 else ((g + 0.055) / 1.055) ** 2.4
        b = b / 12.92 if b <= 0.03928 else ((b + 0.055) / 1.055) ** 2.4

        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def calculate_contrast_ratio(self, color1, color2):
        """
        Menghitung rasio kontras antara 2 warna.
        :param color1, color2: warna RGB dalam rentang 0-255
        :return: float rasio kontras (misal 4.5)
        """
        l1 = self.calculate_relative_luminance(color1)
        l2 = self.calculate_relative_luminance(color2)
        
        light = max(l1, l2)
        dark = min(l1, l2)
        
        return (light + 0.05) / (dark + 0.05)

    def heal_color(self, bg_color, fg_color):
        """
        Mengoreksi fg_color agar kontras terhadap bg_color memenuhi target_ratio.
        Pengubahan dilakukan dengan menggeser Lightness (di ruang HLS) secara perlahan.
        :param bg_color: Warna background (RGB 0-255)
        :param fg_color: Warna foreground (teks/elemen) (RGB 0-255)
        :return: tuple warna RGB baru untuk fg_color, boolean sukses_atau_tidak
        """
        current_ratio = self.calculate_contrast_ratio(bg_color, fg_color)
        if current_ratio >= self.target_ratio:
            return fg_color, True # Sudah aman, tidak perlu diheal
        
        # Konversi ke HLS (Hue, Lightness, Saturation) rentang 0-1
        h, l, s = colorsys.rgb_to_hls(fg_color[0]/255.0, fg_color[1]/255.0, fg_color[2]/255.0)
        bg_l = self.calculate_relative_luminance(bg_color)
        
        # Tentukan arah pergeseran Lightness
        # Jika background cenderung gelap, kita terangkan foreground (L bertambah)
        # Jika background cenderung terang, kita gelapkan foreground (L berkurang)
        step = 0.02
        direction = 1 if bg_l < 0.5 else -1
        
        new_l = l
        max_iterations = 50
        
        for _ in range(max_iterations):
            new_l += step * direction
            
            # Jaga agar lightness tetap dalam batas 0 dan 1
            if new_l <= 0.0:
                new_l = 0.0
            elif new_l >= 1.0:
                new_l = 1.0
                
            # Cek rasio dengan lightness baru
            r, g, b = colorsys.hls_to_rgb(h, new_l, s)
            healed_fg = [int(r * 255), int(g * 255), int(b * 255)]
            
            if self.calculate_contrast_ratio(bg_color, healed_fg) >= self.target_ratio:
                return healed_fg, True
            
            # Berhenti jika sudah mentok hitam/putih murni
            if new_l == 0.0 or new_l == 1.0:
                break
                
        return healed_fg, False # Mengembalikan warna semaksimal mungkin walau mungkin rasio sedikit kurang

    def process_palette(self, bg_color, palette):
        """
        Menerima 1 warna background utama dan sisa palet (misal 4 warna K-Means).
        Mengoreksi (heal) seluruh warna dalam palet agar kontras dengan background.
        """
        healed_palette = []
        for color in palette:
            new_color, _ = self.heal_color(bg_color, color)
            healed_palette.append(new_color)
            
        return healed_palette

# --- BLOK PENGUJIAN ---
if __name__ == "__main__":
    healer = WCAGAutoHeal(target_ratio=4.5)
    
    # Contoh kasus warna yang tidak lulus WCAG (dari catatan notebook-mu)
    background = [220, 220, 220] # Abu-abu terang
    teks_gagal = [150, 150, 150] # Abu-abu medium (Rasio sangat buruk)
    
    rasio_awal = healer.calculate_contrast_ratio(background, teks_gagal)
    print(f"Rasio Awal: {rasio_awal:.2f}:1")
    
    teks_healed, success = healer.heal_color(background, teks_gagal)
    rasio_akhir = healer.calculate_contrast_ratio(background, teks_healed)
    
    print(f"Warna Baru Teks: {teks_healed}")
    print(f"Rasio Setelah Auto-Heal: {rasio_akhir:.2f}:1")
    print(f"Status Lulus WCAG: {'✅' if success else '❌'}")