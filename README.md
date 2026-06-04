## 📌 Deskripsi

Sistem deteksi plat nomor kendaraan Indonesia berbasis pengolahan citra digital. Gambar diproses melalui tahapan preprocessing, segmentasi, ekstraksi fitur geometri, klasifikasi KNN, dan validasi OCR untuk memastikan format plat sesuai standar Indonesia.

---

##  Fitur

- Preprocessing citra: Grayscale, Histogram Equalization, Gaussian Blur
- Segmentasi dengan Canny Edge Detection + Contour Detection
- Klasifikasi menggunakan K-Nearest Neighbors (KNN)
- Validasi format plat Indonesia via EasyOCR
- Visualisasi setiap tahapan preprocessing
- Antarmuka web berbasis Flask

---

##  Pipeline Pemrosesan

### 1. Preprocessing
| Tahap | Nama | Keterangan |
|-------|------|------------|
| 1 | Grayscale Conversion | Mengubah citra RGB menjadi grayscale |
| 2 | Histogram Equalization | Meningkatkan kontras citra secara global |
| 3 | Gaussian Blur | Mengurangi noise dengan kernel 5×5 |

```
INPUT → Grayscale → Histogram EQ → Gaussian Blur
```

### 2. Segmentasi
- **Canny Edge Detection** — mendeteksi tepi objek dan menonjolkan batas plat dari latar belakang
- **Contour Detection** — menemukan kandidat area plat, lalu diseleksi menggunakan aturan geometri

### 3. Ekstraksi Fitur
Lima fitur geometri diekstrak dari setiap kontur kandidat:

| Fitur | Keterangan |
|-------|------------|
| Area | Luas objek (px²) |
| Aspect Ratio | Rasio lebar / tinggi bounding box |
| Solidity | Kepadatan bentuk (area / convex hull) |
| Extent | Kepadatan terhadap bounding box |
| Perimeter Ratio | Karakteristik bentuk berdasarkan keliling |

### 4. Klasifikasi KNN
- **k = 5**, weight = distance
- Menghitung jarak fitur ke data referensi
- Memilih 5 tetangga terdekat
- Menentukan kelas berdasarkan mayoritas tetangga

**Output:**
- ✅ `Plat Indonesia Valid` — terdeteksi + OCR cocok format Indonesia
- ❌ `Bukan Plat Indonesia` — tidak sesuai format
- ⬜ `Plat Tidak Ditemukan` — tidak ada kandidat

---

## 📁 Struktur Proyek

```
Plat-Detection-master/
├── app.py                  # Aplikasi Flask
├── deteksi.py              # Pipeline utama PCD + KNN + OCR
├── requirements.txt        # Dependensi
├── static/
│   ├── style.css
│   └── results/            # Output gambar hasil deteksi
├── templates/
│   └── index.html
└── uploads/                # Gambar yang diupload
```

---

##  Instalasi

**1. Clone repositori**
```bash
git clone <url-repo>
cd Plat-Detection-master
```

**2. Install dependensi**
```bash
pip install -r requirements.txt
```

**3. Jalankan aplikasi**
```bash
python app.py
```

Buka browser: `http://localhost:5000`

> **Catatan:** Saat pertama kali dijalankan, EasyOCR akan mengunduh model (~100 MB). Pastikan koneksi internet tersedia.

---

##  Dependensi

| Library | Kegunaan |
|---------|----------|
| opencv-python | Pemrosesan citra |
| numpy | Operasi array & matriks |
| scikit-learn | Model KNN |
| easyocr | OCR validasi plat |
| flask | Antarmuka web |

---

## 📊 Hasil Pengujian

Pengujian dilakukan terhadap 10 sampel gambar:

| Sample | Ground Truth | Hasil Program |
|--------|-------------|---------------|
| 1 | Plat Indo | Plat Terdeteksi ✅ |
| 2 | Plat Luar | Plat Tidak Ditemukan ✅ |
| 3 | Plat Indo | Plat Terdeteksi ✅ |
| 4 | Plat Indo | Plat Terdeteksi ✅ |
| 5 | Plat Luar | Plat Tidak Ditemukan ✅ |
| 6 | Plat Indo | Plat Terdeteksi ✅ |
| 7 | Plat Luar | Plat Terdeteksi ❌ |
| 8 | Plat Luar | Plat Tidak Ditemukan ✅ |
| 9 | Plat Luar | Plat Terdeteksi ❌ |
| 10 | Plat Indo | Plat Tidak Ditemukan ❌ |

**Akurasi: 7/10 × 100% = 70%**

---

##  Evaluasi

**Kelebihan:**
- Cepat dan ringan
- Mudah dijalankan
- Tidak memerlukan GPU

**Kekurangan:**
- Sensitif terhadap pencahayaan
- Sulit jika plat tertutup objek
- Dataset latih masih sedikit

---

## 📝 Kesimpulan

1. Sistem berhasil mendeteksi plat nomor kendaraan menggunakan OpenCV
2. Tahapan preprocessing dan segmentasi membantu menemukan kandidat plat
3. Ekstraksi fitur geometri dan KNN dapat membedakan plat dan bukan plat
4. Sistem masih dapat ditingkatkan dengan dataset yang lebih banyak dan metode klasifikasi yang lebih kompleks
