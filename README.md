# Plat-Detection
Aplikasi web untuk mendeteksi dan membaca plat nomor kendaraan secara otomatis menggunakan **OpenCV** dan **Flask**.

## Deskripsi
Plat-Detection adalah sistem deteksi plat nomor kendaraan berbasis web yang memungkinkan pengguna untuk mengunggah gambar kendaraan dan mendapatkan hasil deteksi plat nomor secara otomatis. Sistem ini dibangun menggunakan teknik pemrosesan citra dengan OpenCV.

## Struktur Project
Plat-Detection/
├── static/              # File statis (CSS, JS, gambar hasil)
├── templates/           # Template HTML (halaman web)
├── uploads/             # Folder penyimpanan gambar yang diunggah
├── app.py               # Entry point aplikasi Flask
├── detector.py          # Logika deteksi plat nomor (OpenCV)
└── requirements.txt     # Daftar dependensi Python

## Alur Kerja Sistem
Pengguna Upload Gambar
        ↓
  Flask (app.py) menerima request
        ↓
  Gambar disimpan ke /uploads
        ↓
  detector.py memproses gambar
        ↓
  [OpenCV Pipeline]
  1. Konversi ke Grayscale
  2. Noise Reduction (Gaussian Blur)
  3. Edge Detection (Canny)
  4. Deteksi Kontur Plat
  5. Crop & Ekstraksi Region Plat
        ↓
  Hasil dikembalikan ke Flask
        ↓
  Ditampilkan ke Pengguna

## Dataset
Dataset yang digunakan merupakan **dataset custom** yang dikumpulkan secara mandiri.

 Keterangan        Detail   

 Jenis             Custom / Dataset Sendiri            
 Format gambar     JPG / PNG                         
 Objek             Plat nomor kendaraan Indonesia      
 Kondisi gambar    Siang hari, berbagai sudut dan jarak 

> **Catatan:** Dataset tidak disertakan dalam repository ini. Untuk menjalankan pelatihan ulang, siapkan dataset sendiri di folder yang sesuai.

## Teknologi yang Digunakan
 Teknologi    Fungsi                              

 Python       Bahasa pemrograman utama            
 Flask        Web framework (backend)             
 OpenCV       Pemrosesan gambar & deteksi plat    
 HTML/CSS     Tampilan antarmuka pengguna         

##  Cara Instalasi & Menjalankan

# 1. Clone Repository

```bash
git clone https://github.com/Cloryne33/Plat-Detection.git
cd Plat-Detection
```
### 2. Install Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi

```bash
python app.py
```

### 4. Buka di Browser

```
http://localhost:5000
```
## Cara Penggunaan
1. Buka aplikasi di browser
2. Klik tombol **"Upload Gambar"**
3. Pilih foto kendaraan dari perangkatmu
4. Klik **"Deteksi"**
5. Hasil deteksi plat nomor akan muncul di layar

##  Penjelasan File Utama

### `app.py`
File utama Flask yang menangani routing, upload gambar, dan menampilkan hasil ke halaman web.

### `detector.py`
Berisi logika pemrosesan gambar menggunakan OpenCV:
- Preprocessing gambar (grayscale, blur, edge detection)
- Deteksi kontur plat nomor
- Ekstraksi dan crop area plat

### `requirements.txt`
Daftar library Python yang dibutuhkan agar aplikasi berjalan.
## Requirements
```
flask
opencv-python
numpy
```

---

##  Author

**Cloryne33**  
GitHub: [@Cloryne33](https://github.com/Cloryne33)
