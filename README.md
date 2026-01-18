# Hearin

Aplikasi Speech-to-Text (STT) berbasis web menggunakan Flask dan machine learning untuk mengkonversi audio menjadi teks.

## 📋 Deskripsi

Hearin adalah aplikasi web yang memungkinkan pengguna untuk mengkonversi rekaman audio atau file audio menjadi teks menggunakan model machine learning. Aplikasi ini dibangun dengan Flask sebagai backend dan menggunakan model STT yang telah dilatih.

## 🚀 Fitur

- 🎤 Konversi audio ke teks secara real-time
- 📁 Upload file audio untuk transkripsi
- 🤖 Model STT yang dapat dilatih ulang

## 📁 Struktur Proyek

```
Hearin/
├── models/
│   └── stt_model/          # Model Speech-to-Text yang telah dilatih
├── app.py                  # Aplikasi Flask utama
├── train_model.py          # Script untuk melatih model STT
└── requirements.txt        # Dependencies Python
```

## 🛠️ Instalasi

### Prerequisites

- Python 3.7 atau lebih tinggi
- pip (Python package installer)

### Langkah Instalasi

1. Clone repository ini:
```bash
git clone https://github.com/faqih2021/Hearin.git
cd Hearin
```

2. Buat virtual environment (opsional tapi direkomendasikan):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Penggunaan

### Menjalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di `http://localhost:5000` (atau port yang dikonfigurasi).

### Melatih Model

Jika Anda ingin melatih ulang model dengan dataset sendiri:

```bash
python train_model.py
```

## 📦 Dependencies

Dependencies lengkap dapat dilihat di file `requirements.txt`. Beberapa library utama yang digunakan:

- Flask - Web framework
- TensorFlow/PyTorch - Framework machine learning
- Librosa - Audio processing
- NumPy - Numerical computing
