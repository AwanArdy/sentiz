# Analisis Sentimen Publik Terhadap Isu Penutupan Selat Hormuz

Proyek ini adalah sistem Analisis Sentimen (Sentiment Analysis) yang menggunakan metode *Machine Learning* yaitu **Support Vector Machine (SVM)** dan **Random Forest** untuk menganalisis opini publik (komentar YouTube) terkait isu penutupan Selat Hormuz oleh Iran.

Proyek ini dilengkapi dengan:
1. **Jupyter Notebook:** Untuk proses *Crawling*, *Preprocessing*, *Training Model*, dan *Evaluation*.
2. **Streamlit Dashboard:** Antarmuka web interaktif (UI) untuk melihat visualisasi data dan menguji teks komentar baru secara *real-time*.

---

## 🛠️ Prasyarat Sistem

Sebelum menjalankan proyek ini di komputer lokal Anda, pastikan Anda telah menginstal:
*   **Python 3.8** atau versi yang lebih baru ([Download Python](https://www.python.org/downloads/))
*   **Git** (Opsional, jika Anda melakukan *clone* repository)

---

## ⚙️ Cara Instalasi
### 1. Buat Virtual Environment (Sangat Direkomendasikan)
Untuk mencegah bentrok antar versi *library* dengan proyek lain, buatlah *virtual environment* Python:
```bash
# Untuk Windows:
python -m venv .venv
.venv\Scripts\activate

# Untuk Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instal Dependensi (*Libraries*)
Instal semua *library* Python yang dibutuhkan dengan menjalankan perintah berikut:
```bash
pip install -r requirements.txt
```
*(Jika file `requirements.txt` belum ada, Anda bisa menjalankan perintah ini: `pip install streamlit pandas joblib plotly wordcloud matplotlib scikit-learn Sastrawi nltk google-api-python-client seaborn`)*

---

## 🚀 Cara Menjalankan Proyek

### A. Menjalankan Dashboard Interaktif (Streamlit)
Dashboard ini digunakan untuk melihat hasil analisis dan mencoba prediksi model secara langsung.

1.  Pastikan Anda berada di direktori proyek dan *virtual environment* sedang aktif.
2.  Jalankan perintah berikut:
    ```bash
    streamlit run app.py
    ```
3.  Browser Anda akan otomatis terbuka di alamat `http://localhost:8501`. Jika tidak, salin dan tempel alamat tersebut ke browser web Anda.

### B. Menjalankan Ulang Pemodelan Data (Jupyter Notebook)
Jika Anda ingin melihat cara data dibersihkan, melatih ulang model, atau mengambil data YouTube baru:

1.  Jalankan perintah berikut di terminal:
    ```bash
    jupyter notebook
    ```
2.  Di browser yang terbuka, klik file `Analisis_Sentimen_Komentar_Youtube_Tentang_Penutupan_Selat_Hormuz.ipynb`.
3.  Anda dapat menjalankan blok kode satu per satu. **Catatan:** Untuk proses *Crawling*, Anda mungkin memerlukan API Key YouTube Data V3 yang valid.

---

## 📁 Struktur Direktori Penting

*   `app.py` : File utama aplikasi web Streamlit.
*   `Analisis_Sentimen_Komentar_Youtube...ipynb` : Source code *Machine Learning* dan *Data Science*.
*   `model_svm.pkl` & `vectorizer_tfidf.pkl` : File *Machine Learning* yang telah dilatih dan disimpan untuk digunakan oleh dashboard.
*   `Hasil_Preprocessing_Data.csv` : Dataset hasil *cleaning* dan label yang siap dianalisis.
*   `PANDUAN_STREAMLIT.md` : Dokumentasi awal pembuatan dashboard.

---
