# Panduan Membangun Dashboard Analisis Sentimen dengan Streamlit

Dokumen ini berisi panduan langkah demi langkah untuk mengubah proyek Jupyter Notebook "Analisis Sentimen Komentar Youtube" Anda menjadi sebuah web dashboard interaktif menggunakan **Streamlit**.

## Prasyarat (Persiapan Awal)

Sebelum membuat dashboard, pastikan Anda telah menyimpan (export) model Machine Learning dan Vectorizer (pembentuk fitur kata) dari Jupyter Notebook Anda. 

Tambahkan baris kode berikut di bagian paling akhir dari Jupyter Notebook Anda dan jalankan untuk menyimpan model:

```python
import joblib

# Ganti 'model_terbaik' dengan variabel model Anda yang memiliki akurasi paling bagus (misalnya clf_svm, clf_nb, dll)
joblib.dump(model_terbaik, 'model_sentimen.pkl')

# Ganti 'vectorizer' dengan variabel TF-IDF atau CountVectorizer Anda
joblib.dump(vectorizer, 'vectorizer.pkl')
```

## Langkah 1: Instalasi Library

Buka terminal/command prompt di direktori proyek Anda (`/home/ardy/Documents/sentimen analisis`), dan instal Streamlit beserta dependensi lainnya:

```bash
pip install streamlit pandas matplotlib seaborn wordcloud scikit-learn joblib Sastrawi
```

## Langkah 2: Buat File Aplikasi Streamlit

Buat sebuah file baru bernama `app.py` di dalam folder proyek Anda. File ini akan berisi seluruh kode antarmuka dashboard Anda.

## Langkah 3: Kerangka Kode `app.py`

Berikut adalah contoh kerangka kode (template) yang bisa Anda salin dan sesuaikan ke dalam file `app.py`:

```python
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
import re

# ----------------------------------------------------
# 1. Konfigurasi Halaman
# ----------------------------------------------------
st.set_page_config(page_title="Dashboard Sentimen Selat Hormuz", layout="wide")
st.title("📊 Dashboard Analisis Sentimen: Penutupan Selat Hormuz")
st.markdown("Dashboard ini menganalisis sentimen komentar YouTube mengenai isu penutupan Selat Hormuz.")

# ----------------------------------------------------
# 2. Load Model, Vectorizer, dan Data
# ----------------------------------------------------
@st.cache_resource # Mempercepat loading agar tidak diload ulang setiap ada interaksi
def load_model_and_data():
    # Load Model Machine Learning
    model = joblib.load('model_sentimen.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    
    # Load Data CSV
    df_hasil = pd.read_csv('Hasil_Preprocessing_Data.csv')
    df_eval = pd.read_csv('model_evaluation_results.csv')
    return model, vectorizer, df_hasil, df_eval

model, vectorizer, df_hasil, df_eval = load_model_and_data()

# ----------------------------------------------------
# 3. Sidebar untuk Navigasi Menu
# ----------------------------------------------------
menu = st.sidebar.selectbox("Pilih Menu", ["🏠 Beranda & Dataset", "📈 Visualisasi Data", "🤖 Uji Sentimen Baru"])

# ====================================================
# MENU 1: BERANDA & DATASET
# ====================================================
if menu == "🏠 Beranda & Dataset":
    st.header("Dataset Komentar")
    st.write("Menampilkan sampel data yang telah dipreprocessing dan dilabeli.")
    
    # Tampilkan dataframe
    st.dataframe(df_hasil.head(100))
    
    # Menampilkan total data
    st.info(f"Total komentar yang dianalisis: {len(df_hasil)} baris")

# ====================================================
# MENU 2: VISUALISASI DATA
# ====================================================
elif menu == "📈 Visualisasi Data":
    st.header("Visualisasi Hasil Analisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribusi Sentimen")
        # Sesuaikan 'sentimen' dengan nama kolom label Anda di CSV
        fig_dist = plt.figure(figsize=(8, 6))
        sns.countplot(x='sentimen', data=df_hasil, palette='viridis')
        plt.title('Jumlah Komentar per Sentimen')
        st.pyplot(fig_dist)
        
    with col2:
        st.subheader("Perbandingan Model")
        # Visualisasi hasil evaluasi model dari csv
        st.dataframe(df_eval)
        # Jika Anda ingin membuat bar chart dari model_evaluation_results.csv
        # st.bar_chart(df_eval.set_index('Model')['Accuracy'])

    st.markdown("---")
    st.subheader("☁️ WordCloud (Kata yang Sering Muncul)")
    
    # Pilihan sentimen untuk wordcloud
    pilihan_sentimen = st.selectbox("Pilih Sentimen untuk WordCloud:", df_hasil['sentimen'].unique())
    
    # Filter data berdasarkan sentimen
    teks_kumpulan = " ".join(df_hasil[df_hasil['sentimen'] == pilihan_sentimen]['teks_bersih'].dropna()) # sesuaikan nama kolom
    
    # Generate Wordcloud
    if teks_kumpulan:
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(teks_kumpulan)
        fig_wc = plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(fig_wc)
    else:
        st.warning("Tidak ada kata untuk ditampilkan pada sentimen ini.")

# ====================================================
# MENU 3: UJI SENTIMEN (PREDIKSI TEKS BARU)
# ====================================================
elif menu == "🤖 Uji Sentimen Baru":
    st.header("Uji Sentimen Teks Baru")
    st.write("Masukkan komentar atau teks terkait isu Selat Hormuz untuk mengetahui apakah sentimennya Positif, Negatif, atau Netral.")
    
    input_teks = st.text_area("Masukkan teks di sini:")
    
    if st.button("Analisis Sentimen"):
        if input_teks:
            # 1. Preprocessing sederhana (Case folding, dll)
            teks_bersih = input_teks.lower()
            teks_bersih = re.sub(r'[^a-zA-Z\s]', '', teks_bersih)
            # (Jika butuh stemming Sastrawi, masukkan logicnya di sini)
            
            # 2. Vectorization
            vektor_teks = vectorizer.transform([teks_bersih])
            
            # 3. Prediksi
            hasil_prediksi = model.predict(vektor_teks)[0]
            
            st.success(f"Hasil Prediksi Sentimen: **{hasil_prediksi.upper()}**")
        else:
            st.warning("Silakan masukkan teks terlebih dahulu.")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.markdown("---")
st.markdown("Dibuat menggunakan [Streamlit](https://streamlit.io/)")
```

## Langkah 4: Cara Menjalankan Dashboard

1. Pastikan Anda berada di dalam folder proyek (`/home/ardy/Documents/sentimen analisis`).
2. Jalankan perintah berikut di terminal:
   ```bash
   streamlit run app.py
   ```
3. Browser Anda akan terbuka secara otomatis dan menampilkan alamat `http://localhost:8501`.

## Catatan Penting Penyesuaian

- **Nama Kolom CSV:** Pastikan nama kolom pada `df_hasil['sentimen']` dan `df_hasil['teks_bersih']` disesuaikan dengan nama header kolom yang sebenarnya ada di file `Hasil_Preprocessing_Data.csv` Anda.
- **Teks Preprocessing saat Uji Baru:** Di Menu ke-3 (Uji Sentimen Baru), idealnya teks yang dimasukkan pengguna (input) harus melalui proses *cleansing, stopword removal, dan stemming* yang **sama persis** dengan yang Anda lakukan di Jupyter Notebook sebelum diubah menjadi vektor.

## Struktur Direktori Akhir yang Diharapkan
```text
sentimen analisis/
├── Analisis_Sentimen_Komentar_Youtube...ipynb
├── Hasil_Preprocessing_Data.csv
├── model_evaluation_results.csv
├── ...
├── app.py                <-- File Streamlit utama
├── model_sentimen.pkl    <-- File ekspor model terbaik Anda
└── vectorizer.pkl        <-- File ekspor TF-IDF/CountVectorizer Anda
```