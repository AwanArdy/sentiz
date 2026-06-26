import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

st.set_page_config(layout="wide", page_title="Senti-Hormuz Dashboard")

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    .tag-box {
        display: inline-block;
        padding: 5px 12px;
        margin: 4px;
        background-color: #e9ecef;
        border-radius: 5px;
        color: #495057;
        font-weight: 500;
        font-size: 14px;
        border: 1px solid #dee2e6;
    }
    .header-style {
        font-size: 24px;
        font-weight: bold;
        color: #1e3a8a;
        margin-bottom: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_saved_models():
    vectorizer = joblib.load('vectorizer_tfidf.pkl')
    model_svm = joblib.load('model_svm.pkl')
    return vectorizer, model_svm

@st.cache_data
def load_main_data():
    df = pd.read_csv('Hasil_Labelling_Data_Inset_Lax.csv')
    return df

vectorizer_saved, model_svm_saved = load_saved_models()
df_utama = load_main_data()

# Dashboard
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.markdown('<p class="header-style">SENTI-HORMUZ</p>', unsafe_allow_html=True)
    st.caption("GLOBAL DASHBOARD")

with col_head2:
    st.write("")
    st.write(f"<div style='text-align: right; color: gray;'>● Online</div>", unsafe_allow_html=True)

# NAVIGASI TABS
tab1, tab2 = st.tabs(["Beranda", "Klasifikasi Data"])

# Tab 1 : Dashboard Global
with tab1:
    col_left, col_right = st.columns([1.5, 1], gap="large")

    with col_left:
        st.markdown("### **Dashboard**")
        
        # 1. Grafik Fluktuasi Sentimen (Berdasarkan Urutan Index)
        batch_size = 100 # Mengelompokkan setiap 100 komentar
        st.write(f"**Fluktuasi Sentimen (Per {batch_size} Komentar)**")
        
        # Membuat kolom 'Batch' berdasarkan urutan baris data
        df_utama['Batch'] = (df_utama.index // batch_size) + 1
        
        # Menghitung jumlah masing-masing sentimen per kelompok Batch
        df_tren = df_utama.groupby(['Batch', 'Sentiment']).size().reset_index(name='Jumlah')
        
        # Menggambar Line Chart
        fig_line = px.line(
            df_tren, 
            x='Batch', 
            y='Jumlah', 
            color='Sentiment', 
            markers=True, # Menambahkan titik di setiap pertemuan garis
            color_discrete_map={'Positif':'#28a745', 'Netral':'#ffc107', 'Negatif':'#dc3545'}
        )
        
        # Merapikan tampilan grafik
        fig_line.update_layout(
            height=350, 
            margin=dict(l=0, r=0, t=10, b=0), 
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Urutan Kelompok Komentar (Batch)",
            yaxis_title="Frekuensi"
        )
        st.plotly_chart(fig_line, width='stretch')

        st.write("---")

        # 2. Sumber Data Utama (Ikon-ikon)
        st.write("**Profil Pemrosesan Data**")
        c1, c2, col3 = st.columns(3)
        c1.metric("Platform Asal", "YouTube")
        c2.metric("Total Data Bersih", f"{len(df_utama)} Baris")
        col3.metric("Ekstraksi Fitur", "TF-IDF")

    # --- KOLOM KANAN (Donut Chart & Keywords) ---
    with col_right:
        st.markdown(f"### **Analisis Sentimen: Isu Selat Hormuz**")
        
        # 1. Donut Chart (Sesuai Mockup dengan Angka di Tengah)
        counts = df_utama['Sentiment'].value_counts()
        total_data = len(df_utama)
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=counts.index, 
            values=counts.values, 
            hole=.6,
            marker_colors=['#dc3545', '#ffc107', '#28a745'] # Merah, Kuning, Hijau
        )])
        
        fig_donut.update_layout(
            annotations=[dict(text=f'<b>{total_data}</b><br>Data Poin', x=0.5, y=0.5, font_size=18, showarrow=False)],
            showlegend=True,
            height=380,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

        # 2. Kata Kunci Terpopuler (Tags / Badges)
        st.write("**Kata Kunci Terpopuler (Negatif)**")
        # Ambil 10 kata tersering dari kolom stopword removal untuk sentimen negatif
        try:
            from collections import Counter
            text_neg = " ".join(df_utama[df_utama['Sentiment'] == 'Negatif']['stopword removal'].astype(str))
            top_words = Counter(text_neg.split()).most_common(8)
            
            tags_html = ""
            for word, freq in top_words:
                tags_html += f'<span class="tag-box">#{word}</span> '
            st.markdown(tags_html, unsafe_allow_html=True)
        except:
            st.write("Gagal memuat kata kunci.")

        st.write("---")
        
        # 3. Statistik Tambahan (Placeholder)
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            st.write("**Statistik Tambahan**")
            st.caption("Volume meningkat 12% minggu ini.")
        with col_sub2:
            st.write("**Analisis Regional**")
            st.caption("Fokus area: Selat Hormuz.")

with tab2:
    st.subheader("1. Input Data")
    uploaded_file = st.file_uploader("Pilih file dataset baru (.csv)", type=["csv"])

    st.subheader("2. Menjalankan Analisis")
    if uploaded_file is not None:
        st.info("Status: Dataset siap dianalisis secara live.")

        if st.button("Jalankan Analisis Komparatif", type="primary"):
            df_input = pd.read_csv(uploaded_file)

            with st.spinner("Memproses evaluasi seluruh skenario (SVM & Random Forest, Train-Test Split & Stratified 5-Fold CV)..."):
                # Menyesuaikan penamaan kolom & sentimen
                col_t_teks = 'stopword removal' if 'stopword removal' in df_input.columns else 'stop removal'
                col_t_sentimen = 'Sentiment' if 'Sentiment' in df_input.columns else 'sentiment'

                X = df_input[col_t_teks].fillna("").astype(str)
                y = df_input[col_t_sentimen]
                labels = np.unique(y)

                # Fungsi membuat pipeline (TF-IDF -> SMOTE -> Algoritma)
                def create_pipeline(model):
                    return ImbPipeline([
                        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
                        ('smote', SMOTE(random_state=42)),
                        ('model', model)
                    ])

                # Seluruh algoritma & metode evaluasi dijalankan otomatis (tanpa perlu dipilih user)
                models_to_run = {
                    "Support Vector Machine": SVC(C=10, kernel='linear', gamma='scale', random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=5, random_state=42)
                }

                ratio_map = {"Skenario 90:10": 0.1, "Skenario 80:20": 0.2, "Skenario 70:30": 0.3}

                results = []

                # --- Skenario Train-Test Split (90:10, 80:20, 70:30) ---
                for skenario_nama, test_size_val in ratio_map.items():
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_val, random_state=42, stratify=y
                    )
                    for model_name, model_obj in models_to_run.items():
                        pipeline = create_pipeline(model_obj)
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)

                        results.append({
                            "Skenario": skenario_nama,
                            "Model": model_name,
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                            "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
                            "F1-Score": f1_score(y_test, y_pred, average='macro', zero_division=0),
                            "cm": confusion_matrix(y_test, y_pred, labels=labels),
                            "labels": labels
                        })

                # --- Skenario Stratified 5-Fold CV (memakai seluruh data) ---
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                for model_name, model_obj in models_to_run.items():
                    pipeline = create_pipeline(model_obj)
                    y_pred = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=1)

                    results.append({
                        "Skenario": "Stratified 5-Fold CV",
                        "Model": model_name,
                        "Accuracy": accuracy_score(y, y_pred),
                        "Precision": precision_score(y, y_pred, average='macro', zero_division=0),
                        "Recall": recall_score(y, y_pred, average='macro', zero_division=0),
                        "F1-Score": f1_score(y, y_pred, average='macro', zero_division=0),
                        "cm": confusion_matrix(y, y_pred, labels=labels),
                        "labels": labels
                    })

            # Simpan ke session_state supaya hasil tidak hilang saat interaksi lain (mis. pilih skenario CM)
            st.session_state['app_results'] = results
            st.session_state['app_labels'] = labels

        # Tampilkan hasil kalau analisis sudah pernah dijalankan
        if 'app_results' in st.session_state:
            results = st.session_state['app_results']
            labels = st.session_state['app_labels']

            st.subheader("3. Hasil Evaluasi Kinerja")

            df_res = pd.DataFrame(results)
            df_res_tampil = df_res.drop(columns=['cm', 'labels'])

            st.markdown("##### **Ringkasan Seluruh Skenario & Algoritma**")
            st.table(df_res_tampil.set_index(["Skenario", "Model"]))

            # Grafik perbandingan akurasi seluruh kombinasi skenario x model
            fig_bar = px.bar(
                df_res_tampil,
                x="Skenario",
                y="Accuracy",
                color="Model",
                barmode="group",
                text_auto=".2f",
                color_discrete_map={
                    'Support Vector Machine': '#1f77b4',
                    'Random Forest': '#ff7f0e'
                }
            )
            fig_bar.update_layout(yaxis_range=[0, 1.1], yaxis_title="Accuracy")
            st.plotly_chart(fig_bar, use_container_width=True)

            # Confusion Matrix per skenario (dipilih lewat dropdown agar tidak terlalu padat)
            st.markdown("##### **Confusion Matrix**")
            skenario_pilihan = st.selectbox(
                "Pilih Skenario untuk Lihat Confusion Matrix",
                df_res['Skenario'].unique()
            )
            hasil_skenario = [r for r in results if r['Skenario'] == skenario_pilihan]

            cols_cm = st.columns(len(hasil_skenario))
            for idx, res in enumerate(hasil_skenario):
                with cols_cm[idx]:
                    st.write(f"**{res['Model']}**")

                    df_cm_table = pd.DataFrame(
                        res['cm'],
                        index=[f"Aktual {l}" for l in res['labels']],
                        columns=[f"Prediksi {l}" for l in res['labels']]
                    )
                    st.table(df_cm_table)

                    fig_cm = px.imshow(
                        res['cm'],
                        text_auto=True,
                        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                        x=res['labels'],
                        y=res['labels'],
                        color_continuous_scale='Blues'
                    )
                    fig_cm.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                    st.plotly_chart(fig_cm, use_container_width=True)

            # Kesimpulan
            st.write("---")
            st.markdown("##### **Kesimpulan Pengujian**")

            best_idx = df_res['Accuracy'].idxmax()
            best_row = df_res.loc[best_idx]

            st.success(
                f"**Model & Skenario Terbaik:** Dari seluruh kombinasi algoritma dan metode evaluasi yang diuji, "
                f"model **{best_row['Model']}** pada **{best_row['Skenario']}** memiliki performa terbaik "
                f"dengan tingkat akurasi sebesar **{best_row['Accuracy']*100:.2f}%**."
            )

            st.markdown("**Rincian Performa per Kombinasi Skenario & Algoritma:**")

            n_cols = 2
            rows_chunks = [results[i:i + n_cols] for i in range(0, len(results), n_cols)]
            for chunk in rows_chunks:
                cols_detail = st.columns(n_cols)
                for idx, res in enumerate(chunk):
                    with cols_detail[idx]:
                        st.markdown(f"""
                        <div style="background-color: #ffffff; padding: 15px; border-radius: 8px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.02); margin-bottom: 10px;">
                            <h6 style="color: #1e3a8a; margin-top: 0;"><b>{res['Model']}</b> — {res['Skenario']}</h6>
                            <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0;">
                                <li>🔹 <b>Akurasi:</b> <code style="color: #1e3a8a;">{res['Accuracy']*100:.2f}%</code></li>
                                <li>🔹 <b>Precision:</b> <code>{res['Precision']*100:.2f}%</code></li>
                                <li>🔹 <b>Recall:</b> <code>{res['Recall']*100:.2f}%</code></li>
                                <li>🔹 <b>F1-Score:</b> <code>{res['F1-Score']*100:.2f}%</code></li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
    else:
        st.warning("Status: Menunggu unggahan file dataset baru untuk pengujian.")
