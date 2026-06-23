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
tab1, tab2 = st.tabs(["DASHBOARD", "INPUT, ANALISIS & EVALUASI"])

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
    
    st.subheader("2. Pengaturan Analisis")
    col_algo, col_eval = st.columns(2)
    
    with col_algo:
        st.markdown("**Pilih Algoritma**")
        use_svm = st.checkbox("Support Vector Machine (SVC)", value=True)
        use_rf = st.checkbox("Random Forest")
        
    with col_eval:
        st.markdown("**Metode Evaluasi**")
        eval_method = st.radio("Pilih Metode Pengujian", ("Train-Test Split", "Stratified 5-Fold CV"), index=0)

        if eval_method == "Train-Test Split":
            split_ratio = st.selectbox("Rasio Data Pengujian", ("90:10", "80:20", "70:30"), index=1)
            ratio_map = {"90:10": 0.1, "80:20": 0.2, "70:30": 0.3}
            test_size_val = ratio_map[split_ratio]

        split_ratio = st.radio("Rasio Data Pengujian", ("90:10", "80:20", "70:30"), index=1)
        ratio_map = {"90:10": 0.1, "80:20": 0.2, "70:30": 0.3}
        test_size_val = ratio_map[split_ratio]

    st.subheader("3. Status")
    if uploaded_file is not None:
        st.info("Status: Dataset siap dianalisis secara live.")
        
        if st.button("Jalankan Analisis Komparatif", type="primary"):
            df_input = pd.read_csv(uploaded_file)
            
            with st.spinner("Memproses evaluasi dengan metode {eval_method}..."):
                # Menyesuaikan penamaan kolom & sentimen
                col_t_teks = 'stopword removal' if 'stopword removal' in df_input.columns else 'stop removal'
                col_t_sentimen = 'Sentiment' if 'Sentiment' in df_input.columns else 'sentiment'
                
                # Hanya mengambil data, tidak langsung fit_transform untuk menghindari data leakage
                X = df_input[col_t_teks].fillna("").astype(str)
                y = df.input[col_t_sentimen]
                labels = np.unique(y)

                # Fungsi membuat pipeline (TF-IDF -> SMOTE -> Algoritma)
                def create_pipeline(model):
                    return ImbPipeline([
                        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
                        ('smote', SMOTE(random_state=42)),
                        ('model', model)
                    ])

                models_to_run = {}
                if use_svm:
                    models_to_run["Support Vector Machine"] = SVC(C=10, kernel='linear', gamma='scale', random_state=42)
                if use_rf:
                    models_to_run["Random Forest"] = RandomForestClassifier(n_estimator=200, max_depth=None, min_samples_split=5, random_state=42)

                results = []
    
                for model_name, model_obj in models_to_run.items():
                    pipeline = create_pipeline(model_obj)

                    if eval_method == "Train-Test Split":
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42, stratify=y)
                        pipeline.fit(X_train, y_train)
                        y_pred = pipeline.predict(X_test)
                        y_actual = y_test

                    elif eval_method == "Stratified 5-Fold CV":
                        skf = StratifiedKFold(n_split=5, shuffle=True, random_state=42)
                        y_pred = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=1)
                        y_actual = y

                    # Mengkalkulasi matriks evaluasi
                    results.append({
                        "Model": model_name,
                        "Accuracy": accuracy_score(y_actual, y_pred),
                        "Precision": precision_score(y_actual, y_pred, average='macro', zero_division=0),
                        "Recall": recall_score(y_actual, y_pred, average='macro', zero_division=0),
                        "F1-Score": f1_score(y_actual, y_pred, average='macro', zero_division=0),
                        "cm": confusion_matrix(y_actual, y_pred, labels=labels),
                        "labels": labels
                    })

                st.subheader("4. Hasil Evaluasi Kinerja")
                if results:
                    # Tampilkan Confusion Matrix
                    st.markdown("##### **Confusion Matrix**")
                    cols_cm = st.columns(len(results))
                    for idx, res in enumerate(results):
                        with cols_cm[idx]:
                            st.write(f"**{res['Model']}**")
                            fig_cm = px.imshow(
                                rs['cm'],
                                text_auto=True,
                                labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
                                x=res['labels'],
                                y=res['labels'],
                                color_continuous_scale='Blues'
                            )
                            fig_cm.update_layout(margin=dict(l=0, r=0, t=10, b=0))
                            st.plotly_chart(fig_cm, use_container_width=True)

                    st.markdown("##### **Classification Report**")
                    df_res = pd.DataFrame(results)
                    st.table(df_res.drop(columns=['cm', 'labels']).set_index("Model"))

                    df_melt = df_res.drop(columns=['cm', 'labels']).melt(id_vars="Model", var_name="Metrics", value_name="Score")
                    fig_bar = px.bar(df_melt, x="Metrics", y="Score", color="Model", barmode="group", text_auto=".2f")
                    fig_bar.update_layout(yaxis_range=[0, 1.1])
                    st.plotly_chart(fig_bar, use_container_width=True)

                    # Kesimpulan
                    if len(results) > 1:
                        best_model = df_res.loc[df_res['Accuracy'].idxmax()]['Model']
                        eval_text = f"pembagian data {split_ratio}" if eval_method == "Train-Test Split" else "Stratified 5-Fold Cross Validation"
                        st.success(f"**Kesimpulan Pengujian:** Model **{best_model}** menghasilkan nilai akurasi tertinggi menggunakan metode evaluasi {eval_text}.")
                else:
                    st.warning("Silakan pilih minimal satu algoritma di atas.")
    else:
        st.warning("Status: Menunggu unggahan file dataset baru untuk pengujian.")
