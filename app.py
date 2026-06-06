import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide", page_title="Senti-Hormuz Dashboard")

@st.cache_resource
def load_saved_models():
    vectorizer = joblib.load('vectorizer_tfidf.pkl')
    model_svm = joblib.load('model_svm.pkl')
    return vectorizer, model_svm

@st.cache_data
def load_main_data():
    df = pd.read_csv('Hasil_Preprocessing_Data.csv')
    return df

vectorizer_saved, model_svm_saved = load_saved_models()
df_utama = load_main_data()

# Dashboard
col_logo, col_status = st.columns([4, 1])
with col_logo:
    st.title("SENTI-HORMUZ")
    st.caption("Dashboard analisis sentimen isu penutupan selat Hormuz")
with col_status:
    st.write("")
    st.success("System Active")

st.markdown("---")

# Navigasi tab
tab1, tab2 = st.tabs(["Dashboard Global", "Input, Analisis dan Evaluasi"])

# Tab 1 : Dashboard Global
with tab1:
    st.subheader("Analisis Sentimen Global")

    col_pos, col_net, col_neg = st.columns(3)

    nama_kolom_sentimen = 'sentimen'
    counts = df_utama[nama_kolom_sentimen].value_counts()

    with col_pos:
        st.info(f"😊 Positif\n\n## {counts.get('positif', 0)}")
    with col_net:
        st.warning(f"😐 Netral\n\n## {counts.get('netral', 0)}")
    with col_neg:
        st.error(f"😡 Negatif\n\n## {counts.get('negatif', 0)}")

    st.markdown("###")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("#### **Grafik Tren Sentimen**")

        if 'tanggal' in df_utama.columns:
            fig_line = px.line(df_utama, x='tanggal', color=nama_kolom_sentimen, title="Tren Pergerakan Sentimen")
        else:
            fig_line = px.bar(df_utama, x=nama_kolom_sentimen, color=nama_kolom_sentimen, title="Distribusi kelas sentimen")
        st.plotly_chart(fig_line, use_container_width=True)

        st.markdown("### **Sumber Data Utama**")
        st.code("Youtube Komentar | File: Hasil_Preprocessing_Data.csv")

    with col_right:
        st.markdown("#### **Kata Kunci Terpopuler (Negatif)**")
        nama_kolom_teks = 'text_clean' if 'text_clean' in df_utama.columns else 'komentar'
        
        text_negatif = " ".join(df_utama[df_utama[nama_kolom_sentimen] == 'negatif'][nama_kolom_teks].astype(str))
        if text_negatif:
            wordcloud = WordCloud(width=400, height=220, background_color='white').generate(text_negatif)
            fig, ax = plt.subplots(figsize=(6, 3.3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.write("Tidak ada kata kunci negatif yang ditemukan.")

with tab2:
    st.subheader("1. Input Data")
    uploaded_file = st.file_uploader("Pilih file dataset baru (.csv)", type=["csv"])
    
    st.subheader("2. Pengaturan Analisis")
    col_algo, col_split = st.columns(2)
    
    with col_algo:
        st.markdown("**Pilih Algoritma**")
        use_svm = st.checkbox("Support Vector Machine (LinearSVC)", value=True)
        use_rf = st.checkbox("Random Forest")
        
    with col_split:
        st.markdown("**Pembagian Data (Train:Test)**")
        split_ratio = st.radio("Rasio Data Pengujian", ("90:10", "80:20", "70:30"), index=1)
        ratio_map = {"90:10": 0.1, "80:20": 0.2, "70:30": 0.3}
        test_size_val = ratio_map[split_ratio]

    st.subheader("3. Status")
    if uploaded_file is not None:
        st.info("Status: Dataset siap dianalisis secara live.")
        
        if st.button("Jalankan Analisis Komparatif", type="primary"):
            df_input = pd.read_csv(uploaded_file)
            
            with st.spinner("Memproses ekstraksi fitur TF-IDF dan pelatihan model..."):
                col_t_teks = 'text_clean' if 'text_clean' in df_input.columns else 'komentar'
                col_t_sentimen = 'sentimen' if 'sentimen' in df_input.columns else 'label'
                
                from sklearn.feature_extraction.text import TfidfVectorizer
                vec_live = TfidfVectorizer()
                X = vec_live.fit_transform(df_input[col_t_teks].astype(str))
                y = df_input[col_t_sentimen]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)
                
                results = []
                
                if use_svm:
                    clf_svm = LinearSVC()
                    clf_svm.fit(X_train, y_train)
                    y_pred = clf_svm.predict(X_test)
                    results.append({
                        "Model": "Support Vector Machine",
                        "Accuracy": accuracy_score(y_test, y_pred),
                        "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
                        "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
                        "F1-Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
                    })
                    
                if use_rf:
                    clf_rf = RandomForestClassifier(random_state=42)
                    clf_rf.fit(X_train, y_train)
                    y_pred_rf = clf_rf.predict(X_test)
                    results.append({
                        "Model": "Random Forest",
                        "Accuracy": accuracy_score(y_test, y_pred_rf),
                        "Precision": precision_score(y_test, y_pred_rf, average='macro', zero_division=0),
                        "Recall": recall_score(y_test, y_pred_rf, average='macro', zero_division=0),
                        "F1-Score": f1_score(y_test, y_pred_rf, average='macro', zero_division=0)
                    })
                
                st.subheader("4. Hasil Evaluasi Kinerja")
                if results:
                    df_res = pd.DataFrame(results)
                    st.table(df_res.set_index("Model"))
                    
                    df_melt = df_res.melt(id_vars="Model", var_name="Metrics", value_name="Score")
                    fig_bar = px.bar(df_melt, x="Metrics", y="Score", color="Model", barmode="group", text_auto=".2f")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    if len(results) > 1:
                        best_model = df_res.loc[df_res['Accuracy'].idxmax()]['Model']
                        st.success(f"**Kesimpulan Pengujian:** Model **{best_model}** menghasilkan nilai akurasi tertinggi pada pembagian data {split_ratio}.")
                else:
                    st.warning("Silakan pilih minimal satu algoritma di atas.")
    else:
        st.warning("Status: Menunggu unggahan file dataset baru untuk pengujian.")
