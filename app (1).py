import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# Konfigurasi Dashboard
st.set_page_config(page_title='Accident Risk Predictor', layout='wide')

@st.cache_resource
def load_data_and_train():
    file_path = 'Dataset Klasifikasi - Copy.csv'
    if not os.path.exists(file_path):
        st.error(f"File '{file_path}' tidak ditemukan! Pastikan file sudah diunggah ke GitHub.")
        st.stop()

    df = pd.read_csv(file_path, sep=';', encoding='latin1')
    df.columns = df.columns.str.strip().str.lower()

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.lower()

    df['age'] = df['age'].replace(['nan', '(blanks)', 'none'], 'tidak diketahui')
    df['kecepatan'] = pd.to_numeric(df['kecepatan'], errors='coerce').fillna(40)

    selected_features = [
        'cuaca', 'tipe cahaya', 'direction', 'kelas jalan',
        'geometri jalan', 'tipe jalan', 'kecepatan', 'kecamatan',
        'age', 'jenis kelamin', 'jenis kendaraan',
        'atribut_keselamatan', 'kepemilikan_sim'
    ]

    df = df.dropna(subset=['jenis luka'])
    X_model = pd.get_dummies(df[selected_features], dtype=int)
    y_model = df['jenis luka']
    fitur_cols = X_model.columns

    # Fix Error 1: Stratification fail if class count < 2
    try:
        X_train, _, y_train, _ = train_test_split(X_model, y_model, test_size=0.2, stratify=y_model, random_state=42)
    except:
        X_train, _, y_train, _ = train_test_split(X_model, y_model, test_size=0.2, random_state=42)

    # Fix Error 2: SMOTE fail if k_neighbors > n_samples
    min_samples = y_train.value_counts().min()
    k_neigh = min(5, max(1, min_samples - 1))
    
    try:
        smt = SMOTETomek(smote=SMOTE(k_neighbors=k_neigh, random_state=42), random_state=42)
        X_res, y_res = smt.fit_resample(X_train, y_train)
    except:
        X_res, y_res = X_train, y_train

    # Modeling
    m_smote = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', random_state=42)
    m_smote.fit(X_res, y_res)

    m_brf = BalancedRandomForestClassifier(n_estimators=300, max_depth=10, sampling_strategy='all', replacement=True, random_state=42)
    m_brf.fit(X_train, y_train)

    return m_smote, m_brf, fitur_cols, df, selected_features

with st.spinner('Menginisialisasi AI...'):
    model_smote, model_brf, fitur_model, df_raw, features = load_data_and_train()

st.markdown("<div style='background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%); padding: 20px; border-radius: 15px; text-align: center; color: #2c3e50;'><h1>🚗 Accident Risk Predictor</h1><p>Estimasi Tingkat Keparahan Kecelakaan Kab. Gresik</p></div>", unsafe_allow_html=True)

st.sidebar.header("📝 Input Parameter")
inputs = {}
for col in features:
    if df_raw[col].dtype == 'object':
        options = sorted(df_raw[col].unique())
        inputs[col] = st.sidebar.selectbox(col.replace('_',' ').title(), options)
    else:
        inputs[col] = st.sidebar.slider(col.title(), int(df_raw[col].min()), int(df_raw[col].max()), int(df_raw[col].median()))

if st.button("PREDIKSI RISIKO", use_container_width=True):
    data_in = pd.DataFrame([inputs])
    for c in data_in.columns:
        if data_in[c].dtype == 'object':
            data_in[c] = data_in[c].astype(str).str.lower().str.strip()

    data_enc = pd.get_dummies(data_in).reindex(columns=fitur_model, fill_value=0)

    p_acc = model_smote.predict_proba(data_enc)[0]
    p_sen = model_brf.predict_proba(data_enc)[0]

    p_comb = (p_acc * 0.70) + (p_sen * 0.30)
    p_boost = np.array([p_comb[0]*1.3, p_comb[1]*2.0, p_comb[2]*1.1])
    p_final = p_boost / np.sum(p_boost)

    classes = model_smote.classes_
    res = dict(zip(classes, p_final))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Probabilitas")
        for k, v in res.items():
            st.metric(label=k.upper(), value=f'{v*100:.1f}%')
    with c2:
        st.subheader("🎯 Visualisasi")
        fig, ax = plt.subplots()
        ax.pie(p_final, labels=[c.upper() for c in classes], autopct='%1.1f%%', colors=['#f39c12', '#e74c3c', '#27ae60'])
        st.pyplot(fig)