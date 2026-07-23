import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title='Accident Risk Predictor', layout='wide')

@st.cache_resource
def load_and_prepare_all():
    file_path = 'Dataset Klasifikasi - Copy.csv'
    if not os.path.exists(file_path):
        st.error(f"ERROR: File '{file_path}' tidak ditemukan.")
        st.stop()

    try:
        df = pd.read_csv(file_path, sep=';', encoding='latin1')
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    df.columns = df.columns.str.strip().str.lower()
    df['kecepatan'] = pd.to_numeric(df['kecepatan'], errors='coerce').fillna(40).astype(int)
    df['age'] = df['age'].replace(['nan', '(blanks)', 'none'], 'tidak diketahui')

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.lower()

    features = [
        'cuaca', 'tipe cahaya', 'direction', 'kelas jalan',
        'geometri jalan', 'tipe jalan', 'kecepatan', 'kecamatan',
        'age', 'jenis kelamin', 'jenis kendaraan',
        'atribut_keselamatan', 'kepemilikan_sim'
    ]

    df = df.dropna(subset=['jenis luka'])
    X = pd.get_dummies(df[features], dtype=int)
    y = df['jenis luka']
    model_columns = X.columns

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_train, y_train)

    m_smote = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', random_state=42)
    m_smote.fit(X_res, y_res)

    m_brf = BalancedRandomForestClassifier(n_estimators=300, max_depth=10, sampling_strategy='all', random_state=42)
    m_brf.fit(X_train, y_train)

    return m_smote, m_brf, model_columns, df, features

with st.spinner('Menyiapkan Model dan Data...'):
    model_smote, model_brf, fitur_model, df_raw, features_list = load_and_prepare_all()

# --- HEADER UI ---
st.markdown("<div style='background: #f1c40f; padding: 25px; border-radius: 15px; text-align: center; color: #2c3e50;'><h1 style='margin:0;'>🚗 Accident Risk Predictor</h1><p style='font-weight: bold; font-size: 1.1rem;'>Sistem Estimasi Risiko Kecelakaan Kabupaten Gresik</p></div>", unsafe_allow_html=True)

# --- SIDEBAR INPUT ---
st.sidebar.header("📝 Parameter Input")
user_inputs = {}
for col in features_list:
    if df_raw[col].dtype == 'object':
        opsi = sorted(df_raw[col].unique())
        user_inputs[col] = st.sidebar.selectbox(f"Pilih {col.replace('_',' ').title()}", opsi)
    else:
        c_min, c_max, c_val = int(df_raw[col].min()), int(df_raw[col].max()), int(df_raw[col].median())
        user_inputs[col] = st.sidebar.slider(col.title(), c_min, c_max, c_val)

# --- AREA PREDIKSI ---
st.write(" ")
if st.button("MULAI ANALISIS RISIKO", use_container_width=True):
    # Preprocessing Input
    input_df = pd.DataFrame([user_inputs])
    for c in input_df.select_dtypes(include='object').columns:
        input_df[c] = input_df[c].astype(str).str.lower().str.strip()

    # Encoding
    input_encoded = pd.get_dummies(input_df).reindex(columns=fitur_model, fill_value=0)
    
    # Prediksi Hybrid
    p_acc = model_smote.predict_proba(input_encoded)[0]
    p_sen = model_brf.predict_proba(input_encoded)[0]
    p_comb = (p_acc * 0.70) + (p_sen * 0.30)
    p_boosted = np.array([p_comb[0]*1.3, p_comb[1]*2.0, p_comb[2]*1.1])
    p_final = p_boosted / np.sum(p_boosted)

    labels = [c.upper() for c in model_smote.classes_]
    results = dict(zip(labels, p_final))

    # Tampilan Hasil
    st.markdown("--- ")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.subheader("📊 Skor Probabilitas")
        for k, v in results.items():
            st.metric(label=k, value=f'{v*100:.2f}%')
            
    with c2:
        st.subheader("🎯 Visualisasi Risiko")
        color_map = {'KECELAKAAN RINGAN': '#27ae60', 'KECELAKAAN BERAT': '#f39c12', 'KECELAKAAN FATAL': '#e74c3c'}
        slice_colors = [color_map.get(lbl, '#95a5a6') for lbl in labels]
        
        fig, ax = plt.subplots(figsize=(8,8))
        ax.pie(p_final, labels=labels, autopct='%1.1f%%', colors=slice_colors, 
               startangle=140, shadow=True, textprops={'fontweight':'bold', 'fontsize': 12})
        st.pyplot(fig)