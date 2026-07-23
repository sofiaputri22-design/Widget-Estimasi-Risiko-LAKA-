import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.combine import SMOTETomek

# Konfigurasi Dashboard
st.set_page_config(page_title='Accident Risk Predictor', layout='wide')

@st.cache_resource
def load_data_and_train():
    # Pastikan file CSV ini ada di folder yang sama saat di GitHub
    df = pd.read_csv('Dataset Klasifikasi - Copy.csv', sep=';', encoding='latin1')
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

    X_model = pd.get_dummies(df[selected_features], dtype=int)
    y_model = df['jenis luka']
    fitur_cols = X_model.columns

    X_train, _, y_train, _ = train_test_split(X_model, y_model, test_size=0.2, stratify=y_model, random_state=42)

    smt = SMOTETomek(random_state=42)
    X_res, y_res = smt.fit_resample(X_train, y_train)
    m_smote = RandomForestClassifier(n_estimators=500, max_depth=12, class_weight='balanced', random_state=42)
    m_smote.fit(X_res, y_res)

    m_brf = BalancedRandomForestClassifier(n_estimators=300, max_depth=10, sampling_strategy='all', random_state=42)
    m_brf.fit(X_train, y_train)

    return m_smote, m_brf, fitur_cols, df, selected_features

with st.spinner('Menginisialisasi AI...'):
    model_smote, model_brf, fitur_model, df_raw, features = load_data_and_train()

st.markdown("<div style='background: linear-gradient(135deg, #f1c40f 0%, #f39c12 100%); padding: 20px; border-radius: 15px; text-align: center; color: #2c3e50;'><h1>🚗 Accident Risk Predictor</h1><p>Estimasi Tingkat Keparahan Kecelakaan Kab. Gresik</p></div>", unsafe_allow_html=True)

st.sidebar.header("📝 Input Parameter")
inputs = {}
for col in features:
    if df_raw[col].dtype == 'object':
        inputs[col] = st.sidebar.selectbox(col.replace('_',' ').title(), sorted(df_raw[col].unique()))
    else:
        inputs[col] = st.sidebar.slider(col.title(), int(df_raw[col].min()), int(df_raw[col].max()), int(df_raw[col].median()))

st.write(" ")
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

    res = dict(zip(model_smote.classes_, p_final))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📊 Probabilitas")
        for k, v in res.items():
            st.metric(label=k.upper(), value=f'{v*100:.1f}%')
    with c2:
        st.subheader("🎯 Visualisasi")
        fig, ax = plt.subplots()
        ax.pie(p_final, labels=[c.upper() for c in model_smote.classes_], autopct='%1.1f%%', colors=['#f39c12', '#e74c3c', '#27ae60'])
        st.pyplot(fig)