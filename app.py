# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Import dengan error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from streamlit_option_menu import option_menu
    MENU_AVAILABLE = True
except ImportError:
    MENU_AVAILABLE = False

try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMinMax, TimeSeriesScalerMeanVariance
    from tslearn.utils import to_time_series_dataset
    from tslearn.metrics import cdist_dtw
    TSL_AVAILABLE = True
except ImportError:
    TSL_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Komparasi Clustering Pengangguran Terbuka",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        text-align: center;
        padding: 0.9rem 1.5rem 1.1rem 1.5rem;
        border-bottom: 1px solid #e3e8ee;
        margin-bottom: 1.6rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        padding: 1rem 0;
        border-bottom: 2px solid #ecf0f1;
        margin-bottom: 1rem;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 1rem;
        color: #7f8c8d;
        margin-top: 0.5rem;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        padding: 3rem;
        text-align: center;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 2rem;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 1rem 0;
        border-top: 1px solid #ecf0f1;
        margin-top: 2rem;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #2c3e50;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #ecf0f1;
        margin-bottom: 0.8rem;
    }
    .preprocessing-box {
        background-color: #f0f4f8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-header">'
    '📊 Sistem Komparasi Clustering Pengangguran Terbuka'
    '<br><small style="color: #7f8c95;">Berdasarkan Tingkat Pendidikan di Jawa Barat</small>'
    '</div>',
    unsafe_allow_html=True
)

# ==================== SESSION STATE ====================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.pendidikan_order = None
    
    # Hasil K-Means
    st.session_state.km_results = None
    
    # Hasil Time Series K-Means
    st.session_state.ts_results = None

# ==================== FUNGSI PREPROCESSING ====================

def preprocess_data(data):
    """Preprocessing awal untuk kedua model"""
    # Filter tahun 2020-2025
    data = data[(data['tahun'] >= 2020) & (data['tahun'] <= 2025)]
    
    # Gabungkan kategori pendidikan
    data['pendidikan'] = data['pendidikan'].replace(
        ['TIDAK/BELUM PERNAH SEKOLAH/TIDAK/BELUM TAMAT SD', 'SD'],
        'SD KE BAWAH'
    )
    
    pendidikan_order = [
        'SD KE BAWAH', 
        'SMP', 
        'SMA (UMUM)', 
        'SMA (KEJURUAN)', 
        'DIPLOMA I/II/III/AKADEMI/UNIVERSITAS'
    ]
    data['pendidikan'] = pd.Categorical(data['pendidikan'], categories=pendidikan_order, ordered=True)
    
    return data, pendidikan_order

def get_kmeans_preprocessing(data, pendidikan_order):
    """
    Preprocessing untuk K-Means
    Output: (162 objek x 5 fitur) dengan normalisasi global
    """
    # Pivot: (kab/kota, tahun) x pendidikan
    pivot = data.pivot_table(
        index=['nama_kabupaten_kota', 'tahun'],
        columns='pendidikan',
        values='jumlah_pengangguran',
        aggfunc='mean',
        observed=True
    )
    pivot = pivot[pendidikan_order]
    
    # Interpolasi jika ada missing
    if pivot.isnull().values.any():
        pivot = pivot.interpolate(axis=0, limit_direction='both')
    
    # Normalisasi GLOBAL (semua nilai digabung)
    scaler = MinMaxScaler()
    X_flat = pivot.values.reshape(-1, 1)
    X_scaled_flat = scaler.fit_transform(X_flat)
    X = X_scaled_flat.reshape(pivot.shape)
    
    return {
        'pivot': pivot,
        'X': X,
        'scaler': scaler,
        'shape': pivot.shape,
        'description': f"{pivot.shape[0]} objek (kab/kota × tahun) × {pivot.shape[1]} fitur pendidikan"
    }

def get_ts_preprocessing(data, pendidikan_order, scaler_type='minmax'):
    """
    Preprocessing untuk Time Series K-Means
    Output: (135 deret waktu x 6 tahun) dengan normalisasi per-deret
    """
    if not TSL_AVAILABLE:
        return None
    
    tahun_order = sorted(data['tahun'].unique())
    
    # Pivot: (kab/kota, pendidikan) x tahun
    pivot = data.pivot_table(
        index=['nama_kabupaten_kota', 'pendidikan'],
        columns='tahun',
        values='jumlah_pengangguran',
        aggfunc='mean'
    )
    pivot = pivot[tahun_order]
    
    # Interpolasi jika ada missing
    if pivot.isnull().values.any():
        pivot = pivot.interpolate(axis=1, limit_direction='both')
    
    # Bentuk array 3D untuk tslearn
    X_ts_raw = to_time_series_dataset(pivot.values)
    
    # Normalisasi PER-DERET
    if scaler_type == 'minmax':
        scaler_ts = TimeSeriesScalerMinMax()
    else:
        scaler_ts = TimeSeriesScalerMeanVariance()
    
    X_ts = scaler_ts.fit_transform(X_ts_raw)
    
    return {
        'pivot': pivot,
        'X_ts': X_ts,
        'X_ts_raw': X_ts_raw,
        'scaler': scaler_ts,
        'shape': pivot.shape,
        'tahun_order': tahun_order,
        'description': f"{pivot.shape[0]} deret waktu (kab/kota × pendidikan) × {pivot.shape[1]} tahun"
    }

# ==================== FUNGSI K-MEANS ====================

def run_kmeans(data, pendidikan_order, k_value=None):
    """Menjalankan K-Means Clustering"""
    # Preprocessing K-Means
    prep = get_kmeans_preprocessing(data, pendidikan_order)
    X = prep['X']
    pivot = prep['pivot']
    
    # Tentukan k optimal jika tidak ditentukan
    if k_value is None:
        K_range = range(2, 10)
        inertia_list = []
        silhouette_list = []
        dbi_list = []
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            inertia_list.append(kmeans.inertia_)
            silhouette_list.append(silhouette_score(X, labels))
            dbi_list.append(davies_bouldin_score(X, labels))
        
        # Voting untuk menentukan k optimal
        best_silhouette_k = K_range[np.argmax(silhouette_list)]
        best_dbi_k = K_range[np.argmin(dbi_list)]
        
        inertia_diff = np.diff(inertia_list)
        inertia_ratio = inertia_diff[1:] / inertia_diff[:-1]
        best_elbow_k = list(K_range)[np.argmin(inertia_ratio) + 1]
        
        vote_counts = pd.Series([best_elbow_k, best_silhouette_k, best_dbi_k]).value_counts()
        k_value = vote_counts.index[0]
    
    # Clustering final
    kmeans_final = KMeans(n_clusters=k_value, random_state=42, n_init=10)
    labels = kmeans_final.fit_predict(X)
    
    # Hasil
    hasil = pivot.reset_index()
    hasil['cluster'] = labels
    
    sil_score = silhouette_score(X, labels)
    dbi_score = davies_bouldin_score(X, labels)
    
    return {
        'X': X,
        'labels': labels,
        'hasil': hasil,
        'pivot': pivot,
        'k': k_value,
        'silhouette': sil_score,
        'dbi': dbi_score,
        'model': kmeans_final,
        'scaler': prep['scaler'],
        'preprocessing': prep
    }

# ==================== FUNGSI TIME SERIES K-MEANS ====================

def run_timeseries_kmeans(data, pendidikan_order, k_value=None, scaler_type='minmax'):
    """Menjalankan Time Series K-Means Clustering"""
    if not TSL_AVAILABLE:
        return None
    
    # Preprocessing Time Series
    prep = get_ts_preprocessing(data, pendidikan_order, scaler_type)
    if prep is None:
        return None
    
    X_ts = prep['X_ts']
    pivot = prep['pivot']
    X_ts_flat = X_ts.reshape(X_ts.shape[0], X_ts.shape[1])
    
    labels_id = pivot.reset_index()[['nama_kabupaten_kota', 'pendidikan']]
    
    # Tentukan k optimal jika tidak ditentukan
    if k_value is None:
        K_range_ts = range(2, 10)
        inertia_ts = []
        silhouette_ts = []
        dbi_ts = []
        
        for k in K_range_ts:
            model = TimeSeriesKMeans(
                n_clusters=k,
                metric="dtw",
                random_state=42,
                n_init=5,
                max_iter=50,
                n_jobs=-1
            )
            labels_k = model.fit_predict(X_ts)
            inertia_ts.append(model.inertia_)
            
            dist_matrix = cdist_dtw(X_ts)
            sil = silhouette_score(dist_matrix, labels_k, metric="precomputed")
            silhouette_ts.append(sil)
            
            dbi = davies_bouldin_score(X_ts_flat, labels_k)
            dbi_ts.append(dbi)
        
        # Voting untuk menentukan k optimal
        best_sil_k = K_range_ts[np.argmax(silhouette_ts)]
        best_dbi_k = K_range_ts[np.argmin(dbi_ts)]
        
        inertia_diff_ts = np.diff(inertia_ts)
        inertia_ratio_ts = inertia_diff_ts[1:] / inertia_diff_ts[:-1]
        best_elbow_k = list(K_range_ts)[np.argmin(inertia_ratio_ts) + 1]
        
        vote_counts = pd.Series([best_elbow_k, best_sil_k, best_dbi_k]).value_counts()
        k_value = vote_counts.index[0]
    
    # Clustering final
    model_final = TimeSeriesKMeans(
        n_clusters=k_value,
        metric="dtw",
        random_state=42,
        n_init=15,
        max_iter=50,
        n_jobs=-1
    )
    labels = model_final.fit_predict(X_ts)
    
    # Hasil
    hasil = labels_id.copy()
    hasil['cluster'] = labels
    
    dist_matrix_final = cdist_dtw(X_ts)
    sil_score = silhouette_score(dist_matrix_final, labels, metric="precomputed")
    dbi_score = davies_bouldin_score(X_ts_flat, labels)
    
    return {
        'X_ts': X_ts,
        'X_ts_flat': X_ts_flat,
        'labels': labels,
        'hasil': hasil,
        'pivot': pivot,
        'k': k_value,
        'silhouette': sil_score,
        'dbi': dbi_score,
        'model': model_final,
        'scaler': prep['scaler'],
        'tahun_order': prep['tahun_order'],
        'preprocessing': prep
    }

# ==================== SIDEBAR ====================

with st.sidebar:
    st.markdown("### MENU")
    
    if MENU_AVAILABLE:
        selected = option_menu(
            menu_title=None,
            options=[
                "Beranda", 
                "Upload Dataset", 
                "Preprocessing",
                "Exploratory Data Analysis",
                "K-Means Clustering", 
                "Time Series K-Means",
                "Komparasi", 
                "Visualisasi", 
                "Unduh Hasil",
                "Tentang"
            ],
            icons=[
                "house", 
                "cloud-upload", 
                "gear",
                "bar-chart",
                "diagram-3", 
                "clock-history",
                "shuffle", 
                "graph-up", 
                "download",
                "info-circle"
            ],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#1f77b4", "font-size": "15px"},
                "nav-link": {"font-size": "13.5px", "text-align": "left", "margin": "1px 0",
                            "padding": "8px 10px", "border-radius": "6px",
                            "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#1f77b4", "font-weight": "500"},
            }
        )
    else:
        menu_options = [
            "Beranda", "Upload Dataset", "Preprocessing", "Exploratory Data Analysis",
            "K-Means Clustering", "Time Series K-Means",
            "Komparasi", "Visualisasi", "Unduh Hasil", "Tentang"
        ]
        selected = st.selectbox("Pilih Menu", menu_options)

# ==================== MENU: BERANDA ====================

if selected == "Beranda":
    st.markdown('<div class="sub-header">🏠 Beranda</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 Sistem Komparasi Clustering Pengangguran Terbuka
        
        Aplikasi ini digunakan untuk melakukan analisis clustering menggunakan 
        **K-Means** dan **Time Series K-Means (DTW)** serta membandingkan hasil 
        kedua algoritma.
        
        #### 🔍 Fitur Utama:
        1. **Upload Dataset** - Unggah file Excel
        2. **Preprocessing** - Lihat proses preprocessing kedua model
        3. **EDA** - Analisis eksploratif data
        4. **K-Means** - Clustering dengan K-Means
        5. **Time Series K-Means** - Clustering dengan DTW
        6. **Komparasi** - Perbandingan kedua algoritma
        7. **Visualisasi** - Visualisasi interaktif
        8. **Unduh Hasil** - Download hasil
        """)
    
    with col2:
        st.markdown("""
        #### 📋 Format Data
        
        **File:** .xlsx
        
        **Kolom yang diperlukan:**
        - `nama_kabupaten_kota`
        - `tahun` (2020-2025)
        - `pendidikan`
        - `jumlah_pengangguran`
        """)
        
        if st.button("📥 Gunakan Data Contoh", use_container_width=True):
            with st.spinner("Membuat dataset contoh..."):
                np.random.seed(42)
                
                kabupaten = [
                    'KABUPATEN BOGOR', 'KABUPATEN SUKABUMI', 'KABUPATEN CIANJUR',
                    'KABUPATEN BANDUNG', 'KABUPATEN GARUT', 'KABUPATEN TASIKMALAYA',
                    'KABUPATEN CIAMIS', 'KABUPATEN KUNINGAN', 'KABUPATEN CIREBON',
                    'KABUPATEN MAJALENGKA', 'KABUPATEN SUMEDANG', 'KABUPATEN INDRAMAYU',
                    'KABUPATEN SUBANG', 'KABUPATEN PURWAKARTA', 'KABUPATEN KARAWANG',
                    'KABUPATEN BEKASI', 'KABUPATEN BANDUNG BARAT', 'KABUPATEN PANGANDARAN',
                    'KOTA BOGOR', 'KOTA SUKABUMI', 'KOTA BANDUNG', 'KOTA CIREBON',
                    'KOTA BEKASI', 'KOTA DEPOK', 'KOTA CIMAHI', 'KOTA TASIKMALAYA',
                    'KOTA BANJAR'
                ]
                
                pendidikan = [
                    'SD KE BAWAH', 'SMP', 'SMA (UMUM)', 
                    'SMA (KEJURUAN)', 'DIPLOMA I/II/III/AKADEMI/UNIVERSITAS'
                ]
                
                data_list = []
                for kab in kabupaten:
                    for edu in pendidikan:
                        base = np.random.randint(500, 5000)
                        for tahun in range(2020, 2026):
                            trend = 1 + 0.1 * (tahun - 2020) + np.random.normal(0, 0.05)
                            if tahun in [2020, 2021]:
                                covid = 1.3
                            else:
                                covid = 1.0
                            nilai = int(base * trend * covid * np.random.uniform(0.7, 1.3))
                            data_list.append({
                                'nama_kabupaten_kota': kab,
                                'tahun': tahun,
                                'pendidikan': edu,
                                'jumlah_pengangguran': max(10, nilai)
                            })
                
                df_sample = pd.DataFrame(data_list)
                st.session_state.df = df_sample
                st.session_state.data_loaded = True
                
                df_proc, edu_order = preprocess_data(df_sample)
                st.session_state.pendidikan_order = edu_order
                
                st.success(f"✅ Dataset contoh dimuat! {len(df_sample):,} baris")
                st.rerun()
    
    if st.session_state.data_loaded:
        st.success("✅ Dataset telah dimuat!")
        st.dataframe(st.session_state.df.head(), use_container_width=True)
    else:
        st.info("ℹ️ Silakan upload dataset atau gunakan data contoh.")

# ==================== MENU: UPLOAD DATASET ====================

elif selected == "Upload Dataset":
    st.markdown('<div class="sub-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-area">
        <h4>📁 Upload file dataset (.xlsx)</h4>
        <p style="color: #7f8c8d;">Drag and drop file di sini</p>
        <p style="color: #95a5a6; font-size: 0.9rem;">Maksimal 200MB - Format XLSX</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file XLSX", type=['xlsx'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_excel(uploaded_file)
            
            required_cols = ['nama_kabupaten_kota', 'tahun', 'pendidikan', 'jumlah_pengangguran']
            missing = [col for col in required_cols if col not in df_raw.columns]
            
            if missing:
                st.error(f"❌ Kolom tidak ditemukan: {missing}")
                st.info("Kolom yang diperlukan: nama_kabupaten_kota, tahun, pendidikan, jumlah_pengangguran")
            else:
                st.session_state.df = df_raw
                st.session_state.data_loaded = True
                
                df_proc, edu_order = preprocess_data(df_raw)
                st.session_state.pendidikan_order = edu_order
                
                st.success(f"✅ File berhasil diupload! {len(df_proc):,} baris (2020-2025)")
                
                st.markdown("#### Preview Dataset")
                st.dataframe(df_proc.head(), use_container_width=True)
                
                st.markdown("#### Ringkasan Data")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Wilayah", df_proc['nama_kabupaten_kota'].nunique())
                with col2:
                    st.metric("Pendidikan", df_proc['pendidikan'].nunique())
                with col3:
                    st.metric("Tahun", df_proc['tahun'].nunique())
                with col4:
                    st.metric("Total Data", len(df_proc))
                    
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        if st.session_state.data_loaded:
            st.info(f"Dataset saat ini: {len(st.session_state.df):,} baris")
            st.dataframe(st.session_state.df.head(), use_container_width=True)
        else:
            st.info("Belum ada dataset. Silakan upload file Excel.")

# ==================== MENU: PREPROCESSING ====================

elif selected == "Preprocessing":
    st.markdown('<div class="sub-header">⚙️ Preprocessing Data</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        pendidikan_order = st.session_state.pendidikan_order
        
        st.markdown("""
        <div class="preprocessing-box">
            <h4>📌 Perbedaan Preprocessing K-Means vs Time Series K-Means</h4>
            <ul>
                <li><strong>K-Means:</strong> Pivot <code>(kab/kota, tahun) × pendidikan</code> → Normalisasi GLOBAL</li>
                <li><strong>Time Series K-Means:</strong> Pivot <code>(kab/kota, pendidikan) × tahun</code> → Normalisasi PER-DERET</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["📊 K-Means Preprocessing", "⏱️ Time Series Preprocessing"])
        
        with tab1:
            st.markdown("#### Preprocessing untuk K-Means")
            
            prep_km = get_kmeans_preprocessing(df, pendidikan_order)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Dimensi Data", prep_km['description'])
            with col2:
                st.metric("Shape", f"{prep_km['shape'][0]} × {prep_km['shape'][1]}")
            
            st.markdown("##### 📋 Pivot Table (Sebelum Normalisasi)")
            st.dataframe(prep_km['pivot'].head(10), use_container_width=True)
            
            st.markdown("##### 📊 Data Setelah Normalisasi Global")
            df_normalized = pd.DataFrame(
                prep_km['X'],
                index=prep_km['pivot'].index,
                columns=prep_km['pivot'].columns
            )
            st.dataframe(df_normalized.head(10), use_container_width=True)
            
            st.info("""
            **💡 Penjelasan:**
            - Data di-pivot menjadi 162 objek (27 kab/kota × 6 tahun) dengan 5 fitur pendidikan
            - Normalisasi GLOBAL: semua nilai (162×5 = 810 nilai) digabung lalu dinormalisasi MinMax
            - Tujuan: mempertahankan perbedaan skala antar jenjang pendidikan
            """)
        
        with tab2:
            st.markdown("#### Preprocessing untuk Time Series K-Means")
            
            if not TSL_AVAILABLE:
                st.error("⚠️ tslearn tidak tersedia.")
            else:
                prep_ts = get_ts_preprocessing(df, pendidikan_order, 'minmax')
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Dimensi Data", prep_ts['description'])
                with col2:
                    st.metric("Shape", f"{prep_ts['shape'][0]} × {prep_ts['shape'][1]}")
                
                st.markdown("##### 📋 Pivot Table (Sebelum Normalisasi)")
                st.dataframe(prep_ts['pivot'].head(10), use_container_width=True)
                
                st.markdown("##### 📊 Data Setelah Normalisasi Per-Deret")
                df_normalized = pd.DataFrame(
                    prep_ts['X_ts'].reshape(prep_ts['shape'][0], prep_ts['shape'][1]),
                    index=prep_ts['pivot'].index,
                    columns=prep_ts['pivot'].columns
                )
                st.dataframe(df_normalized.head(10), use_container_width=True)
                
                st.info("""
                **💡 Penjelasan:**
                - Data di-pivot menjadi 135 deret waktu (27 kab/kota × 5 pendidikan) dengan 6 tahun
                - Normalisasi PER-DERET: masing-masing deret dinormalisasi sendiri (MinMax)
                - Tujuan: mempertahankan BENTUK/POLA waktu, bukan nilai absolut
                """)

# ==================== MENU: EDA ====================

elif selected == "Exploratory Data Analysis":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wilayah", df['nama_kabupaten_kota'].nunique())
        with col2:
            st.metric("Pendidikan", df['pendidikan'].nunique())
        with col3:
            st.metric("Tahun", df['tahun'].nunique())
        with col4:
            st.metric("Total Data", len(df))
        
        st.markdown("#### Statistik Deskriptif")
        st.dataframe(df['jumlah_pengangguran'].describe(), use_container_width=True)
        
        if PLOTLY_AVAILABLE:
            st.markdown("#### Visualisasi")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(
                    df, x='tahun', y='jumlah_pengangguran',
                    color='pendidikan', title='Tren Pengangguran per Pendidikan'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df, x='tahun', y='jumlah_pengangguran',
                    title='Distribusi Pengangguran per Tahun'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            pivot = df.pivot_table(
                index='pendidikan', columns='tahun',
                values='jumlah_pengangguran', aggfunc='mean'
            )
            fig = px.imshow(
                pivot, title='Heatmap Rata-rata Pengangguran',
                color_continuous_scale='RdBu_r', aspect="auto"
            )
            fig.update_layout(height=450)
            st.plotly_chart(fig, use_container_width=True)

# ==================== MENU: K-MEANS ====================

elif selected == "K-Means Clustering":
    st.markdown('<div class="sub-header">🎯 K-Means Clustering</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        pendidikan_order = st.session_state.pendidikan_order
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Konfigurasi")
            k_method = st.radio(
                "Metode pemilihan k",
                ["Otomatis", "Manual"],
                index=0,
                key="km_method"
            )
            if k_method == "Manual":
                k_value = st.slider("Jumlah Cluster (k)", 2, 10, 3, key="km_k")
            else:
                k_value = None
        
        with col2:
            st.markdown("#### Status")
            if st.session_state.km_results is not None:
                st.success(f"✅ Selesai (k={st.session_state.km_results['k']})")
            else:
                st.info("ℹ️ Belum dijalankan")
        
        if st.button("🚀 Jalankan K-Means", use_container_width=True):
            with st.spinner("Menjalankan K-Means..."):
                try:
                    results = run_kmeans(df, pendidikan_order, k_value)
                    st.session_state.km_results = results
                    st.success(f"✅ Selesai! Cluster optimal: k={results['k']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.km_results is not None:
            results = st.session_state.km_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cluster Optimal", f"k={results['k']}")
            with col2:
                st.metric("Silhouette Score", f"{results['silhouette']:.4f}")
            with col3:
                st.metric("Davies-Bouldin", f"{results['dbi']:.4f}")
            
            st.markdown("##### Distribusi Cluster")
            counts = results['hasil']['cluster'].value_counts().sort_index()
            st.bar_chart(counts)
            
            st.markdown("##### Preview Hasil")
            st.dataframe(results['hasil'].head(10), use_container_width=True)

# ==================== MENU: TIME SERIES K-MEANS ====================

elif selected == "Time Series K-Means":
    st.markdown('<div class="sub-header">⏱️ Time Series K-Means (DTW)</div>', unsafe_allow_html=True)
    
    if not TSL_AVAILABLE:
        st.error("⚠️ tslearn tidak tersedia. Install: pip install tslearn")
    elif not st.session_state.data_loaded:
        st.warning("⚠️ Upload dataset terlebih dahulu.")
    else:
        df = st.session_state.df
        pendidikan_order = st.session_state.pendidikan_order
        
        col1, col2, col3 = st.columns([1.5, 1, 1])
        
        with col1:
            st.markdown("#### Konfigurasi")
            k_method = st.radio(
                "Metode pemilihan k",
                ["Otomatis", "Manual"],
                index=0,
                key="ts_method"
            )
            if k_method == "Manual":
                k_value = st.slider("Jumlah Cluster (k)", 2, 10, 3, key="ts_k")
            else:
                k_value = None
        
        with col2:
            st.markdown("#### Scaler")
            scaler_type = st.radio(
                "Tipe Normalisasi",
                ["MinMax", "Z-Score"],
                index=0,
                key="ts_scaler"
            )
        
        with col3:
            st.markdown("#### Status")
            if st.session_state.ts_results is not None:
                st.success(f"✅ Selesai (k={st.session_state.ts_results['k']})")
            else:
                st.info("ℹ️ Belum dijalankan")
        
        if st.button("🚀 Jalankan Time Series K-Means", use_container_width=True):
            with st.spinner("Menjalankan Time Series K-Means..."):
                try:
                    scaler = 'minmax' if scaler_type == "MinMax" else 'zscore'
                    results = run_timeseries_kmeans(df, pendidikan_order, k_value, scaler)
                    st.session_state.ts_results = results
                    st.success(f"✅ Selesai! Cluster optimal: k={results['k']}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if st.session_state.ts_results is not None:
            results = st.session_state.ts_results
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cluster Optimal", f"k={results['k']}")
            with col2:
                st.metric("Silhouette Score", f"{results['silhouette']:.4f}")
            with col3:
                st.metric("Davies-Bouldin", f"{results['dbi']:.4f}")
            
            st.markdown("##### Distribusi Cluster")
            counts = results['hasil']['cluster'].value_counts().sort_index()
            st.bar_chart(counts)
            
            st.markdown("##### Preview Hasil")
            st.dataframe(results['hasil'].head(10), use_container_width=True)

# ==================== MENU: KOMPARASI ====================

elif selected == "Komparasi":
    st.markdown('<div class="sub-header">🔍 Komparasi Algoritma</div>', unsafe_allow_html=True)
    
    km = st.session_state.km_results
    ts = st.session_state.ts_results
    
    if km is None and ts is None:
        st.warning("⚠️ Jalankan K-Means dan/atau Time Series K-Means terlebih dahulu.")
    else:
        data = []
        if km is not None:
            data.append({
                'Algoritma': 'K-Means',
                'k': km['k'],
                'Silhouette': f"{km['silhouette']:.4f}",
                'DBI': f"{km['dbi']:.4f}"
            })
        if ts is not None:
            data.append({
                'Algoritma': 'Time Series K-Means',
                'k': ts['k'],
                'Silhouette': f"{ts['silhouette']:.4f}",
                'DBI': f"{ts['dbi']:.4f}"
            })
        
        df_comp = pd.DataFrame(data)
        st.dataframe(df_comp, use_container_width=True)
        
        if PLOTLY_AVAILABLE and len(data) > 1:
            fig = go.Figure()
            for row in data:
                fig.add_trace(go.Bar(
                    name=row['Algoritma'],
                    x=['Silhouette', 'DBI'],
                    y=[float(row['Silhouette']), float(row['DBI'])]
                ))
            fig.update_layout(
                title='Perbandingan Metrik',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if km is not None and ts is not None:
            if km['silhouette'] > ts['silhouette']:
                better = "K-Means"
            else:
                better = "Time Series K-Means"
            
            st.info(f"""
            **📊 Kesimpulan:**
            
            - **K-Means:** k={km['k']}, Silhouette={km['silhouette']:.4f}
            - **Time Series K-Means:** k={ts['k']}, Silhouette={ts['silhouette']:.4f}
            - **Terbaik:** {better} memiliki Silhouette Score lebih tinggi
            """)

# ==================== MENU: VISUALISASI ====================

elif selected == "Visualisasi":
    st.markdown('<div class="sub-header">📈 Visualisasi</div>', unsafe_allow_html=True)
    
    km = st.session_state.km_results
    ts = st.session_state.ts_results
    
    if km is None and ts is None:
        st.warning("⚠️ Jalankan analisis terlebih dahulu.")
    elif not PLOTLY_AVAILABLE:
        st.warning("⚠️ Plotly tidak tersedia.")
    else:
        tabs = st.tabs(["PCA Scatter", "Cluster Means", "Heatmap", "Time Series"])
        
        with tabs[0]:
            if km is not None:
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(km['X'])
                df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                df_pca['Cluster'] = km['labels'].astype(str)
                
                fig = px.scatter(
                    df_pca, x='PC1', y='PC2', color='Cluster',
                    title=f'PCA - K-Means (k={km["k"]})'
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[1]:
            if km is not None:
                edu_order = st.session_state.pendidikan_order
                means = km['hasil'].groupby('cluster')[edu_order].mean()
                fig = px.bar(
                    means.reset_index().melt(id_vars='cluster'),
                    x='cluster', y='value', color='variable',
                    title='Rata-rata per Cluster - K-Means'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[2]:
            if km is not None:
                pivot = km['hasil'].pivot_table(
                    index='nama_kabupaten_kota', columns='tahun', values='cluster'
                )
                fig = px.imshow(
                    pivot, title=f'Heatmap Cluster - K-Means (k={km["k"]})',
                    color_continuous_scale='Viridis', aspect="auto"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with tabs[3]:
            if ts is not None:
                X_ts = ts['X_ts']
                labels = ts['labels']
                tahun = ts['tahun_order']
                
                fig = go.Figure()
                for c in sorted(np.unique(labels)):
                    idx = np.where(labels == c)[0]
                    mean_pattern = X_ts[idx].mean(axis=0).ravel()
                    fig.add_trace(go.Scatter(
                        x=tahun, y=mean_pattern,
                        mode='lines+markers',
                        name=f'Cluster {c} (n={len(idx)})'
                    ))
                
                fig.update_layout(
                    title='Pola Rata-rata per Cluster - Time Series K-Means',
                    xaxis_title='Tahun',
                    yaxis_title='Nilai (Normalized)',
                    height=450
                )
                st.plotly_chart(fig, use_container_width=True)

# ==================== MENU: UNDUH HASIL ====================

elif selected == "Unduh Hasil":
    st.markdown('<div class="sub-header">📥 Unduh Hasil</div>', unsafe_allow_html=True)
    
    km = st.session_state.km_results
    ts = st.session_state.ts_results
    
    if km is None and ts is None:
        st.info("Belum ada hasil. Jalankan analisis terlebih dahulu.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            if km is not None:
                csv = km['hasil'].to_csv(index=False)
                st.download_button(
                    "📄 K-Means (.csv)",
                    data=csv,
                    file_name="hasil_kmeans.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if ts is not None:
                csv = ts['hasil'].to_csv(index=False)
                st.download_button(
                    "📄 Time Series K-Means (.csv)",
                    data=csv,
                    file_name="hasil_timeseries.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        if km is not None and ts is not None:
            comp = pd.DataFrame({
                'Algoritma': ['K-Means', 'Time Series K-Means'],
                'k': [km['k'], ts['k']],
                'Silhouette': [km['silhouette'], ts['silhouette']],
                'DBI': [km['dbi'], ts['dbi']]
            })
            csv_comp = comp.to_csv(index=False)
            st.download_button(
                "📄 Komparasi (.csv)",
                data=csv_comp,
                file_name="hasil_komparasi.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== MENU: TENTANG ====================

elif selected == "Tentang":
    st.markdown('<div class="sub-header">ℹ️ Tentang Aplikasi</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px;">
        <h3>📊 Sistem Komparasi Clustering Pengangguran Terbuka</h3>
        <p><strong>Versi:</strong> 3.0.0</p>
        <p><strong>Tahun:</strong> 2026</p>
        <hr>
        <h4>🎯 Tujuan</h4>
        <p>Membantu analisis dan perbandingan clustering antara K-Means dan Time Series K-Means 
        untuk data pengangguran terbuka di Jawa Barat.</p>
        <hr>
        <h4>📚 Teknologi</h4>
        <ul>
            <li><strong>Framework:</strong> Streamlit</li>
            <li><strong>ML:</strong> Scikit-learn, Tslearn</li>
            <li><strong>Visualisasi:</strong> Plotly, Matplotlib</li>
            <li><strong>Data:</strong> Pandas, NumPy</li>
        </ul>
        <hr>
        <h4>📖 Metodologi</h4>
        <ul>
            <li><strong>K-Means:</strong> Normalisasi global, Euclidean distance</li>
            <li><strong>Time Series K-Means:</strong> Normalisasi per-deret, DTW distance</li>
            <li><strong>Evaluasi:</strong> Silhouette Score, Davies-Bouldin Index</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
    <div class="footer">
        © 2026 Sistem Komparasi Clustering Pengangguran Terbuka. All rights reserved.
    </div>
""", unsafe_allow_html=True)
