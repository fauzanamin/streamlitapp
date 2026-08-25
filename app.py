# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import io
import base64
import warnings
warnings.filterwarnings('ignore')

# Coba import dengan error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from streamlit_option_menu import option_menu
    MENU_AVAILABLE = True
except ImportError:
    MENU_AVAILABLE = False

# Coba import tslearn untuk TimeSeriesKMeans
try:
    from tslearn.clustering import TimeSeriesKMeans
    from tslearn.preprocessing import TimeSeriesScalerMinMax
    from tslearn.utils import to_time_series_dataset
    from tslearn.metrics import cdist_dtw
    from tslearn.clustering import TimeSeriesKMeans
    TSKLEARN_AVAILABLE = True
except ImportError:
    TSKLEARN_AVAILABLE = False

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Komparasi Clustering Pengangguran Terbuka",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling yang lebih baik
st.markdown("""
    <style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.3px;
        line-height: 1.35;
        color: #1a2733;
        text-align: center;
        padding: 0.9rem 1.5rem 1.1rem 1.5rem;
        border-bottom: 1px solid #e3e8ee;
        margin-bottom: 1.6rem;
    }
    .main-header-icon {
        color: #1f77b4;
        margin-right: 0.35rem;
    }
    .app-subtitle {
        font-size: 0.92rem;
        font-weight: 400;
        color: #7f8c95;
        text-align: center;
        margin-top: 0.35rem;
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] {
        min-width: 230px !important;
        max-width: 230px !important;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.5rem;
    }
    section[data-testid="stSidebar"] h3 {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        color: #8a94a3;
        padding-left: 0.3rem;
        margin-bottom: 0.4rem;
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
    .metric-good {
        color: #27ae60;
        font-weight: bold;
    }
    .metric-bad {
        color: #e74c3c;
        font-weight: bold;
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
    .hero-section {
        background: linear-gradient(135deg, #1f77b4 0%, #2c3e50 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-section h1 {
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .hero-section p {
        font-size: 1.1rem;
        opacity: 0.9;
        max-width: 700px;
        margin: 0 auto;
    }
    .feature-card {
        background-color: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        text-align: center;
        height: 100%;
        border-top: 4px solid #1f77b4;
        transition: transform 0.2s;
    }
    .feature-card:hover {
        transform: translateY(-4px);
    }
    .feature-icon {
        font-size: 2.8rem;
        margin-bottom: 0.8rem;
    }
    .feature-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #7f8c8d;
        line-height: 1.5;
    }
    .step-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .step-number {
        display: inline-block;
        background-color: #1f77b4;
        color: white;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        line-height: 32px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-ready {
        background-color: #d4edda;
        color: #155724;
    }
    .status-processing {
        background-color: #fff3cd;
        color: #856404;
    }
    .status-done {
        background-color: #cce5ff;
        color: #004085;
    }
    .algo-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .algo-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .algo-desc {
        font-size: 0.9rem;
        color: #7f8c8d;
        line-height: 1.5;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-header">'
    '<span class="main-header-icon">📊</span>Sistem Komparasi Clustering Pengangguran Terbuka'
    '<div class="app-subtitle">Perbandingan TimeSeriesKMeans vs Hierarchical Agglomerative Clustering dengan DTW</div>'
    '</div>',
    unsafe_allow_html=True
)

# Inisialisasi session state untuk menyimpan data
if 'data' not in st.session_state:
    st.session_state.data = None
if 'pivot_km' not in st.session_state:
    st.session_state.pivot_km = None
if 'X' not in st.session_state:
    st.session_state.X = None
if 'X_ts' not in st.session_state:
    st.session_state.X_ts = None
if 'tahun_order' not in st.session_state:
    st.session_state.tahun_order = None
if 'hasil_ts' not in st.session_state:
    st.session_state.hasil_ts = None
if 'hasil_hac' not in st.session_state:
    st.session_state.hasil_hac = None
if 'final_k_ts' not in st.session_state:
    st.session_state.final_k_ts = None
if 'final_k_hac' not in st.session_state:
    st.session_state.final_k_hac = None
if 'results_ts_df' not in st.session_state:
    st.session_state.results_ts_df = None
if 'results_hac_df' not in st.session_state:
    st.session_state.results_hac_df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'pendidikan_order' not in st.session_state:
    st.session_state.pendidikan_order = None

# Sidebar untuk menu
with st.sidebar:
    st.markdown("### MENU")
    
    if MENU_AVAILABLE:
        selected = option_menu(
            menu_title=None,
            options=["Beranda", "Upload Dataset", "Exploratory Data Analysis", "Clustering", 
                    "Komparasi Algoritma", "Visualisasi", "Unduh Hasil", "Tentang Aplikasi"],
            icons=["house", "cloud-upload", "bar-chart", "diagram-3", "shuffle", 
                   "graph-up", "download", "info-circle"],
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
        menu_options = ["Beranda", "Upload Dataset", "Exploratory Data Analysis", "Clustering", 
                       "Komparasi Algoritma", "Visualisasi", "Unduh Hasil", "Tentang Aplikasi"]
        selected = st.selectbox("Pilih Menu", menu_options)

# ==================== FUNGSI UTAMA ====================

def preprocess_data(df):
    """Preprocessing data dengan penggabungan pendidikan yang konsisten"""
    data = df.copy()
    
    # Filter tahun 2017-2025
    data = data[(data['tahun'] >= 2017) & (data['tahun'] <= 2025)]
    
    # Gabung kategori pendidikan dengan konsisten
    pendidikan_mapping = {
        'TIDAK/BELUM PERNAH SEKOLAH/TIDAK/BELUM TAMAT SD': 'SD KE BAWAH',
        'SD': 'SD KE BAWAH',
        'SMP': 'SMP',
        'SMP SEDERAJAT': 'SMP',
        'SMA (UMUM)': 'SMA',
        'SMA (KEJURUAN)': 'SMA',
        'SMA SEDERAJAT': 'SMA',
        'SMK': 'SMA',
        'DIPLOMA I/II/III/AKADEMI/UNIVERSITAS': 'DIPLOMA/UNIVERSITAS',
        'DIPLOMA': 'DIPLOMA/UNIVERSITAS',
        'UNIVERSITAS': 'DIPLOMA/UNIVERSITAS',
        'AKADEMI': 'DIPLOMA/UNIVERSITAS'
    }
    
    data['pendidikan'] = data['pendidikan'].replace(pendidikan_mapping)
    
    # Pastikan hanya 4 kategori yang tersisa
    pendidikan_order = ['SD KE BAWAH', 'SMP', 'SMA', 'DIPLOMA/UNIVERSITAS']
    data = data[data['pendidikan'].isin(pendidikan_order)]
    data['pendidikan'] = pd.Categorical(data['pendidikan'], categories=pendidikan_order, ordered=True)
    
    return data, pendidikan_order

def create_pivot_for_clustering(data, pendidikan_order):
    """Membuat pivot untuk clustering (27 kab/kota x 4 pendidikan x 9 tahun)"""
    pivot = data.pivot_table(
        index=['nama_kabupaten_kota', 'tahun'],
        columns='pendidikan',
        values='jumlah_pengangguran',
        aggfunc='mean',
        observed=True
    )
    pivot = pivot[pendidikan_order]
    
    # Interpolasi jika ada missing value
    if pivot.isnull().values.any():
        pivot = pivot.interpolate(axis=0, limit_direction='both')
    
    return pivot

def create_timeseries_data(data, pendidikan_order):
    """Membuat data time series untuk clustering (27 kab x 4 pendidikan x 9 tahun)"""
    ts_data = data.copy()
    tahun_order = sorted(ts_data['tahun'].unique())
    
    # Pivot untuk time series
    pivot_ts = ts_data.pivot_table(
        index=['nama_kabupaten_kota', 'pendidikan'],
        columns='tahun',
        values='jumlah_pengangguran',
        aggfunc='mean'
    )
    pivot_ts = pivot_ts[tahun_order]
    
    if pivot_ts.isnull().values.any():
        pivot_ts = pivot_ts.interpolate(axis=1, limit_direction='both')
    
    # Bentuk array time series 3D (n_samples x n_timesteps x n_features)
    # n_samples = 27 kabupaten/kota * 4 pendidikan = 108
    # n_timesteps = 9 tahun (2017-2025)
    # n_features = 1 (jumlah pengangguran)
    
    X_ts_raw = to_time_series_dataset(pivot_ts.values)
    
    # Normalisasi per-deret
    scaler_ts = TimeSeriesScalerMinMax()
    X_ts = scaler_ts.fit_transform(X_ts_raw)
    
    labels_id = pivot_ts.reset_index()[['nama_kabupaten_kota', 'pendidikan']]
    
    return X_ts, labels_id, tahun_order, pivot_ts

def evaluate_timeseries_clustering(X_ts, K_range=range(2, 8)):
    """Evaluasi TimeSeriesKMeans clustering dengan DTW"""
    inertia_ts = []
    silhouette_ts = []
    dbi_ts = []
    
    for k in K_range:
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
        
        # Silhouette dengan DTW distance matrix
        dist_matrix = cdist_dtw(X_ts)
        sil = silhouette_score(dist_matrix, labels_k, metric="precomputed")
        silhouette_ts.append(sil)
        
        # Davies-Bouldin dengan flattened data
        X_flat = X_ts.reshape(X_ts.shape[0], -1)
        dbi = davies_bouldin_score(X_flat, labels_k)
        dbi_ts.append(dbi)
    
    results_ts_df = pd.DataFrame({
        'Jumlah Cluster': list(K_range),
        'Inertia': inertia_ts,
        'Silhouette Score (DTW)': silhouette_ts,
        'Davies-Bouldin Index': dbi_ts
    })
    
    # Penentuan cluster terbaik
    best_sil_k_ts = results_ts_df.loc[results_ts_df['Silhouette Score (DTW)'].idxmax(), 'Jumlah Cluster']
    best_dbi_k_ts = results_ts_df.loc[results_ts_df['Davies-Bouldin Index'].idxmin(), 'Jumlah Cluster']
    
    inertia_diff_ts = np.diff(inertia_ts)
    inertia_ratio_ts = inertia_diff_ts[1:] / inertia_diff_ts[:-1]
    best_elbow_k_ts = list(K_range)[np.argmin(inertia_ratio_ts) + 1]
    
    vote_counts_ts = pd.Series([best_elbow_k_ts, best_sil_k_ts, best_dbi_k_ts]).value_counts()
    final_k_ts = vote_counts_ts.index[0]
    
    # Clustering final
    model_final_ts = TimeSeriesKMeans(
        n_clusters=final_k_ts,
        metric="dtw",
        random_state=42,
        n_init=5,
        max_iter=50,
        n_jobs=-1
    )
    final_labels_ts = model_final_ts.fit_predict(X_ts)
    
    return results_ts_df, final_k_ts, final_labels_ts

def evaluate_hac_clustering(X_ts, K_range=range(2, 8)):
    """Evaluasi Hierarchical Agglomerative Clustering dengan DTW"""
    # Hitung DTW distance matrix
    dist_matrix = cdist_dtw(X_ts)
    
    silhouette_hac = []
    dbi_hac = []
    
    for k in K_range:
        hac = AgglomerativeClustering(
            n_clusters=k,
            metric='precomputed',
            linkage='average'
        )
        labels_k = hac.fit_predict(dist_matrix)
        
        sil = silhouette_score(dist_matrix, labels_k, metric="precomputed")
        silhouette_hac.append(sil)
        
        X_flat = X_ts.reshape(X_ts.shape[0], -1)
        dbi = davies_bouldin_score(X_flat, labels_k)
        dbi_hac.append(dbi)
    
    results_hac_df = pd.DataFrame({
        'Jumlah Cluster': list(K_range),
        'Silhouette Score (DTW)': silhouette_hac,
        'Davies-Bouldin Index': dbi_hac
    })
    
    # Penentuan cluster terbaik
    best_sil_k_hac = results_hac_df.loc[results_hac_df['Silhouette Score (DTW)'].idxmax(), 'Jumlah Cluster']
    best_dbi_k_hac = results_hac_df.loc[results_hac_df['Davies-Bouldin Index'].idxmin(), 'Jumlah Cluster']
    
    vote_counts_hac = pd.Series([best_sil_k_hac, best_dbi_k_hac]).value_counts()
    final_k_hac = vote_counts_hac.index[0]
    
    # Clustering final
    hac_final = AgglomerativeClustering(
        n_clusters=final_k_hac,
        metric='precomputed',
        linkage='average'
    )
    final_labels_hac = hac_final.fit_predict(dist_matrix)
    
    return results_hac_df, final_k_hac, final_labels_hac

def run_comparative_clustering(data, pendidikan_order):
    """Menjalankan kedua algoritma dan membandingkan hasilnya"""
    if not TSKLEARN_AVAILABLE:
        return None, None, None, None, None, None, None
    
    # Siapkan data time series
    X_ts, labels_id, tahun_order, pivot_ts = create_timeseries_data(data, pendidikan_order)
    
    K_range = range(2, 8)  # Range cluster yang dievaluasi
    
    # TimeSeriesKMeans
    results_ts_df, final_k_ts, final_labels_ts = evaluate_timeseries_clustering(X_ts, K_range)
    
    # Hierarchical Agglomerative Clustering
    results_hac_df, final_k_hac, final_labels_hac = evaluate_hac_clustering(X_ts, K_range)
    
    # Tambahkan hasil clustering ke labels_id
    hasil_ts = labels_id.copy()
    hasil_ts['cluster_ts'] = final_labels_ts
    hasil_ts['cluster_hac'] = final_labels_hac
    
    return hasil_ts, X_ts, results_ts_df, results_hac_df, final_k_ts, final_k_hac, tahun_order

# ==================== BERANDA (LANDING PAGE) ====================
if selected == "Beranda":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <h1>📊 Selamat Datang di Sistem Komparasi Clustering</h1>
        <p>Sistem ini membandingkan dua algoritma clustering berbasis DTW untuk data pengangguran terbuka di Jawa Barat:</p>
        <p style="margin-top: 0.5rem;">
            <span style="background: #3498db; padding: 0.3rem 1rem; border-radius: 20px; margin: 0 0.3rem;">TimeSeriesKMeans</span>
            <span style="background: #e74c3c; padding: 0.3rem 1rem; border-radius: 20px; margin: 0 0.3rem;">Hierarchical Agglomerative</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Ringkasan - DIPISAHKAN DENGAN JELAS
    col_status1, col_status2, col_status3, col_status4 = st.columns(4)
    
    with col_status1:
        if st.session_state.data is not None:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #27ae60; text-align: center;">
                <div style="font-size: 1.8rem;">✅</div>
                <div class="stat-label" style="font-size: 0.85rem;">Dataset Terupload</div>
                <div style="font-size: 0.75rem; color: #27ae60;">{len(st.session_state.data)} baris</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stat-card" style="border-left-color: #95a5a6; text-align: center;">
                <div style="font-size: 1.8rem;">⏳</div>
                <div class="stat-label" style="font-size: 0.85rem;">Belum Upload</div>
                <div style="font-size: 0.75rem; color: #95a5a6;">Upload dataset</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_status2:
        if st.session_state.X_ts is not None:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #27ae60; text-align: center;">
                <div style="font-size: 1.8rem;">✅</div>
                <div class="stat-label" style="font-size: 0.85rem;">Preprocessing Selesai</div>
                <div style="font-size: 0.75rem; color: #27ae60;">{st.session_state.X_ts.shape[0]} time series</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stat-card" style="border-left-color: #95a5a6; text-align: center;">
                <div style="font-size: 1.8rem;">⏳</div>
                <div class="stat-label" style="font-size: 0.85rem;">Preprocessing</div>
                <div style="font-size: 0.75rem; color: #95a5a6;">Menunggu upload</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_status3:
        if st.session_state.hasil_ts is not None:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #3498db; text-align: center;">
                <div style="font-size: 1.8rem;">✅</div>
                <div class="stat-label" style="font-size: 0.85rem;">TimeSeriesKMeans</div>
                <div style="font-size: 0.75rem; color: #3498db;">k={st.session_state.final_k_ts} cluster</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stat-card" style="border-left-color: #95a5a6; text-align: center;">
                <div style="font-size: 1.8rem;">⏳</div>
                <div class="stat-label" style="font-size: 0.85rem;">TimeSeriesKMeans</div>
                <div style="font-size: 0.75rem; color: #95a5a6;">Belum dijalankan</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col_status4:
        if st.session_state.hasil_hac is not None:
            st.markdown(f"""
            <div class="stat-card" style="border-left-color: #e74c3c; text-align: center;">
                <div style="font-size: 1.8rem;">✅</div>
                <div class="stat-label" style="font-size: 0.85rem;">Hierarchical Agglomerative</div>
                <div style="font-size: 0.75rem; color: #e74c3c;">k={st.session_state.final_k_hac} cluster</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="stat-card" style="border-left-color: #95a5a6; text-align: center;">
                <div style="font-size: 1.8rem;">⏳</div>
                <div class="stat-label" style="font-size: 0.85rem;">Hierarchical Agglomerative</div>
                <div style="font-size: 0.75rem; color: #95a5a6;">Belum dijalankan</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Penjelasan Algoritma - DIPISAHKAN
    st.markdown('<h3 style="text-align: center; margin-bottom: 1.5rem;">🤖 Algoritma Clustering yang Dibandingkan</h3>', unsafe_allow_html=True)
    
    col_algo1, col_algo2 = st.columns(2)
    
    with col_algo1:
        st.markdown("""
        <div class="algo-card" style="border-left-color: #3498db;">
            <div class="algo-title" style="color: #3498db;">📌 TimeSeriesKMeans</div>
            <div class="algo-desc">
                <p><strong>Metode:</strong> K-Means berbasis Dynamic Time Warping (DTW)</p>
                <p><strong>Karakteristik:</strong></p>
                <ul>
                    <li>Prototype-based clustering</li>
                    <li>Efisien untuk data berukuran besar</li>
                    <li>Menghasilkan centroid untuk interpretasi</li>
                    <li>Menggunakan DTW sebagai distance metric</li>
                </ul>
                <p><strong>Keunggulan:</strong> Cepat, scalable, mudah diinterpretasi</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_algo2:
        st.markdown("""
        <div class="algo-card" style="border-left-color: #e74c3c;">
            <div class="algo-title" style="color: #e74c3c;">📌 Hierarchical Agglomerative Clustering</div>
            <div class="algo-desc">
                <p><strong>Metode:</strong> Agglomerative Clustering dengan DTW distance matrix</p>
                <p><strong>Karakteristik:</strong></p>
                <ul>
                    <li>Hierarchical clustering</li>
                    <li>Tidak perlu menentukan k awal</li>
                    <li>Dapat divisualisasikan dengan dendrogram</li>
                    <li>Menggunakan DTW distance matrix</li>
                </ul>
                <p><strong>Keunggulan:</strong> Menangkap struktur hierarki, flexible</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Fitur Utama
    st.markdown('<h3 style="text-align: center; margin-bottom: 1.5rem;">🚀 Fitur Utama Aplikasi</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📤</div>
            <div class="feature-title">Upload Dataset</div>
            <div class="feature-desc">Upload file Excel (.xlsx) data pengangguran terbuka di Jawa Barat (2017-2025)</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Exploratory Data Analysis</div>
            <div class="feature-desc">Analisis eksploratif data dengan visualisasi tren dan distribusi</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Clustering dengan DTW</div>
            <div class="feature-desc">Jalankan TimeSeriesKMeans dan HAC dengan metric DTW secara bersamaan</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Komparasi & Visualisasi</div>
            <div class="feature-desc">Bandingkan hasil kedua algoritma dengan visualisasi interaktif</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Informasi Dataset
    st.markdown('<h3 style="text-align: center; margin-bottom: 1rem;">📋 Informasi Dataset</h3>', unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns([1, 1])
    
    with col_info1:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
            <h4>📌 Format Dataset</h4>
            <ul>
                <li><strong>nama_kabupaten_kota</strong> - Nama kabupaten/kota di Jawa Barat</li>
                <li><strong>tahun</strong> - Tahun (2017 - 2025)</li>
                <li><strong>pendidikan</strong> - Tingkat pendidikan (4 kategori)</li>
                <li><strong>jumlah_pengangguran</strong> - Jumlah pengangguran</li>
            </ul>
            <br>
            <h4>📊 Struktur Data</h4>
            <ul>
                <li><strong>27</strong> Kabupaten/Kota</li>
                <li><strong>9</strong> Tahun (2017-2025)</li>
                <li><strong>4</strong> Kategori Pendidikan</li>
                <li><strong>~972</strong> Total Observasi</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
            <h4>🎯 Perbandingan Algoritma</h4>
            <ul>
                <li><strong>TimeSeriesKMeans</strong> - K-Means berbasis DTW (Dynamic Time Warping)</li>
                <li><strong>Hierarchical Agglomerative</strong> - HAC dengan DTW distance matrix</li>
            </ul>
            <br>
            <h4>📈 Metrik Evaluasi</h4>
            <ul>
                <li><strong>Elbow Method</strong> - Inertia untuk optimal cluster</li>
                <li><strong>Silhouette Score</strong> - Mengukur kualitas clustering</li>
                <li><strong>Davies-Bouldin Index</strong> - Mengukur separasi antar cluster</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start
    if st.session_state.data is None:
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem; background-color: #e8f4f8; border-radius: 10px;">
            <h4>🚀 Mulai Analisis Sekarang</h4>
            <p style="color: #5a6a7a;">Upload dataset Anda melalui menu <strong>Upload Dataset</strong> untuk memulai analisis clustering</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== UPLOAD DATASET ====================
elif selected == "Upload Dataset":
    st.markdown('<div class="sub-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="upload-area">
            <h4>📁 Upload file dataset (.xlsx)</h4>
            <p style="color: #7f8c8d;">Drag and drop file here</p>
            <p style="color: #95a5a6; font-size: 0.9rem;">Limit 200MB per file - XLSX</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file XLSX", type=['xlsx'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ File berhasil diupload! {len(df)} baris data")
            
            # Preprocessing
            with st.spinner("Sedang memproses data..."):
                data, pendidikan_order = preprocess_data(df)
                
                # Simpan ke session state
                st.session_state.data = data
                st.session_state.pendidikan_order = pendidikan_order
            
            st.markdown("#### Preview Dataset Original")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("#### Preview Data setelah Preprocessing")
            st.dataframe(data.head(10), use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Jumlah Wilayah", data['nama_kabupaten_kota'].nunique())
            with col2:
                st.metric("Jumlah Pendidikan", data['pendidikan'].nunique())
            with col3:
                st.metric("Jumlah Tahun", len(data['tahun'].unique()))
            with col4:
                st.metric("Total Observasi", len(data))
            
            # Tampilkan distribusi pendidikan per tahun
            st.markdown("#### Distribusi Pendidikan per Tahun")
            edu_year = data.groupby(['tahun', 'pendidikan']).size().unstack(fill_value=0)
            st.dataframe(edu_year, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error membaca file: {e}")
    else:
        st.info("📁 Silakan upload file dataset untuk memulai")
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                <h5>📋 Format Dataset yang Diharapkan</h5>
                <ul>
                    <li><strong>nama_kabupaten_kota</strong>: Nama kabupaten/kota</li>
                    <li><strong>tahun</strong>: Tahun (2017-2025)</li>
                    <li><strong>pendidikan</strong>: Tingkat pendidikan</li>
                    <li><strong>jumlah_pengangguran</strong>: Jumlah pengangguran</li>
                </ul>
                <p style="color: #7f8c8d; font-size: 0.9rem;">⚠️ <strong>Catatan:</strong> Data pendidikan akan digabung menjadi 4 kategori: SD KE BAWAH, SMP, SMA, DIPLOMA/UNIVERSITAS</p>
            </div>
        """, unsafe_allow_html=True)

# ==================== EXPLORATORY DATA ANALYSIS ====================
elif selected == "Exploratory Data Analysis":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Statistik ringkasan
        st.markdown("#### Ringkasan Data")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jumlah Wilayah", data['nama_kabupaten_kota'].nunique())
        with col2:
            st.metric("Jumlah Pendidikan", data['pendidikan'].nunique())
        with col3:
            st.metric("Jumlah Tahun", len(data['tahun'].unique()))
        with col4:
            st.metric("Total Observasi", len(data))
        
        # Statistik deskriptif lengkap
        st.markdown("#### Statistik Deskriptif Jumlah Pengangguran")
        st.dataframe(data['jumlah_pengangguran'].describe(), use_container_width=True)
        
        st.markdown("#### Distribusi Jumlah Pengangguran per Tingkat Pendidikan")
        edu_stats = data.groupby('pendidikan')['jumlah_pengangguran'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)
        st.dataframe(edu_stats, use_container_width=True)
        
        # Cek missing value
        st.markdown("#### Cek Missing Value")
        missing_data = data.isnull().sum()
        st.dataframe(pd.DataFrame({'Missing Value': missing_data}), use_container_width=True)
        
        if PLOTLY_AVAILABLE:
            st.markdown("#### Visualisasi Data")
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Tren Pengangguran", 
                "Distribusi per Tahun", 
                "Heatmap Correlasi",
                "Distribusi per Pendidikan",
                "Statistik Lanjutan"
            ])
            
            with tab1:
                # Tren pengangguran per pendidikan
                fig = px.line(
                    data, x='tahun', y='jumlah_pengangguran', 
                    color='pendidikan', 
                    title='Tren Pengangguran Terbuka per Pendidikan (2017-2025)',
                    markers=True
                )
                fig.update_layout(height=450, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Insight:** Grafik ini menunjukkan tren pengangguran per tingkat pendidikan dari tahun 2017-2025")
            
            with tab2:
                # Boxplot distribusi per tahun
                fig = px.box(
                    data, x='tahun', y='jumlah_pengangguran',
                    title='Distribusi Jumlah Pengangguran per Tahun'
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Insight:** Boxplot menunjukkan variasi dan outlier data pengangguran per tahun")
            
            with tab3:
                # Heatmap correlasi antar tahun
                st.markdown("##### Heatmap Correlasi Antar Tahun")
                pivot_corr = data.pivot_table(
                    index=['nama_kabupaten_kota', 'pendidikan'],
                    columns='tahun',
                    values='jumlah_pengangguran',
                    aggfunc='mean'
                )
                corr_matrix = pivot_corr.corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title='Heatmap Correlasi Antar Tahun',
                    color_continuous_scale='RdBu_r',
                    aspect='auto',
                    text_auto=True
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)
                st.info("💡 **Insight:** Heatmap menunjukkan korelasi antar tahun. Warna merah = korelasi positif tinggi")
            
            with tab4:
                # Distribusi per pendidikan
                st.markdown("##### Distribusi Jumlah Pengangguran per Pendidikan")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Bar chart
                    edu_sum = data.groupby('pendidikan')['jumlah_pengangguran'].sum().reset_index()
                    fig = px.bar(
                        edu_sum, x='pendidikan', y='jumlah_pengangguran',
                        title='Total Pengangguran per Tingkat Pendidikan',
                        color='jumlah_pengangguran',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Boxplot per pendidikan
                    fig = px.box(
                        data, x='pendidikan', y='jumlah_pengangguran',
                        title='Distribusi Pengangguran per Tingkat Pendidikan'
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                # Statistik lanjutan
                st.markdown("##### Statistik per Kelompok Pendidikan")
                
                stats_per_edu = data.groupby('pendidikan')['jumlah_pengangguran'].agg([
                    ('Mean', 'mean'),
                    ('Median', 'median'),
                    ('Std Dev', 'std'),
                    ('Min', 'min'),
                    ('Max', 'max')
                ]).round(2)
                st.dataframe(stats_per_edu, use_container_width=True)
                
                st.markdown("##### Statistik per Tahun")
                stats_per_year = data.groupby('tahun')['jumlah_pengangguran'].agg([
                    ('Mean', 'mean'),
                    ('Median', 'median'),
                    ('Std Dev', 'std'),
                    ('Min', 'min'),
                    ('Max', 'max')
                ]).round(2)
                st.dataframe(stats_per_year, use_container_width=True)
                
                st.markdown("##### Distribusi Pendidikan per Tahun")
                edu_year_dist = data.groupby(['tahun', 'pendidikan']).size().unstack(fill_value=0)
                st.dataframe(edu_year_dist, use_container_width=True)
        else:
            st.warning("⚠️ Plotly tidak tersedia. Install dengan: pip install plotly")
    else:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")

# ==================== CLUSTERING ====================
elif selected == "Clustering":
    st.markdown('<div class="sub-header">🎯 Clustering dengan DTW</div>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        if not TSKLEARN_AVAILABLE:
            st.error("⚠️ tslearn tidak tersedia. Install dengan: pip install tslearn")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Pilih Mode Analisis")
                algorithm = st.radio(
                    "Pilih Mode",
                    ["TimeSeriesKMeans Only", "Hierarchical Agglomerative Only", "Komparasi Kedua Algoritma"],
                    index=0,
                    horizontal=True
                )
            
            with col2:
                st.markdown("#### Jumlah Cluster (k)")
                cluster_method = st.radio(
                    "Metode pemilihan k",
                    ["Otomatis (Rekomendasi)", "Manual"],
                    index=0
                )
                if cluster_method == "Manual":
                    k_value = st.slider("Jumlah Cluster", 2, 10, 3)
            
            if st.button("🚀 Jalankan Analisis", use_container_width=True):
                with st.spinner("Sedang menganalisis data..."):
                    # Jalankan kedua algoritma
                    hasil_ts, X_ts, results_ts_df, results_hac_df, final_k_ts, final_k_hac, tahun_order = run_comparative_clustering(
                        st.session_state.data, st.session_state.pendidikan_order
                    )
                    
                    st.session_state.X_ts = X_ts
                    st.session_state.tahun_order = tahun_order
                    
                    # Simpan hasil sesuai mode
                    if algorithm in ["TimeSeriesKMeans Only", "Komparasi Kedua Algoritma"]:
                        st.session_state.hasil_ts = hasil_ts[['nama_kabupaten_kota', 'pendidikan', 'cluster_ts']].copy()
                        st.session_state.hasil_ts.rename(columns={'cluster_ts': 'cluster'}, inplace=True)
                        st.session_state.results_ts_df = results_ts_df
                        st.session_state.final_k_ts = final_k_ts
                    
                    if algorithm in ["Hierarchical Agglomerative Only", "Komparasi Kedua Algoritma"]:
                        st.session_state.hasil_hac = hasil_ts[['nama_kabupaten_kota', 'pendidikan', 'cluster_hac']].copy()
                        st.session_state.hasil_hac.rename(columns={'cluster_hac': 'cluster'}, inplace=True)
                        st.session_state.results_hac_df = results_hac_df
                        st.session_state.final_k_hac = final_k_hac
                
                st.success("✅ Analisis clustering selesai!")
            
            # Tampilkan hasil jika sudah ada - DIPISAHKAN
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                if st.session_state.hasil_ts is not None:
                    st.markdown("#### 🔵 TimeSeriesKMeans")
                    st.markdown(f"**Cluster Terbaik (Voting):** {st.session_state.final_k_ts}")
                    st.dataframe(st.session_state.results_ts_df, use_container_width=True)
                    
                    st.markdown("**Distribusi Cluster**")
                    cluster_counts_ts = st.session_state.hasil_ts['cluster'].value_counts().sort_index()
                    st.dataframe(cluster_counts_ts, use_container_width=True)
                    
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            x=cluster_counts_ts.index, y=cluster_counts_ts.values,
                            title='Distribusi Cluster TimeSeriesKMeans',
                            labels={'x': 'Cluster', 'y': 'Jumlah'},
                            color_discrete_sequence=['#3498db']
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            with col_res2:
                if st.session_state.hasil_hac is not None:
                    st.markdown("#### 🔴 Hierarchical Agglomerative")
                    st.markdown(f"**Cluster Terbaik (Voting):** {st.session_state.final_k_hac}")
                    st.dataframe(st.session_state.results_hac_df, use_container_width=True)
                    
                    st.markdown("**Distribusi Cluster**")
                    cluster_counts_hac = st.session_state.hasil_hac['cluster'].value_counts().sort_index()
                    st.dataframe(cluster_counts_hac, use_container_width=True)
                    
                    if PLOTLY_AVAILABLE:
                        fig = px.bar(
                            x=cluster_counts_hac.index, y=cluster_counts_hac.values,
                            title='Distribusi Cluster Hierarchical Agglomerative',
                            labels={'x': 'Cluster', 'y': 'Jumlah'},
                            color_discrete_sequence=['#e74c3c']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")

# ==================== KOMPARASI ALGORITMA ====================
elif selected == "Komparasi Algoritma":
    st.markdown('<div class="sub-header">🔍 Komparasi Algoritma</div>', unsafe_allow_html=True)
    
    if st.session_state.hasil_ts is not None and st.session_state.hasil_hac is not None:
        st.markdown("#### Perbandingan Metrik")
        
        col1, col2 = st.columns(2)
        
        with col1:
            ts_silhouette = st.session_state.results_ts_df.loc[
                st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts, 
                'Silhouette Score (DTW)'
            ].values[0]
            ts_dbi = st.session_state.results_ts_df.loc[
                st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts, 
                'Davies-Bouldin Index'
            ].values[0]
            
            st.markdown(f"""
                <div class="result-card" style="border-left-color: #3498db;">
                    <h4 style="color: #3498db;">🔵 TimeSeriesKMeans</h4>
                    <p><strong>Cluster Terbaik:</strong> {st.session_state.final_k_ts}</p>
                    <p><strong>Silhouette Score:</strong> <span class="metric-good">{ts_silhouette:.4f}</span></p>
                    <p><strong>Davies-Bouldin Index:</strong> <span class="metric-bad">{ts_dbi:.4f}</span></p>
                    <p><strong>Keunggulan:</strong></p>
                    <ul>
                        <li>Menggunakan prototype-based clustering</li>
                        <li>Efisien untuk data berukuran besar</li>
                        <li>Menghasilkan centroid untuk interpretasi</li>
                        <li>Menggunakan DTW (Dynamic Time Warping)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            hac_silhouette = st.session_state.results_hac_df.loc[
                st.session_state.results_hac_df['Jumlah Cluster'] == st.session_state.final_k_hac, 
                'Silhouette Score (DTW)'
            ].values[0]
            hac_dbi = st.session_state.results_hac_df.loc[
                st.session_state.results_hac_df['Jumlah Cluster'] == st.session_state.final_k_hac, 
                'Davies-Bouldin Index'
            ].values[0]
            
            st.markdown(f"""
                <div class="result-card" style="border-left-color: #e74c3c;">
                    <h4 style="color: #e74c3c;">🔴 Hierarchical Agglomerative</h4>
                    <p><strong>Cluster Terbaik:</strong> {st.session_state.final_k_hac}</p>
                    <p><strong>Silhouette Score:</strong> <span class="metric-good">{hac_silhouette:.4f}</span></p>
                    <p><strong>Davies-Bouldin Index:</strong> <span class="metric-good">{hac_dbi:.4f}</span></p>
                    <p><strong>Keunggulan:</strong></p>
                    <ul>
                        <li>Tidak perlu menentukan jumlah cluster awal</li>
                        <li>Dapat divisualisasikan dengan dendrogram</li>
                        <li>Menangkap struktur hierarki data</li>
                        <li>Menggunakan DTW distance matrix</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            st.markdown("#### Visualisasi Perbandingan Metrik")
            
            metrics = ['Silhouette Score', 'Davies-Bouldin Index']
            tskm_values = [ts_silhouette, ts_dbi]
            hac_values = [hac_silhouette, hac_dbi]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='TimeSeriesKMeans',
                x=metrics,
                y=tskm_values,
                marker_color='#3498db'
            ))
            fig.add_trace(go.Bar(
                name='Hierarchical Agglomerative',
                x=metrics,
                y=hac_values,
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                title='Perbandingan Metrik Clustering',
                barmode='group',
                height=400,
                yaxis_title='Nilai'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Insight
            better_ts = ts_silhouette > hac_silhouette and ts_dbi < hac_dbi
            better_hac = hac_silhouette > ts_silhouette and hac_dbi < ts_dbi
            
            if better_ts:
                st.info(f"""
                💡 **Analisis Komparasi:**
                - **Silhouette Score:** TimeSeriesKMeans ({ts_silhouette:.4f}) lebih baik dari HAC ({hac_silhouette:.4f})
                - **Davies-Bouldin Index:** TimeSeriesKMeans ({ts_dbi:.4f}) lebih baik dari HAC ({hac_dbi:.4f})
                - **Kesimpulan:** TimeSeriesKMeans memberikan hasil clustering yang lebih baik untuk dataset ini
                """)
            elif better_hac:
                st.info(f"""
                💡 **Analisis Komparasi:**
                - **Silhouette Score:** HAC ({hac_silhouette:.4f}) lebih baik dari TimeSeriesKMeans ({ts_silhouette:.4f})
                - **Davies-Bouldin Index:** HAC ({hac_dbi:.4f}) lebih baik dari TimeSeriesKMeans ({ts_dbi:.4f})
                - **Kesimpulan:** Hierarchical Agglomerative Clustering memberikan hasil yang lebih baik untuk dataset ini
                """)
            else:
                st.info(f"""
                💡 **Analisis Komparasi:**
                - **Silhouette Score:** TimeSeriesKMeans ({ts_silhouette:.4f}) vs HAC ({hac_silhouette:.4f})
                - **Davies-Bouldin Index:** TimeSeriesKMeans ({ts_dbi:.4f}) vs HAC ({hac_dbi:.4f})
                - **Kesimpulan:** Kedua algoritma memiliki performa yang kompetitif, pilih berdasarkan kebutuhan analisis
                """)
    else:
        st.warning("⚠️ Silakan jalankan analisis clustering terlebih dahulu di menu 'Clustering'")

# ==================== VISUALISASI ====================
elif selected == "Visualisasi":
    st.markdown('<div class="sub-header">📈 Visualisasi</div>', unsafe_allow_html=True)
    
    if st.session_state.X_ts is not None:
        if PLOTLY_AVAILABLE:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Elbow & Metrik", 
                "PCA Visualization", 
                "Pola Time Series",
                "Heatmap Clustering", 
                "Perbandingan"
            ])
            
            with tab1:
                st.markdown("#### Elbow Method dan Metrik Evaluasi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.results_ts_df is not None:
                        # Elbow Curve TimeSeriesKMeans
                        fig = px.line(
                            st.session_state.results_ts_df, x='Jumlah Cluster', y='Inertia',
                            markers=True,
                            title='Elbow Curve - TimeSeriesKMeans'
                        )
                        fig.add_vline(
                            x=st.session_state.final_k_ts, line_dash="dash", line_color="blue",
                            annotation_text=f"Optimal k={st.session_state.final_k_ts}"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("##### 🔵 TimeSeriesKMeans")
                    if st.session_state.results_ts_df is not None:
                        col1a, col1b, col1c = st.columns(3)
                        with col1a:
                            st.metric(
                                "Silhouette Score", 
                                f"{st.session_state.results_ts_df.loc[st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts, 'Silhouette Score (DTW)'].values[0]:.4f}"
                            )
                        with col1b:
                            st.metric(
                                "Davies-Bouldin Index", 
                                f"{st.session_state.results_ts_df.loc[st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts, 'Davies-Bouldin Index'].values[0]:.4f}"
                            )
                        with col1c:
                            st.metric("Cluster Optimal", st.session_state.final_k_ts)
                
                with col2:
                    if st.session_state.results_hac_df is not None:
                        st.markdown("##### 🔴 Hierarchical Agglomerative")
                        col2a, col2b, col2c = st.columns(3)
                        with col2a:
                            st.metric(
                                "Silhouette Score", 
                                f"{st.session_state.results_hac_df.loc[st.session_state.results_hac_df['Jumlah Cluster'] == st.session_state.final_k_hac, 'Silhouette Score (DTW)'].values[0]:.4f}"
                            )
                        with col2b:
                            st.metric(
                                "Davies-Bouldin Index", 
                                f"{st.session_state.results_hac_df.loc[st.session_state.results_hac_df['Jumlah Cluster'] == st.session_state.final_k_hac, 'Davies-Bouldin Index'].values[0]:.4f}"
                            )
                        with col2c:
                            st.metric("Cluster Optimal", st.session_state.final_k_hac)
            
            with tab2:
                st.markdown("#### PCA Visualization")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.hasil_ts is not None:
                        # PCA untuk TimeSeriesKMeans
                        X_flat = st.session_state.X_ts.reshape(st.session_state.X_ts.shape[0], -1)
                        pca = PCA(n_components=2, random_state=42)
                        X_pca = pca.fit_transform(X_flat)
                        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                        df_pca['Cluster'] = st.session_state.hasil_ts['cluster'].values
                        df_pca['Kab/Kota'] = st.session_state.hasil_ts['nama_kabupaten_kota'].values
                        df_pca['Pendidikan'] = st.session_state.hasil_ts['pendidikan'].values
                        
                        fig = px.scatter(
                            df_pca, x='PC1', y='PC2', color='Cluster',
                            hover_data=['Kab/Kota', 'Pendidikan'],
                            title=f'🔵 TimeSeriesKMeans (k={st.session_state.final_k_ts})',
                            color_continuous_scale='Viridis',
                            size_max=15
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}% | PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
                
                with col2:
                    if st.session_state.hasil_hac is not None:
                        # PCA untuk HAC
                        X_flat = st.session_state.X_ts.reshape(st.session_state.X_ts.shape[0], -1)
                        pca = PCA(n_components=2, random_state=42)
                        X_pca = pca.fit_transform(X_flat)
                        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                        df_pca['Cluster'] = st.session_state.hasil_hac['cluster'].values
                        df_pca['Kab/Kota'] = st.session_state.hasil_hac['nama_kabupaten_kota'].values
                        df_pca['Pendidikan'] = st.session_state.hasil_hac['pendidikan'].values
                        
                        fig = px.scatter(
                            df_pca, x='PC1', y='PC2', color='Cluster',
                            hover_data=['Kab/Kota', 'Pendidikan'],
                            title=f'🔴 Hierarchical Agglomerative (k={st.session_state.final_k_hac})',
                            color_continuous_scale='Plasma',
                            size_max=15
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}% | PC2: {pca.explained_variance_ratio_[1]*100:.1f}%")
            
            with tab3:
                st.markdown("#### Pola Time Series per Cluster")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.hasil_ts is not None and st.session_state.tahun_order is not None:
                        st.markdown("##### 🔵 TimeSeriesKMeans")
                        fig = go.Figure()
                        for c in range(st.session_state.final_k_ts):
                            idx = np.where(st.session_state.hasil_ts['cluster'] == c)[0]
                            if len(idx) > 0:
                                mean_pattern = st.session_state.X_ts[idx].mean(axis=0).ravel()
                                mean_reshaped = mean_pattern.reshape(len(st.session_state.tahun_order), -1)
                                mean_per_year = mean_reshaped.mean(axis=1)
                                fig.add_trace(go.Scatter(
                                    x=st.session_state.tahun_order,
                                    y=mean_per_year,
                                    mode='lines+markers',
                                    name=f'Cluster {c}',
                                    line=dict(width=3)
                                ))
                        
                        fig.update_layout(
                            title=f'Rata-rata Pola Deret Waktu per Cluster',
                            xaxis_title='Tahun',
                            yaxis_title='Jumlah Pengangguran (Normalized)',
                            height=350,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if st.session_state.hasil_hac is not None and st.session_state.tahun_order is not None:
                        st.markdown("##### 🔴 Hierarchical Agglomerative")
                        fig = go.Figure()
                        for c in range(st.session_state.final_k_hac):
                            idx = np.where(st.session_state.hasil_hac['cluster'] == c)[0]
                            if len(idx) > 0:
                                mean_pattern = st.session_state.X_ts[idx].mean(axis=0).ravel()
                                mean_reshaped = mean_pattern.reshape(len(st.session_state.tahun_order), -1)
                                mean_per_year = mean_reshaped.mean(axis=1)
                                fig.add_trace(go.Scatter(
                                    x=st.session_state.tahun_order,
                                    y=mean_per_year,
                                    mode='lines+markers',
                                    name=f'Cluster {c}',
                                    line=dict(width=3)
                                ))
                        
                        fig.update_layout(
                            title=f'Rata-rata Pola Deret Waktu per Cluster',
                            xaxis_title='Tahun',
                            yaxis_title='Jumlah Pengangguran (Normalized)',
                            height=350,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                st.info("💡 **Insight:** Setiap cluster menunjukkan pola waktu yang berbeda - menurun, stabil, atau meningkat")
            
            with tab4:
                st.markdown("#### Heatmap Clustering")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.session_state.hasil_ts is not None:
                        st.markdown("##### 🔵 TimeSeriesKMeans")
                        pivot_cluster_ts = st.session_state.hasil_ts.pivot_table(
                            index='nama_kabupaten_kota', columns='pendidikan', values='cluster'
                        )
                        
                        fig = px.imshow(
                            pivot_cluster_ts,
                            title=f'TimeSeriesKMeans (k={st.session_state.final_k_ts})',
                            color_continuous_scale='Viridis',
                            aspect='auto'
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if st.session_state.hasil_hac is not None:
                        st.markdown("##### 🔴 Hierarchical Agglomerative")
                        pivot_cluster_hac = st.session_state.hasil_hac.pivot_table(
                            index='nama_kabupaten_kota', columns='pendidikan', values='cluster'
                        )
                        
                        fig = px.imshow(
                            pivot_cluster_hac,
                            title=f'Hierarchical Agglomerative (k={st.session_state.final_k_hac})',
                            color_continuous_scale='Plasma',
                            aspect='auto'
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                st.markdown("#### Perbandingan Visualisasi")
                
                if st.session_state.hasil_ts is not None and st.session_state.hasil_hac is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribusi Cluster TimeSeriesKMeans
                        ts_counts = st.session_state.hasil_ts['cluster'].value_counts().sort_index()
                        fig = px.pie(
                            values=ts_counts.values, names=ts_counts.index,
                            title=f'🔵 TimeSeriesKMeans (k={st.session_state.final_k_ts})',
                            color_discrete_sequence=px.colors.sequential.Viridis
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Distribusi Cluster HAC
                        hac_counts = st.session_state.hasil_hac['cluster'].value_counts().sort_index()
                        fig = px.pie(
                            values=hac_counts.values, names=hac_counts.index,
                            title=f'🔴 Hierarchical Agglomerative (k={st.session_state.final_k_hac})',
                            color_discrete_sequence=px.colors.sequential.Plasma
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Perbandingan Heatmap
                    st.markdown("##### Perbandingan Cluster per Kabupaten/Kota")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        pivot_ts = st.session_state.hasil_ts.pivot_table(
                            index='nama_kabupaten_kota', columns='pendidikan', values='cluster'
                        )
                        fig = px.imshow(
                            pivot_ts,
                            title='🔵 TimeSeriesKMeans',
                            color_continuous_scale='Viridis',
                            aspect='auto'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        pivot_hac = st.session_state.hasil_hac.pivot_table(
                            index='nama_kabupaten_kota', columns='pendidikan', values='cluster'
                        )
                        fig = px.imshow(
                            pivot_hac,
                            title='🔴 Hierarchical Agglomerative',
                            color_continuous_scale='Plasma',
                            aspect='auto'
                        )
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.info("""
                    💡 **Perbandingan Visual:**
                    - **🔵 TimeSeriesKMeans** mengelompokkan berdasarkan prototype centroid dengan DTW
                    - **🔴 Hierarchical Agglomerative** mengelompokkan berdasarkan distance matrix DTW
                    - Perhatikan perbedaan jumlah cluster dan distribusi antar cluster
                    """)
        else:
            st.warning("⚠️ Plotly tidak tersedia. Install dengan: pip install plotly")
    else:
        st.warning("⚠️ Silakan upload dataset dan jalankan analisis terlebih dahulu")

# ==================== UNDUH HASIL ====================
elif selected == "Unduh Hasil":
    st.markdown('<div class="sub-header">📥 Unduh Hasil</div>', unsafe_allow_html=True)
    
    if st.session_state.hasil_ts is not None:
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
                <h4>📊 Hasil siap untuk diunduh</h4>
                <p style="color: #7f8c8d;">Hasil analisis clustering dapat diunduh dalam berbagai format</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.session_state.hasil_ts is not None:
                csv_ts = st.session_state.hasil_ts.to_csv(index=False)
                st.download_button(
                    label="🔵 Unduh Hasil TimeSeriesKMeans (CSV)",
                    data=csv_ts,
                    file_name="hasil_timeserieskmeans.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            if st.session_state.hasil_hac is not None:
                csv_hac = st.session_state.hasil_hac.to_csv(index=False)
                st.download_button(
                    label="🔴 Unduh Hasil HAC (CSV)",
                    data=csv_hac,
                    file_name="hasil_hac.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if st.session_state.results_ts_df is not None and st.session_state.results_hac_df is not None:
                summary_data = {
                    'Algoritma': ['TimeSeriesKMeans', 'Hierarchical Agglomerative'],
                    'Cluster Terbaik': [st.session_state.final_k_ts, st.session_state.final_k_hac],
                    'Silhouette Score': [
                        st.session_state.results_ts_df.loc[
                            st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts,
                            'Silhouette Score (DTW)'
                        ].values[0],
                        st.session_state.results_hac_df.loc[
                            st.session_state.results_hac_df['Jumlah Cluster'] == st.session_state.final_k_hac,
                            'Silhouette Score (DTW)'
                        ].values[0]
                    ],
                    'Davies-Bouldin Index': [
                        st.session_state.results_ts_df.loc[
                            st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts,
                            'Davies-Bouldin Index'
                        ].values[0],
                        st.session_state.results_hac_df.loc[
                            st.session_state.results_hac_df['Jumlah Cluster'] == st.session_state.final_k_hac,
                            'Davies-Bouldin Index'
                        ].values[0]
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Unduh Ringkasan Hasil (CSV)",
                    data=csv_summary,
                    file_name="hasil_ringkasan.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.markdown("#### 🔵 Preview Hasil TimeSeriesKMeans")
        st.dataframe(st.session_state.hasil_ts.head(10), use_container_width=True)
        
        if st.session_state.hasil_hac is not None:
            st.markdown("#### 🔴 Preview Hasil HAC")
            st.dataframe(st.session_state.hasil_hac.head(10), use_container_width=True)
    else:
        st.warning("⚠️ Belum ada hasil analisis. Silakan jalankan analisis di menu 'Clustering'")

# ==================== TENTANG APLIKASI ====================
elif selected == "Tentang Aplikasi":
    st.markdown('<div class="sub-header">ℹ️ Tentang Aplikasi</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px;">
                <h3>📊 Sistem Komparasi Clustering Pengangguran Terbuka</h3>
                <p><strong>Versi:</strong> 3.0.0</p>
                <p><strong>Tanggal Rilis:</strong> 2026</p>
                <hr>
                <h4>🎯 Tujuan</h4>
                <p>Membantu analisis dan perbandingan hasil clustering antara algoritma 
                <strong>TimeSeriesKMeans</strong> dan <strong>Hierarchical Agglomerative Clustering (HAC)</strong> 
                untuk data pengangguran terbuka di Jawa Barat.</p>
                <hr>
                <h4>📚 Teknologi yang Digunakan</h4>
                <ul>
                    <li><strong>Framework:</strong> Streamlit</li>
                    <li><strong>Machine Learning:</strong> Scikit-learn, Tslearn</li>
                    <li><strong>Visualisasi:</strong> Plotly, Matplotlib, Seaborn</li>
                    <li><strong>Data Processing:</strong> Pandas, NumPy</li>
                </ul>
                <hr>
                <h4>📋 Metodologi</h4>
                <ol>
                    <li><strong>Preprocessing:</strong> Penggabungan pendidikan menjadi 4 kategori, filtering tahun 2017-2025</li>
                    <li><strong>Normalisasi:</strong> TimeSeriesScalerMinMax untuk kedua algoritma</li>
                    <li><strong>Clustering:</strong> TimeSeriesKMeans dan HAC dengan metric DTW</li>
                    <li><strong>Evaluasi:</strong> Elbow Method, Silhouette Score, Davies-Bouldin Index</li>
                    <li><strong>Komparasi:</strong> Perbandingan hasil kedua algoritma</li>
                </ol>
                <hr>
                <h4>📊 Perbedaan Algoritma</h4>
                <ul>
                    <li><strong>🔵 TimeSeriesKMeans:</strong> Prototype-based, efisien untuk data besar</li>
                    <li><strong>🔴 HAC:</strong> Hierarchical, tidak perlu menentukan k awal</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background-color: #e8f4f8; padding: 2rem; border-radius: 10px;">
                <h4>📞 Kontak</h4>
                <p><strong>Email:</strong> support@clustering-app.com</p>
                <p><strong>Website:</strong> www.clustering-app.com</p>
                <hr>
                <h4>🔗 Link Terkait</h4>
                <ul>
                    <li><a href="#">Dokumentasi</a></li>
                    <li><a href="#">GitHub Repository</a></li>
                    <li><a href="#">Laporan Akhir</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
    <div class="footer">
        © 2026 Sistem Komparasi Clustering Pengangguran Terbuka | TimeSeriesKMeans vs Hierarchical Agglomerative Clustering
    </div>
""", unsafe_allow_html=True)
