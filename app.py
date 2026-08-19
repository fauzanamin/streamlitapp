import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
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
    TSKLEARN_AVAILABLE = True
except ImportError:
    TSKLEARN_AVAILABLE = False
    st.warning("⚠️ tslearn tidak tersedia. Install dengan: pip install tslearn")

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
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown(
    '<div class="main-header">'
    '<span class="main-header-icon">📊</span>Sistem Komparasi Clustering Pengangguran Terbuka'
    '<div class="app-subtitle">Berdasarkan Tingkat Pendidikan di Jawa Barat</div>'
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
if 'hasil_km' not in st.session_state:
    st.session_state.hasil_km = None
if 'hasil_ts' not in st.session_state:
    st.session_state.hasil_ts = None
if 'final_k' not in st.session_state:
    st.session_state.final_k = None
if 'final_k_ts' not in st.session_state:
    st.session_state.final_k_ts = None
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'results_ts_df' not in st.session_state:
    st.session_state.results_ts_df = None

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
    """Preprocessing data sesuai dengan kode Google Colab"""
    data = df.copy()
    
    # Filter tahun 2020-2025
    data = data[(data['tahun'] >= 2020) & (data['tahun'] <= 2025)]
    
    # Gabung kategori pendidikan
    data['pendidikan'] = data['pendidikan'].replace(
        ['TIDAK/BELUM PERNAH SEKOLAH/TIDAK/BELUM TAMAT SD', 'SD'],
        'SD KE BAWAH'
    )
    
    pendidikan_order = ['SD KE BAWAH', 'SMP', 'SMA (UMUM)', 'SMA (KEJURUAN)', 'DIPLOMA I/II/III/AKADEMI/UNIVERSITAS']
    data['pendidikan'] = pd.Categorical(data['pendidikan'], categories=pendidikan_order, ordered=True)
    
    return data, pendidikan_order

def create_pivot_km(data, pendidikan_order):
    """Membuat pivot untuk K-Means (162 objek x 5 fitur)"""
    pivot_km = data.pivot_table(
        index=['nama_kabupaten_kota', 'tahun'],
        columns='pendidikan',
        values='jumlah_pengangguran',
        aggfunc='mean',
        observed=True
    )
    pivot_km = pivot_km[pendidikan_order]
    
    # Interpolasi jika ada missing value
    if pivot_km.isnull().values.any():
        pivot_km = pivot_km.interpolate(axis=0, limit_direction='both')
    
    return pivot_km

def normalize_data(pivot_km):
    """Normalisasi Min-Max GLOBAL"""
    scaler = MinMaxScaler()
    X_km_flat = pivot_km.values.reshape(-1, 1)
    X_km_scaled_flat = scaler.fit_transform(X_km_flat)
    X = X_km_scaled_flat.reshape(pivot_km.shape)
    
    pivot_km_scaled = pd.DataFrame(X, index=pivot_km.index, columns=pivot_km.columns)
    return pivot_km_scaled, X, scaler

def run_kmeans_clustering(X, K_range=range(2, 10)):
    """Menjalankan K-Means clustering dengan evaluasi"""
    inertia_list = []
    silhouette_list = []
    dbi_list = []
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        inertia_list.append(kmeans.inertia_)
        silhouette_list.append(silhouette_score(X, labels))
        dbi_list.append(davies_bouldin_score(X, labels))
    
    results_df = pd.DataFrame({
        'Jumlah Cluster': list(K_range),
        'Inertia': inertia_list,
        'Silhouette Score': silhouette_list,
        'Davies-Bouldin Index': dbi_list
    })
    
    # Penentuan cluster terbaik (voting)
    best_silhouette_k = results_df.loc[results_df['Silhouette Score'].idxmax(), 'Jumlah Cluster']
    best_dbi_k = results_df.loc[results_df['Davies-Bouldin Index'].idxmin(), 'Jumlah Cluster']
    
    inertia_diff = np.diff(inertia_list)
    inertia_ratio = inertia_diff[1:] / inertia_diff[:-1]
    best_elbow_k = list(K_range)[np.argmin(inertia_ratio) + 1]
    
    vote_counts = pd.Series([best_elbow_k, best_silhouette_k, best_dbi_k]).value_counts()
    final_k = vote_counts.index[0]
    
    return results_df, final_k, best_elbow_k, best_silhouette_k, best_dbi_k

def run_timeseries_clustering(data, pendidikan_order, K_range=range(2, 10)):
    """Menjalankan TimeSeries K-Means clustering dengan DTW"""
    if not TSKLEARN_AVAILABLE:
        return None, None, None, None, None, None
    
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
    
    # Bentuk array time series 3D
    X_ts_raw = to_time_series_dataset(pivot_ts.values)
    
    # Normalisasi per-deret
    scaler_ts = TimeSeriesScalerMinMax()
    X_ts = scaler_ts.fit_transform(X_ts_raw)
    X_ts_flat = X_ts.reshape(X_ts.shape[0], X_ts.shape[1])
    
    labels_id = pivot_ts.reset_index()[['nama_kabupaten_kota', 'pendidikan']]
    
    # Evaluasi
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
        
        dist_matrix = cdist_dtw(X_ts)
        sil = silhouette_score(dist_matrix, labels_k, metric="precomputed")
        silhouette_ts.append(sil)
        
        dbi = davies_bouldin_score(X_ts_flat, labels_k)
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
    
    hasil_ts = labels_id.copy()
    hasil_ts['cluster'] = final_labels_ts
    
    return hasil_ts, X_ts, X_ts_flat, results_ts_df, final_k_ts, tahun_order

# ==================== BERANDA ====================
if selected == "Beranda":
    st.markdown('<div class="sub-header">📊 Sistem Komparasi Clustering Pengangguran Terbuka Berdasarkan Tingkat Pendidikan di Jawa Barat</div>', unsafe_allow_html=True)
    st.caption("Aplikasi ini digunakan untuk melakukan analisis clustering menggunakan algoritma K-Means dan TimeSeriesKMeans serta membandingkan hasil kedua algoritma.")
    
    col_left, col_right = st.columns([1, 1.25], gap="large")

    with col_left:
        st.markdown('<div class="section-title">1. Upload Dataset</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Pilih file XLSX", type=['xlsx'], label_visibility="collapsed", key="home_uploader"
        )
        
        if uploaded_file is not None:
            try:
                df_home = pd.read_excel(uploaded_file)
                st.success(f"✅ File berhasil diupload! {len(df_home)} baris data")
                
                # Preprocessing
                data_home, pendidikan_order_home = preprocess_data(df_home)
                pivot_km_home = create_pivot_km(data_home, pendidikan_order_home)
                pivot_km_scaled_home, X_home, _ = normalize_data(pivot_km_home)
                
                st.session_state.data = data_home
                st.session_state.pivot_km = pivot_km_home
                st.session_state.X = X_home
                st.session_state.pendidikan_order = pendidikan_order_home
                
                st.dataframe(pivot_km_home.head(3), use_container_width=True, height=140)
                st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 0.6rem 1rem; border-radius: 5px; margin-top: 0.4rem; font-size: 0.85rem;">
                        <strong>Jumlah Baris:</strong> {len(pivot_km_home):,}&nbsp;&nbsp;|&nbsp;&nbsp;<strong>Jumlah Kolom:</strong> {len(pivot_km_home.columns)}
                    </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error membaca file: {e}")
        else:
            st.info("📁 Silakan upload file dataset untuk memulai analisis")
            st.markdown("""
                <div style="background-color: #f8f9fa; padding: 0.6rem 1rem; border-radius: 5px; margin-top: 0.4rem; font-size: 0.85rem;">
                    <strong>Format yang diharapkan:</strong> Kolom: nama_kabupaten_kota, tahun, pendidikan, jumlah_pengangguran
                </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.data is not None:
            st.markdown('<div class="section-title" style="margin-top:1.5rem;">4. Hasil Clustering</div>', unsafe_allow_html=True)
            
            if st.button("▶ Jalankan Analisis", use_container_width=True, key="run_home"):
                with st.spinner("Sedang menganalisis data..."):
                    # Jalankan K-Means
                    results_df, final_k, _, _, _ = run_kmeans_clustering(st.session_state.X)
                    st.session_state.results_df = results_df
                    st.session_state.final_k = final_k
                    
                    # K-Means final
                    kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
                    final_labels_km = kmeans_final.fit_predict(st.session_state.X)
                    
                    hasil_km = st.session_state.pivot_km.reset_index()
                    hasil_km['cluster'] = final_labels_km
                    st.session_state.hasil_km = hasil_km
                    
                    # TimeSeries K-Means
                    if TSKLEARN_AVAILABLE and st.session_state.data is not None:
                        hasil_ts, X_ts, X_ts_flat, results_ts_df, final_k_ts, _ = run_timeseries_clustering(
                            st.session_state.data, st.session_state.pendidikan_order
                        )
                        st.session_state.hasil_ts = hasil_ts
                        st.session_state.results_ts_df = results_ts_df
                        st.session_state.final_k_ts = final_k_ts
                
                st.success("✅ Analisis clustering selesai!")
                
                # Tampilkan ringkasan hasil
                result_data = {
                    "Algoritma": ["K-Means", "TimeSeriesKMeans"] if TSKLEARN_AVAILABLE else ["K-Means"],
                    "Cluster Terbaik": [final_k, st.session_state.final_k_ts] if TSKLEARN_AVAILABLE else [final_k],
                    "Silhouette Score": [
                        results_df.loc[results_df['Jumlah Cluster'] == final_k, 'Silhouette Score'].values[0],
                        st.session_state.results_ts_df.loc[st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts, 'Silhouette Score (DTW)'].values[0]
                    ] if TSKLEARN_AVAILABLE else [results_df.loc[results_df['Jumlah Cluster'] == final_k, 'Silhouette Score'].values[0]],
                }
                st.dataframe(pd.DataFrame(result_data), use_container_width=True, hide_index=True)

    with col_right:
        if st.session_state.data is not None:
            st.markdown('<div class="section-title">2. Ringkasan Data</div>', unsafe_allow_html=True)
            
            data = st.session_state.data
            r1, r2, r3, r4 = st.columns(4)
            
            stats = [
                (data['nama_kabupaten_kota'].nunique(), "Jumlah Wilayah", "Kab/Kota", "#3498db"),
                (data['pendidikan'].nunique(), "Jumlah Pendidikan", "Kategori", "#2ecc71"),
                (len(data['tahun'].unique()), "Jumlah Tahun", "(2020 - 2025)", "#e67e22"),
                (len(data), "Total Observasi", "Baris Data", "#e74c3c"),
            ]
            
            for col, (num, label, sub, color) in zip([r1, r2, r3, r4], stats):
                with col:
                    st.markdown(f"""
                        <div class="stat-card" style="border-left-color: {color}; padding: 0.9rem; margin-bottom:0;">
                            <div class="stat-number" style="font-size:1.3rem;">{num}</div>
                            <div class="stat-label" style="font-size:0.75rem;">{label}</div>
                            <div class="stat-label" style="font-size:0.7rem; margin-top:0;">{sub}</div>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title" style="margin-top:1.5rem;">3. Pilih Algoritma</div>', unsafe_allow_html=True)
            a1, a2 = st.columns([1.3, 1])
            with a1:
                st.markdown("**Pilih Algoritma Clustering**")
                algorithm_home = st.radio(
                    "Algoritma", ["K-Means", "TimeSeriesKMeans (DTW)", "Komparasi Kedua Algoritma"],
                    index=0, label_visibility="collapsed", key="algo_home"
                )
            with a2:
                st.markdown("**Jumlah Cluster (k)**")
                k_method_home = st.radio(
                    "Metode k", ["Otomatis (Rekomendasi)", "Manual"],
                    index=0, label_visibility="collapsed", key="k_method_home"
                )
                if k_method_home == "Manual":
                    st.number_input("Jumlah Cluster", min_value=2, max_value=10, value=3, key="k_manual_home")

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
                pivot_km = create_pivot_km(data, pendidikan_order)
                pivot_km_scaled, X, _ = normalize_data(pivot_km)
                
                st.session_state.data = data
                st.session_state.pivot_km = pivot_km
                st.session_state.X = X
                st.session_state.pendidikan_order = pendidikan_order
            
            st.markdown("#### Preview Dataset")
            st.dataframe(df.head(), use_container_width=True)
            
            st.markdown("#### Preview Data setelah Preprocessing (Pivot)")
            st.dataframe(pivot_km.head(), use_container_width=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Jumlah Wilayah", data['nama_kabupaten_kota'].nunique())
            with col2:
                st.metric("Jumlah Pendidikan", data['pendidikan'].nunique())
            with col3:
                st.metric("Jumlah Tahun", len(data['tahun'].unique()))
            with col4:
                st.metric("Total Observasi", len(data))
            
        except Exception as e:
            st.error(f"Error membaca file: {e}")
    else:
        st.info("📁 Silakan upload file dataset untuk memulai")
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                <h5>📋 Format Dataset yang Diharapkan</h5>
                <ul>
                    <li><strong>nama_kabupaten_kota</strong>: Nama kabupaten/kota</li>
                    <li><strong>tahun</strong>: Tahun (2020-2025)</li>
                    <li><strong>pendidikan</strong>: Tingkat pendidikan</li>
                    <li><strong>jumlah_pengangguran</strong>: Jumlah pengangguran</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ==================== EXPLORATORY DATA ANALYSIS ====================
elif selected == "Exploratory Data Analysis":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Statistik ringkasan
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Jumlah Wilayah", data['nama_kabupaten_kota'].nunique())
        with col2:
            st.metric("Jumlah Pendidikan", data['pendidikan'].nunique())
        with col3:
            st.metric("Jumlah Tahun", len(data['tahun'].unique()))
        with col4:
            st.metric("Total Observasi", len(data))
        
        st.markdown("#### Statistik Deskriptif Jumlah Pengangguran")
        st.dataframe(data['jumlah_pengangguran'].describe(), use_container_width=True)
        
        st.markdown("#### Distribusi Jumlah Pengangguran per Tingkat Pendidikan")
        edu_stats = data.groupby('pendidikan')['jumlah_pengangguran'].sum().sort_values(ascending=False)
        st.dataframe(edu_stats, use_container_width=True)
        
        if PLOTLY_AVAILABLE:
            st.markdown("#### Visualisasi Data")
            
            tab1, tab2, tab3 = st.tabs(["Tren Pengangguran", "Distribusi per Tahun", "Heatmap"])
            
            with tab1:
                fig = px.line(
                    data, x='tahun', y='jumlah_pengangguran', 
                    color='pendidikan', 
                    title='Tren Pengangguran Terbuka per Pendidikan (2020-2025)',
                    markers=True
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                fig = px.box(
                    data, x='tahun', y='jumlah_pengangguran',
                    title='Distribusi Jumlah Pengangguran per Tahun'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                # Heatmap dari pivot
                if st.session_state.pivot_km is not None:
                    heatmap_data = st.session_state.pivot_km.T
                    fig = px.imshow(
                        heatmap_data,
                        title='Heatmap Pengangguran per Kab/Kota dan Pendidikan',
                        color_continuous_scale='RdBu_r',
                        aspect='auto'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Plotly tidak tersedia. Install dengan: pip install plotly")
    else:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")

# ==================== CLUSTERING ====================
elif selected == "Clustering":
    st.markdown('<div class="sub-header">🎯 Clustering</div>', unsafe_allow_html=True)
    
    if st.session_state.X is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Pilih Algoritma Clustering")
            algorithm = st.radio(
                "Pilih Algoritma",
                ["K-Means", "TimeSeriesKMeans (DTW)", "Komparasi Kedua Algoritma"],
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
                if algorithm in ["K-Means", "Komparasi Kedua Algoritma"]:
                    results_df, final_k, best_elbow, best_sil, best_dbi = run_kmeans_clustering(st.session_state.X)
                    st.session_state.results_df = results_df
                    st.session_state.final_k = final_k
                    
                    kmeans_final = KMeans(n_clusters=final_k, random_state=42, n_init=10)
                    final_labels_km = kmeans_final.fit_predict(st.session_state.X)
                    
                    hasil_km = st.session_state.pivot_km.reset_index()
                    hasil_km['cluster'] = final_labels_km
                    st.session_state.hasil_km = hasil_km
                
                if algorithm in ["TimeSeriesKMeans (DTW)", "Komparasi Kedua Algoritma"]:
                    if TSKLEARN_AVAILABLE:
                        hasil_ts, X_ts, X_ts_flat, results_ts_df, final_k_ts, _ = run_timeseries_clustering(
                            st.session_state.data, st.session_state.pendidikan_order
                        )
                        st.session_state.hasil_ts = hasil_ts
                        st.session_state.results_ts_df = results_ts_df
                        st.session_state.final_k_ts = final_k_ts
                    else:
                        st.error("⚠️ tslearn tidak tersedia. Install dengan: pip install tslearn")
            
            st.success("✅ Analisis clustering selesai!")
        
        # Tampilkan hasil jika sudah ada
        if st.session_state.hasil_km is not None:
            st.markdown("#### Hasil Clustering K-Means")
            st.markdown(f"**Cluster Terbaik (Voting):** {st.session_state.final_k}")
            st.dataframe(st.session_state.results_df, use_container_width=True)
            
            st.markdown("#### Distribusi Cluster")
            cluster_counts = st.session_state.hasil_km['cluster'].value_counts().sort_index()
            st.dataframe(cluster_counts, use_container_width=True)
            
            if PLOTLY_AVAILABLE:
                fig = px.bar(
                    x=cluster_counts.index, y=cluster_counts.values,
                    title='Distribusi Cluster K-Means',
                    labels={'x': 'Cluster', 'y': 'Jumlah'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        if st.session_state.hasil_ts is not None:
            st.markdown("#### Hasil Clustering TimeSeriesKMeans")
            st.markdown(f"**Cluster Terbaik (Voting):** {st.session_state.final_k_ts}")
            st.dataframe(st.session_state.results_ts_df, use_container_width=True)
            
            st.markdown("#### Distribusi Cluster")
            cluster_counts_ts = st.session_state.hasil_ts['cluster'].value_counts().sort_index()
            st.dataframe(cluster_counts_ts, use_container_width=True)
    else:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'")

# ==================== KOMPARASI ALGORITMA ====================
elif selected == "Komparasi Algoritma":
    st.markdown('<div class="sub-header">🔍 Komparasi Algoritma</div>', unsafe_allow_html=True)
    
    if st.session_state.hasil_km is not None and st.session_state.hasil_ts is not None:
        st.markdown("#### Perbandingan Metrik")
        
        col1, col2 = st.columns(2)
        
        with col1:
            km_silhouette = st.session_state.results_df.loc[
                st.session_state.results_df['Jumlah Cluster'] == st.session_state.final_k, 
                'Silhouette Score'
            ].values[0]
            km_dbi = st.session_state.results_df.loc[
                st.session_state.results_df['Jumlah Cluster'] == st.session_state.final_k, 
                'Davies-Bouldin Index'
            ].values[0]
            
            st.markdown(f"""
                <div class="result-card">
                    <h4>K-Means</h4>
                    <p><strong>Cluster Terbaik:</strong> {st.session_state.final_k}</p>
                    <p><strong>Silhouette Score:</strong> <span class="metric-good">{km_silhouette:.4f}</span></p>
                    <p><strong>Davies-Bouldin Index:</strong> <span class="metric-bad">{km_dbi:.4f}</span></p>
                    <p><strong>Keunggulan:</strong></p>
                    <ul>
                        <li>Cepat dan efisien</li>
                        <li>Mudah diimplementasikan</li>
                        <li>Cocok untuk data dengan dimensi rendah</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ts_silhouette = st.session_state.results_ts_df.loc[
                st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts, 
                'Silhouette Score (DTW)'
            ].values[0]
            ts_dbi = st.session_state.results_ts_df.loc[
                st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts, 
                'Davies-Bouldin Index'
            ].values[0]
            
            st.markdown(f"""
                <div class="result-card">
                    <h4>TimeSeriesKMeans (DTW)</h4>
                    <p><strong>Cluster Terbaik:</strong> {st.session_state.final_k_ts}</p>
                    <p><strong>Silhouette Score:</strong> <span class="metric-good">{ts_silhouette:.4f}</span></p>
                    <p><strong>Davies-Bouldin Index:</strong> <span class="metric-good">{ts_dbi:.4f}</span></p>
                    <p><strong>Keunggulan:</strong></p>
                    <ul>
                        <li>Memperhatikan pola waktu</li>
                        <li>Lebih akurat untuk data time series</li>
                        <li>Menggunakan DTW (Dynamic Time Warping)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        if PLOTLY_AVAILABLE:
            st.markdown("#### Visualisasi Perbandingan")
            
            metrics = ['Silhouette Score', 'Davies-Bouldin Index']
            kmeans_values = [km_silhouette, km_dbi]
            tskm_values = [ts_silhouette, ts_dbi]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name='K-Means',
                x=metrics,
                y=kmeans_values,
                marker_color='#3498db'
            ))
            fig.add_trace(go.Bar(
                name='TimeSeriesKMeans',
                x=metrics,
                y=tskm_values,
                marker_color='#e74c3c'
            ))
            
            fig.update_layout(
                title='Perbandingan Metrik Clustering',
                barmode='group',
                height=400,
                yaxis_title='Nilai'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Tambahan insight
            st.info(f"""
            💡 **Analisis Komparasi:**
            - **Silhouette Score:** TimeSeriesKMeans ({ts_silhouette:.4f}) {'lebih baik' if ts_silhouette > km_silhouette else 'kurang baik'} dari K-Means ({km_silhouette:.4f})
            - **Davies-Bouldin Index:** TimeSeriesKMeans ({ts_dbi:.4f}) {'lebih baik' if ts_dbi < km_dbi else 'kurang baik'} dari K-Means ({km_dbi:.4f})
            - **Kesimpulan:** Algoritma {'TimeSeriesKMeans' if ts_silhouette > km_silhouette and ts_dbi < km_dbi else 'K-Means'} memberikan hasil clustering yang lebih baik untuk dataset ini
            """)
    else:
        st.warning("⚠️ Silakan jalankan analisis clustering terlebih dahulu di menu 'Clustering'")

# ==================== VISUALISASI ====================
elif selected == "Visualisasi":
    st.markdown('<div class="sub-header">📈 Visualisasi</div>', unsafe_allow_html=True)
    
    if st.session_state.X is not None:
        if PLOTLY_AVAILABLE:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Elbow Curve", "Scatter Plot", "Heatmap", "Time Series Plot", "Perbandingan"
            ])
            
            with tab1:
                st.markdown("#### Elbow Curve")
                
                if st.session_state.results_df is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.line(
                            st.session_state.results_df, x='Jumlah Cluster', y='Inertia',
                            markers=True,
                            title='Elbow Curve - K-Means'
                        )
                        fig.add_vline(
                            x=st.session_state.final_k, line_dash="dash", line_color="red",
                            annotation_text=f"Optimal k={st.session_state.final_k}"
                        )
                        fig.update_layout(height=350)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if st.session_state.results_ts_df is not None:
                            fig = px.line(
                                st.session_state.results_ts_df, x='Jumlah Cluster', y='Inertia',
                                markers=True,
                                title='Elbow Curve - TimeSeriesKMeans'
                            )
                            fig.add_vline(
                                x=st.session_state.final_k_ts, line_dash="dash", line_color="red",
                                annotation_text=f"Optimal k={st.session_state.final_k_ts}"
                            )
                            fig.update_layout(height=350)
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("#### Scatter Plot (PCA 2D)")
                
                # PCA untuk visualisasi
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(st.session_state.X)
                
                if st.session_state.hasil_km is not None:
                    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                    df_pca['Cluster'] = st.session_state.hasil_km['cluster'].values
                    
                    fig = px.scatter(
                        df_pca, x='PC1', y='PC2', color='Cluster',
                        title=f'Visualisasi Cluster K-Means (k={st.session_state.final_k})',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}% variance | PC2: {pca.explained_variance_ratio_[1]*100:.1f}% variance")
            
            with tab3:
                st.markdown("#### Heatmap Cluster")
                
                if st.session_state.hasil_km is not None:
                    # Heatmap cluster per kab/kota dan tahun
                    pivot_cluster = st.session_state.hasil_km.pivot_table(
                        index='nama_kabupaten_kota', columns='tahun', values='cluster'
                    )
                    
                    fig = px.imshow(
                        pivot_cluster,
                        title=f'Cluster K-Means per Kabupaten/Kota dan Tahun (k={st.session_state.final_k})',
                        color_continuous_scale='Viridis',
                        aspect='auto'
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.markdown("#### Time Series Plot per Cluster")
                
                if st.session_state.hasil_ts is not None:
                    # Data untuk time series plot
                    data = st.session_state.data
                    data_ts = data.merge(
                        st.session_state.hasil_ts[['nama_kabupaten_kota', 'pendidikan', 'cluster']],
                        on=['nama_kabupaten_kota', 'pendidikan'],
                        how='left'
                    )
                    
                    # Plot rata-rata per cluster
                    fig = go.Figure()
                    
                    for c in sorted(data_ts['cluster'].unique()):
                        cluster_data = data_ts[data_ts['cluster'] == c]
                        mean_by_year = cluster_data.groupby('tahun')['jumlah_pengangguran'].mean()
                        
                        fig.add_trace(go.Scatter(
                            x=mean_by_year.index,
                            y=mean_by_year.values,
                            mode='lines+markers',
                            name=f'Cluster {c}'
                        ))
                    
                    fig.update_layout(
                        title=f'Rata-rata Pola Pengangguran per Cluster (k={st.session_state.final_k_ts})',
                        xaxis_title='Tahun',
                        yaxis_title='Jumlah Pengangguran',
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab5:
                st.markdown("#### Perbandingan Algoritma")
                
                if st.session_state.hasil_km is not None and st.session_state.hasil_ts is not None:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # PCA K-Means
                        pca = PCA(n_components=2, random_state=42)
                        X_pca = pca.fit_transform(st.session_state.X)
                        df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
                        df_pca['Cluster'] = st.session_state.hasil_km['cluster'].values
                        
                        fig1 = px.scatter(
                            df_pca, x='PC1', y='PC2', color='Cluster',
                            title=f'K-Means (k={st.session_state.final_k})',
                            color_continuous_scale='Viridis'
                        )
                        fig1.update_layout(height=350)
                        st.plotly_chart(fig1, use_container_width=True)
                    
                    with col2:
                        # PCA TimeSeriesKMeans
                        if st.session_state.hasil_ts is not None:
                            # Ambil data dari pivot_km untuk TS
                            X_ts_flat = st.session_state.pivot_km.values
                            pca_ts = PCA(n_components=2, random_state=42)
                            X_pca_ts = pca_ts.fit_transform(X_ts_flat)
                            
                            # Match labels ke pivot
                            labels_ts = []
                            for idx in range(len(X_pca_ts)):
                                kab = st.session_state.pivot_km.index[idx][0]
                                tahun = st.session_state.pivot_km.index[idx][1]
                                # Cari cluster dari hasil_ts (yang berdasarkan kab+pendidikan)
                                # Untuk visualisasi, kita ambil cluster dari kab saja
                                cluster_val = st.session_state.hasil_ts[
                                    st.session_state.hasil_ts['nama_kabupaten_kota'] == kab
                                ]['cluster'].mode().values[0]
                                labels_ts.append(cluster_val)
                            
                            df_pca_ts = pd.DataFrame(X_pca_ts, columns=['PC1', 'PC2'])
                            df_pca_ts['Cluster'] = labels_ts
                            
                            fig2 = px.scatter(
                                df_pca_ts, x='PC1', y='PC2', color='Cluster',
                                title=f'TimeSeriesKMeans (k={st.session_state.final_k_ts})',
                                color_continuous_scale='Plasma'
                            )
                            fig2.update_layout(height=350)
                            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("⚠️ Plotly tidak tersedia. Install dengan: pip install plotly")
    else:
        st.warning("⚠️ Silakan upload dataset dan jalankan analisis terlebih dahulu")

# ==================== UNDUH HASIL ====================
elif selected == "Unduh Hasil":
    st.markdown('<div class="sub-header">📥 Unduh Hasil</div>', unsafe_allow_html=True)
    
    if st.session_state.hasil_km is not None:
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
                <h4>📊 Hasil siap untuk diunduh</h4>
                <p style="color: #7f8c8d;">Hasil analisis clustering dapat diunduh dalam berbagai format</p>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV K-Means
            csv_km = st.session_state.hasil_km.to_csv(index=False)
            st.download_button(
                label="📊 Unduh Hasil K-Means (CSV)",
                data=csv_km,
                file_name="hasil_kmeans.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # CSV TimeSeriesKMeans
            if st.session_state.hasil_ts is not None:
                csv_ts = st.session_state.hasil_ts.to_csv(index=False)
                st.download_button(
                    label="📊 Unduh Hasil TimeSeriesKMeans (CSV)",
                    data=csv_ts,
                    file_name="hasil_timeserieskmeans.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            # CSV Ringkasan
            if st.session_state.results_df is not None:
                summary_data = {
                    'Algoritma': ['K-Means'],
                    'Cluster Terbaik': [st.session_state.final_k],
                    'Silhouette Score': [st.session_state.results_df.loc[
                        st.session_state.results_df['Jumlah Cluster'] == st.session_state.final_k,
                        'Silhouette Score'
                    ].values[0]],
                    'Davies-Bouldin Index': [st.session_state.results_df.loc[
                        st.session_state.results_df['Jumlah Cluster'] == st.session_state.final_k,
                        'Davies-Bouldin Index'
                    ].values[0]]
                }
                
                if st.session_state.results_ts_df is not None:
                    summary_data['Algoritma'].append('TimeSeriesKMeans')
                    summary_data['Cluster Terbaik'].append(st.session_state.final_k_ts)
                    summary_data['Silhouette Score'].append(st.session_state.results_ts_df.loc[
                        st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts,
                        'Silhouette Score (DTW)'
                    ].values[0])
                    summary_data['Davies-Bouldin Index'].append(st.session_state.results_ts_df.loc[
                        st.session_state.results_ts_df['Jumlah Cluster'] == st.session_state.final_k_ts,
                        'Davies-Bouldin Index'
                    ].values[0])
                
                summary_df = pd.DataFrame(summary_data)
                csv_summary = summary_df.to_csv(index=False)
                st.download_button(
                    label="📊 Unduh Ringkasan Hasil (CSV)",
                    data=csv_summary,
                    file_name="hasil_ringkasan.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.markdown("#### Preview Hasil K-Means")
        st.dataframe(st.session_state.hasil_km.head(10), use_container_width=True)
        
        if st.session_state.hasil_ts is not None:
            st.markdown("#### Preview Hasil TimeSeriesKMeans")
            st.dataframe(st.session_state.hasil_ts.head(10), use_container_width=True)
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
                <p><strong>Versi:</strong> 2.0.0</p>
                <p><strong>Tanggal Rilis:</strong> 2026</p>
                <hr>
                <h4>🎯 Tujuan</h4>
                <p>Membantu analisis dan perbandingan hasil clustering antara algoritma 
                K-Means dan TimeSeriesKMeans untuk data pengangguran terbuka di Jawa Barat.</p>
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
                    <li><strong>Preprocessing:</strong> Penggabungan kategori pendidikan, filtering tahun 2020-2025</li>
                    <li><strong>Normalisasi:</strong> Min-Max Global untuk K-Means, TimeSeriesScalerMinMax untuk TS-KMeans</li>
                    <li><strong>Clustering:</strong> K-Means dan TimeSeriesKMeans dengan DTW</li>
                    <li><strong>Evaluasi:</strong> Elbow Method, Silhouette Score, Davies-Bouldin Index</li>
                    <li><strong>Komparasi:</strong> Perbandingan hasil kedua algoritma</li>
                </ol>
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
        © 2026 Sistem Komparasi Clustering Pengangguran Terbuka. All rights reserved.
    </div>
""", unsafe_allow_html=True)
