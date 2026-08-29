# -*- coding: utf-8 -*-
"""
SISTEM ANALISIS PENGELOMPOKAN KABUPATEN/KOTA DI JAWA BARAT
BERDASARKAN KARAKTERISTIK PERKEMBANGAN JUMLAH PENGANGGURAN TERBUKA
MENURUT TINGKAT PENDIDIKAN MENGGUNAKAN METODE K-MEDOIDS

Rakha Rizky Mahendra - NPM 21083010013
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# ==========================================================
# KONFIGURASI HALAMAN
# ==========================================================
st.set_page_config(
    page_title="Analisis K-Medoids Pengangguran Jawa Barat",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# IMPLEMENTASI K-MEDOIDS MANUAL (PAM Algorithm)
# ==========================================================

class KMedoidsManual:
    """
    Implementasi K-Medoids menggunakan algoritma PAM
    (Partitioning Around Medoids) - Compatible dengan Python 3.14+
    """
    
    def __init__(self, n_clusters=3, metric='euclidean', random_state=None, 
                 max_iter=300, init='heuristic'):
        self.n_clusters = n_clusters
        self.metric = metric
        self.random_state = random_state
        self.max_iter = max_iter
        self.init = init
        self.medoid_indices_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def fit(self, X):
        """Menjalankan algoritma PAM K-Medoids"""
        
        n_samples, n_features = X.shape
        
        if n_samples < self.n_clusters:
            raise ValueError(f"n_samples={n_samples} < n_clusters={self.n_clusters}")
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Inisialisasi medoids
        if self.init == 'heuristic':
            medoid_indices = self._heuristic_init(X, self.n_clusters)
        else:
            medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        # Iterasi PAM
        for iteration in range(self.max_iter):
            self.n_iter_ = iteration + 1
            
            distances = cdist(X, X[medoid_indices], metric=self.metric)
            labels = np.argmin(distances, axis=1)
            current_cost = np.sum(np.min(distances, axis=1))
            
            improved = False
            best_medoids = medoid_indices.copy()
            best_cost = current_cost
            
            non_medoid_indices = [i for i in range(n_samples) if i not in medoid_indices]
            
            if len(non_medoid_indices) > 100:
                np.random.shuffle(non_medoid_indices)
                non_medoid_indices = non_medoid_indices[:100]
            
            for i in non_medoid_indices:
                for j in range(self.n_clusters):
                    new_medoids = medoid_indices.copy()
                    new_medoids[j] = i
                    
                    new_distances = cdist(X, X[new_medoids], metric=self.metric)
                    new_cost = np.sum(np.min(new_distances, axis=1))
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_medoids = new_medoids.copy()
                        improved = True
            
            if improved:
                medoid_indices = best_medoids
            else:
                break
        
        self.medoid_indices_ = medoid_indices
        distances = cdist(X, X[medoid_indices], metric=self.metric)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1))
        
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def _heuristic_init(self, X, k):
        n_samples = X.shape[0]
        
        if k >= n_samples:
            return np.arange(k)
        
        medoid_indices = []
        
        if n_samples > 0:
            first_idx = np.random.choice(n_samples)
            medoid_indices.append(first_idx)
            
            for _ in range(k - 1):
                if len(medoid_indices) > 0:
                    distances = cdist(X, X[medoid_indices], metric=self.metric)
                    min_distances = np.min(distances, axis=1)
                    next_idx = np.argmax(min_distances)
                    medoid_indices.append(next_idx)
                else:
                    medoid_indices.append(np.random.choice(n_samples))
        
        return np.array(medoid_indices)

# ==========================================================
# CUSTOM CSS - DIPERBAIKI
# ==========================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2rem;
        color: #1f77b4;
        text-align: center;
        padding: 0.8rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.3rem;
        color: #2c3e50;
        padding: 0.5rem 0;
        border-bottom: 2px solid #ecf0f1;
        margin-bottom: 1rem;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #7f8c8d;
    }
    .result-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .upload-area {
        border: 2px dashed #1f77b4;
        padding: 2rem;
        text-align: center;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1.5rem;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding: 1rem 0;
        border-top: 1px solid #ecf0f1;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================
if 'page' not in st.session_state:
    st.session_state.page = "Beranda"

# Data states
if 'data' not in st.session_state:
    st.session_state.data = None
if 'pivot_data' not in st.session_state:
    st.session_state.pivot_data = None
if 'df_karakteristik' not in st.session_state:
    st.session_state.df_karakteristik = None
if 'df_standardized' not in st.session_state:
    st.session_state.df_standardized = None
if 'clustering_results' not in st.session_state:
    st.session_state.clustering_results = None
if 'best_k' not in st.session_state:
    st.session_state.best_k = None
if 'best_labels' not in st.session_state:
    st.session_state.best_labels = None
if 'best_medoids' not in st.session_state:
    st.session_state.best_medoids = None
if 'evaluation_df' not in st.session_state:
    st.session_state.evaluation_df = None
if 'cluster_means' not in st.session_state:
    st.session_state.cluster_means = None
if 'karakteristik_columns' not in st.session_state:
    st.session_state.karakteristik_columns = None
if 'pendidikan_order' not in st.session_state:
    st.session_state.pendidikan_order = ['SD KE BAWAH', 'SMP', 'SMA/SLTA', 'PERGURUAN TINGGI']

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
    <div class="main-header">
        📊 Analisis Pengelompokan Kabupaten/Kota di Jawa Barat<br>
        <small style="font-size: 0.9rem; color: #555;">
            Berdasarkan Karakteristik Perkembangan Jumlah Pengangguran Terbuka<br>
            Menggunakan Metode K-Medoids
        </small>
    </div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR MENU - DIPERBAIKI
# ==========================================================
with st.sidebar:
    st.markdown("### 📋 MENU NAVIGASI")
    st.markdown("---")
    
    # Menggunakan tombol biasa untuk navigasi (lebih robust)
    if st.button("🏠 Beranda", use_container_width=True, key="btn_beranda"):
        st.session_state.page = "Beranda"
        st.rerun()
    
    if st.button("📤 Upload Dataset", use_container_width=True, key="btn_upload"):
        st.session_state.page = "Upload Dataset"
        st.rerun()
    
    if st.button("📊 Exploratory Data Analysis", use_container_width=True, key="btn_eda"):
        st.session_state.page = "EDA"
        st.rerun()
    
    if st.button("⚙️ Preprocessing & Karakteristik", use_container_width=True, key="btn_preprocess"):
        st.session_state.page = "Preprocessing"
        st.rerun()
    
    if st.button("🎯 Clustering K-Medoids", use_container_width=True, key="btn_cluster"):
        st.session_state.page = "Clustering"
        st.rerun()
    
    if st.button("📈 Evaluasi & Hasil", use_container_width=True, key="btn_eval"):
        st.session_state.page = "Evaluasi"
        st.rerun()
    
    if st.button("📥 Unduh Hasil", use_container_width=True, key="btn_download"):
        st.session_state.page = "Unduh"
        st.rerun()
    
    if st.button("ℹ️ Tentang", use_container_width=True, key="btn_about"):
        st.session_state.page = "Tentang"
        st.rerun()
    
    st.markdown("---")
    st.markdown(f"""
        <div style="font-size: 0.8rem; color: #7f8c8d; text-align: center;">
            <strong>Rakha Rizky Mahendra</strong><br>
            NPM 21083010013<br>
            Sains Data - UPN Veteran Jatim
        </div>
    """, unsafe_allow_html=True)
    
    # Status data
    if st.session_state.data is not None:
        st.success("✅ Data loaded")
    else:
        st.warning("⚠️ No data")

# ==========================================================
# FUNGSI-FUNGSI UTAMA
# ==========================================================

def calculate_slope(y):
    t = np.arange(len(y))
    if len(y) < 2:
        return 0
    return np.polyfit(t, y, 1)[0]

def calculate_cv(y):
    mean_y = np.mean(y)
    if mean_y == 0:
        return 0
    return np.std(y, ddof=1) / mean_y

def process_data(data):
    """Proses data lengkap"""
    data = data[(data['tahun'] >= 2017) & (data['tahun'] <= 2025)]
    
    # Harmonisasi pendidikan
    data['pendidikan'] = data['pendidikan'].replace([
        'TIDAK/BELUM PERNAH SEKOLAH/TIDAK/BELUM TAMAT SD', 'SD'
    ], 'SD KE BAWAH')
    data['pendidikan'] = data['pendidikan'].replace([
        'SMA (UMUM)', 'SMA (KEJURUAN)'
    ], 'SMA/SLTA')
    data['pendidikan'] = data['pendidikan'].replace([
        'DIPLOMA I/II/III/AKADEMI/UNIVERSITAS'
    ], 'PERGURUAN TINGGI')
    
    pendidikan_order = ['SD KE BAWAH', 'SMP', 'SMA/SLTA', 'PERGURUAN TINGGI']
    data['pendidikan'] = pd.Categorical(data['pendidikan'], categories=pendidikan_order, ordered=True)
    
    # Pivot
    pivot_data = data.pivot_table(
        index=['nama_kabupaten_kota', 'tahun'],
        columns='pendidikan',
        values='jumlah_pengangguran',
        aggfunc='mean',
        observed=True
    )
    pivot_data = pivot_data[pendidikan_order]
    
    if pivot_data.isnull().values.any():
        pivot_data = pivot_data.interpolate(axis=0, limit_direction='both')
    
    # Karakteristik
    kabupaten_list = pivot_data.index.get_level_values('nama_kabupaten_kota').unique()
    karakteristik_all = {}
    
    for kab in kabupaten_list:
        kab_data = pivot_data.xs(kab, level='nama_kabupaten_kota')
        karak = []
        for edu in pendidikan_order:
            series = kab_data[edu].values
            karak.extend([np.mean(series), calculate_slope(series), calculate_cv(series)])
        karakteristik_all[kab] = karak
    
    karakteristik_columns = []
    for edu in pendidikan_order:
        karakteristik_columns.extend([f'Mean_{edu}', f'Slope_{edu}', f'CV_{edu}'])
    
    df_karakteristik = pd.DataFrame.from_dict(karakteristik_all, orient='index', columns=karakteristik_columns)
    df_karakteristik.index.name = 'Kabupaten_Kota'
    df_karakteristik = df_karakteristik.reset_index()
    
    # Standardisasi dengan RobustScaler
    fitur = df_karakteristik[karakteristik_columns]
    scaler = RobustScaler()
    fitur_standardized = scaler.fit_transform(fitur)
    df_standardized = pd.DataFrame(
        fitur_standardized,
        columns=karakteristik_columns,
        index=df_karakteristik['Kabupaten_Kota']
    )
    
    return {
        'data': data,
        'pivot_data': pivot_data,
        'df_karakteristik': df_karakteristik,
        'df_standardized': df_standardized,
        'pendidikan_order': pendidikan_order,
        'karakteristik_columns': karakteristik_columns,
        'kabupaten_list': kabupaten_list
    }

def run_clustering(df_standardized, k_range):
    """Menjalankan K-Medoids clustering"""
    results = {}
    X = df_standardized.values
    
    for k in k_range:
        kmedoids = KMedoidsManual(
            n_clusters=k,
            metric='euclidean',
            random_state=42,
            max_iter=300
        )
        
        labels = kmedoids.fit_predict(X)
        medoid_indices = kmedoids.medoid_indices_
        medoid_labels = df_standardized.index[medoid_indices].tolist()
        
        sil_score = silhouette_score(X, labels, metric='euclidean')
        dbi_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        results[k] = {
            'labels': labels,
            'medoid_indices': medoid_indices,
            'medoid_labels': medoid_labels,
            'silhouette': sil_score,
            'dbi': dbi_score,
            'ch': ch_score,
            'inertia': kmedoids.inertia_
        }
    
    return results

# ==========================================================
# MENU: BERANDA
# ==========================================================
if st.session_state.page == "Beranda":
    st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f4f8 0%, #f0f8ff 100%); 
                    padding: 2rem; border-radius: 10px; margin-bottom: 1.5rem;">
            <h3 style="color: #1f77b4; margin-bottom: 0.5rem;">👋 Selamat Datang!</h3>
            <p style="font-size: 1.05rem; color: #2c3e50;">
                Aplikasi ini digunakan untuk menganalisis pengelompokan 27 kabupaten/kota di Provinsi Jawa Barat
                berdasarkan karakteristik perkembangan jumlah pengangguran terbuka menurut tingkat pendidikan
                menggunakan metode <strong>K-Medoids</strong>.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #3498db;">
                <div class="stat-number">27</div>
                <div class="stat-label">Kabupaten/Kota</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #2ecc71;">
                <div class="stat-number">4</div>
                <div class="stat-label">Kategori Pendidikan</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e67e22;">
                <div class="stat-number">12</div>
                <div class="stat-label">Fitur Karakteristik</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e74c3c;">
                <div class="stat-number">2017-2025</div>
                <div class="stat-label">Periode Analisis</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="result-card">
            <h4>📋 Langkah Analisis</h4>
            <ol>
                <li><strong>Upload Dataset</strong> - Unggah file data pengangguran (.xlsx)</li>
                <li><strong>Exploratory Data Analysis</strong> - Eksplorasi karakteristik data</li>
                <li><strong>Preprocessing & Karakteristik</strong> - Harmonisasi data dan pembentukan 12 fitur</li>
                <li><strong>Clustering K-Medoids</strong> - Pengelompokan dengan K-Medoids (k=2 sampai k=10)</li>
                <li><strong>Evaluasi & Hasil</strong> - Evaluasi dengan Silhouette Score dan analisis cluster</li>
                <li><strong>Unduh Hasil</strong> - Download hasil analisis</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

# ==========================================================
# MENU: UPLOAD DATASET
# ==========================================================
elif st.session_state.page == "Upload Dataset":
    st.markdown('<div class="sub-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="upload-area">
            <h4>📁 Upload Dataset Pengangguran</h4>
            <p style="color: #7f8c8d;">Format: .xlsx (Excel)</p>
            <p style="color: #95a5a6; font-size: 0.85rem;">
                Data bersumber dari Open Data Jawa Barat - Dinas Tenaga Kerja dan Transmigrasi
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file XLSX", type=['xlsx'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        try:
            data = pd.read_excel(uploaded_file)
            st.session_state.data = data
            
            st.success(f"✅ File berhasil diupload! {len(data)} baris data")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Preview Dataset")
                st.dataframe(data.head(), use_container_width=True)
            with col2:
                st.markdown("#### Informasi Dataset")
                st.write(f"**Jumlah Baris:** {len(data)}")
                st.write(f"**Jumlah Kolom:** {len(data.columns)}")
                st.write(f"**Kolom:** {', '.join(data.columns.tolist())}")
                st.write(f"**Tahun:** {sorted(data['tahun'].unique())}")
                st.write(f"**Kabupaten/Kota:** {data['nama_kabupaten_kota'].nunique()}")
            
            # Proses data otomatis
            with st.spinner("Memproses data..."):
                processed = process_data(data)
                st.session_state.pivot_data = processed['pivot_data']
                st.session_state.df_karakteristik = processed['df_karakteristik']
                st.session_state.df_standardized = processed['df_standardized']
                st.session_state.pendidikan_order = processed['pendidikan_order']
                st.session_state.karakteristik_columns = processed['karakteristik_columns']
            
            st.success("✅ Data berhasil diproses! Siap melanjutkan ke tahap selanjutnya.")
            
        except Exception as e:
            st.error(f"❌ Error membaca file: {e}")
    else:
        st.info("📌 Silakan upload file dataset untuk memulai analisis.")

# ==========================================================
# MENU: EDA
# ==========================================================
elif st.session_state.page == "EDA":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Jumlah Baris", f"{len(data):,}")
        with col2:
            st.metric("Kabupaten/Kota", data['nama_kabupaten_kota'].nunique())
        with col3:
            st.metric("Periode", f"{data['tahun'].min()} - {data['tahun'].max()}")
        with col4:
            st.metric("Kategori Pendidikan", data['pendidikan'].nunique())
        
        st.markdown("#### Statistik Deskriptif")
        st.dataframe(data['jumlah_pengangguran'].describe(), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Distribusi per Pendidikan")
            edu_dist = data.groupby('pendidikan')['jumlah_pengangguran'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5))
            edu_dist.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Total Pengangguran per Pendidikan')
            ax.set_xlabel('Pendidikan')
            ax.set_ylabel('Jumlah Pengangguran')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.markdown("#### Tren Pengangguran per Tahun")
            year_data = data.groupby('tahun')['jumlah_pengangguran'].sum()
            fig, ax = plt.subplots(figsize=(10, 5))
            year_data.plot(kind='line', marker='o', ax=ax, color='green')
            ax.set_title('Tren Total Pengangguran (2017-2025)')
            ax.set_xlabel('Tahun')
            ax.set_ylabel('Jumlah Pengangguran')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        st.markdown("#### Tren per Pendidikan")
        fig, ax = plt.subplots(figsize=(12, 6))
        for edu in data['pendidikan'].unique():
            edu_data = data[data['pendidikan'] == edu].groupby('tahun')['jumlah_pengangguran'].sum()
            ax.plot(edu_data.index, edu_data.values, marker='o', label=edu, linewidth=2)
        ax.set_title('Tren Pengangguran per Pendidikan (2017-2025)')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Jumlah Pengangguran')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
    else:
        st.warning("⚠️ Silakan upload dataset terlebih dahulu di menu 'Upload Dataset'.")

# ==========================================================
# MENU: PREPROCESSING
# ==========================================================
elif st.session_state.page == "Preprocessing":
    st.markdown('<div class="sub-header">⚙️ Preprocessing & Pembentukan Karakteristik</div>', unsafe_allow_html=True)
    
    if st.session_state.df_karakteristik is not None:
        
        st.markdown("""
            <div class="result-card">
                <h4>📌 Harmonisasi Kategori Pendidikan</h4>
                <p>Kategori pendidikan diharmonisasikan menjadi 4 kelompok:</p>
                <ul>
                    <li><strong>SD KE BAWAH</strong> - Tidak/Belum Sekolah, Tidak/Belum Tamat SD, SD</li>
                    <li><strong>SMP</strong> - Sekolah Menengah Pertama</li>
                    <li><strong>SMA/SLTA</strong> - SMA Umum dan SMA Kejuruan</li>
                    <li><strong>PERGURUAN TINGGI</strong> - Diploma dan Universitas</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### Data Karakteristik Perkembangan")
        st.markdown("""
            Setiap kabupaten/kota direpresentasikan oleh <strong>12 fitur</strong>:
            <br>3 karakteristik (Mean, Slope, CV) × 4 kategori pendidikan
        """, unsafe_allow_html=True)
        
        st.dataframe(st.session_state.df_karakteristik, use_container_width=True)
        
        st.markdown("#### Data Setelah Standardisasi (RobustScaler)")
        st.markdown("""
            <div style="background-color: #e8f8e8; padding: 0.5rem 1rem; border-radius: 5px; margin-bottom: 0.5rem;">
                ✅ Menggunakan <strong>RobustScaler</strong> - lebih tahan terhadap outlier
            </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(st.session_state.df_standardized, use_container_width=True)
        
    else:
        st.warning("⚠️ Silakan upload dan proses dataset terlebih dahulu.")

# ==========================================================
# MENU: CLUSTERING
# ==========================================================
elif st.session_state.page == "Clustering":
    st.markdown('<div class="sub-header">🎯 Clustering K-Medoids</div>', unsafe_allow_html=True)
    
    if st.session_state.df_standardized is not None:
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Konfigurasi Clustering")
            st.markdown("""
                <ul>
                    <li><strong>Metode:</strong> K-Medoids</li>
                    <li><strong>Distance Metric:</strong> Euclidean Distance</li>
                    <li><strong>Algoritma:</strong> PAM (Partitioning Around Medoids)</li>
                    <li><strong>Kandidat Cluster:</strong> k = 2 sampai 10</li>
                </ul>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### Jalankan Analisis")
            if st.button("🚀 Jalankan Clustering", use_container_width=True):
                with st.spinner("Sedang melakukan clustering dengan K-Medoids..."):
                    results = run_clustering(
                        st.session_state.df_standardized,
                        range(2, 11)
                    )
                    st.session_state.clustering_results = results
                    
                    # Tentukan cluster terbaik
                    best_k = max(results.keys(), key=lambda k: results[k]['silhouette'])
                    st.session_state.best_k = best_k
                    st.session_state.best_labels = results[best_k]['labels']
                    st.session_state.best_medoids = results[best_k]['medoid_labels']
                    
                    # Buat evaluation_df
                    evaluation_df = pd.DataFrame({
                        'Jumlah Cluster': list(results.keys()),
                        'Silhouette Score': [results[k]['silhouette'] for k in results.keys()],
                        'Davies-Bouldin': [results[k]['dbi'] for k in results.keys()],
                        'Calinski-Harabasz': [results[k]['ch'] for k in results.keys()],
                        'Inertia': [results[k]['inertia'] for k in results.keys()]
                    })
                    st.session_state.evaluation_df = evaluation_df
                    
                    # Hitung cluster means
                    df_karakteristik = st.session_state.df_karakteristik
                    df_karakteristik['Cluster'] = results[best_k]['labels']
                    cluster_means = df_karakteristik.groupby('Cluster')[
                        st.session_state.karakteristik_columns
                    ].mean()
                    st.session_state.cluster_means = cluster_means
                    
                st.success(f"✅ Clustering selesai! Cluster terbaik: k={best_k}")
        
        if st.session_state.clustering_results is not None:
            st.markdown("---")
            st.markdown("#### Hasil Clustering")
            
            results = st.session_state.clustering_results
            eval_df = st.session_state.evaluation_df
            st.dataframe(eval_df, use_container_width=True)
            
            # Best cluster info
            best_k = st.session_state.best_k
            st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid #27ae60;">
                    <h4>🏆 Cluster Terbaik: k = {best_k}</h4>
                    <p>
                        <strong>Silhouette Score:</strong> {results[best_k]['silhouette']:.4f} &nbsp;|&nbsp;
                        <strong>Davies-Bouldin:</strong> {results[best_k]['dbi']:.4f}
                    </p>
                    <p><strong>Medoid:</strong> {', '.join(results[best_k]['medoid_labels'])}</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Detail cluster
            st.markdown("#### Detail Anggota Cluster")
            labels = results[best_k]['labels']
            df_cluster = pd.DataFrame({
                'Kabupaten_Kota': st.session_state.df_standardized.index,
                'Cluster': labels
            })
            
            for c in range(best_k):
                members = df_cluster[df_cluster['Cluster'] == c]['Kabupaten_Kota'].tolist()
                medoid = results[best_k]['medoid_labels'][c] if c < len(results[best_k]['medoid_labels']) else '-'
                
                with st.expander(f"📁 Cluster {c} - {len(members)} anggota | Medoid: {medoid}"):
                    st.write(", ".join(members))
    
    else:
        st.warning("⚠️ Silakan upload dan proses dataset terlebih dahulu.")

# ==========================================================
# MENU: EVALUASI
# ==========================================================
elif st.session_state.page == "Evaluasi":
    st.markdown('<div class="sub-header">📈 Evaluasi & Hasil Analisis</div>', unsafe_allow_html=True)
    
    if st.session_state.clustering_results is not None:
        
        st.markdown("#### Visualisasi Evaluasi Clustering")
        
        eval_df = st.session_state.evaluation_df
        best_k = st.session_state.best_k
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Silhouette Score
        axes[0, 0].plot(eval_df['Jumlah Cluster'], eval_df['Silhouette Score'], 
                       marker='o', linewidth=2, color='blue', markersize=8)
        axes[0, 0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best: k={best_k}')
        axes[0, 0].set_xlabel('Jumlah Cluster (k)')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Davies-Bouldin
        axes[0, 1].plot(eval_df['Jumlah Cluster'], eval_df['Davies-Bouldin'], 
                       marker='o', linewidth=2, color='green', markersize=8)
        axes[0, 1].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best: k={best_k}')
        axes[0, 1].set_xlabel('Jumlah Cluster (k)')
        axes[0, 1].set_ylabel('Davies-Bouldin Index')
        axes[0, 1].set_title('Davies-Bouldin (semakin kecil semakin baik)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calinski-Harabasz
        axes[1, 0].plot(eval_df['Jumlah Cluster'], eval_df['Calinski-Harabasz'], 
                       marker='o', linewidth=2, color='orange', markersize=8)
        axes[1, 0].axvline(x=best_k, color='red', linestyle='--', linewidth=2, label=f'Best: k={best_k}')
        axes[1, 0].set_xlabel('Jumlah Cluster (k)')
        axes[1, 0].set_ylabel('Calinski-Harabasz Index')
        axes[1, 0].set_title('Calinski-Harabasz (semakin besar semakin baik)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Inertia (Elbow)
        axes[1, 1].plot(eval_df['Jumlah Cluster'], eval_df['Inertia'], 
                       marker='o', linewidth=2, color='purple', markersize=8)
        axes[1, 1].set_xlabel('Jumlah Cluster (k)')
        axes[1, 1].set_ylabel('Inertia')
        axes[1, 1].set_title('Elbow Method (Inertia)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # PCA Visualization
        st.markdown("#### Visualisasi Cluster dengan PCA")
        
        df_standardized = st.session_state.df_standardized
        labels = st.session_state.best_labels
        best_medoids = st.session_state.best_medoids
        
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(df_standardized.values)
        
        df_pca = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': labels,
            'Kabupaten_Kota': df_standardized.index
        })
        df_pca['Is_Medoid'] = df_pca['Kabupaten_Kota'].isin(best_medoids)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(
            df_pca['PC1'], 
            df_pca['PC2'], 
            c=df_pca['Cluster'], 
            cmap='viridis', 
            s=80, 
            alpha=0.7
        )
        
        # Tandai medoid
        medoid_pca = df_pca[df_pca['Is_Medoid']]
        ax.scatter(
            medoid_pca['PC1'], 
            medoid_pca['PC2'], 
            c='red', 
            s=200, 
            marker='D', 
            edgecolor='black', 
            linewidth=2,
            label='Medoid'
        )
        
        for idx, row in medoid_pca.iterrows():
            ax.annotate(
                row['Kabupaten_Kota'],
                (row['PC1'], row['PC2']),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=9,
                fontweight='bold'
            )
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax.set_title(f'Visualisasi Cluster K-Medoids (k={st.session_state.best_k})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(scatter, ax=ax, label='Cluster')
        st.pyplot(fig)
        plt.close()
        
        # Profil Cluster
        st.markdown("#### Profil Karakteristik per Cluster")
        
        cluster_means = st.session_state.cluster_means
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            cluster_means,
            annot=True,
            fmt='.2f',
            cmap='RdYlBu_r',
            center=0,
            cbar_kws={'label': 'Nilai Karakteristik'},
            ax=ax
        )
        ax.set_title(f'Profil Karakteristik per Cluster (k={st.session_state.best_k})')
        ax.set_xlabel('Karakteristik')
        ax.set_ylabel('Cluster')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    else:
        st.warning("⚠️ Silakan jalankan clustering terlebih dahulu.")

# ==========================================================
# MENU: UNDUH
# ==========================================================
elif st.session_state.page == "Unduh":
    st.markdown('<div class="sub-header">📥 Unduh Hasil Analisis</div>', unsafe_allow_html=True)
    
    if st.session_state.clustering_results is not None:
        
        df_standardized = st.session_state.df_standardized
        labels = st.session_state.best_labels
        best_medoids = st.session_state.best_medoids
        best_k = st.session_state.best_k
        
        df_hasil = pd.DataFrame({
            'Kabupaten_Kota': df_standardized.index,
            'Cluster': labels,
            'Is_Medoid': df_standardized.index.isin(best_medoids)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df_hasil.to_csv(index=False)
            st.download_button(
                label="📊 Unduh CSV",
                data=csv,
                file_name="hasil_clustering_kmedoids.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### Preview Hasil")
            st.dataframe(df_hasil, use_container_width=True)
            
            st.markdown("#### Ringkasan")
            st.write(f"**Jumlah Cluster:** {best_k}")
            st.write(f"**Medoid:** {', '.join(best_medoids)}")
            for c in range(best_k):
                count = (labels == c).sum()
                st.write(f"  - Cluster {c}: {count} anggota")
    
    else:
        st.warning("⚠️ Silakan jalankan clustering terlebih dahulu.")

# ==========================================================
# MENU: TENTANG
# ==========================================================
elif st.session_state.page == "Tentang":
    st.markdown('<div class="sub-header">ℹ️ Tentang Aplikasi</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px;">
            <h3>📊 Sistem Analisis Pengelompokan Kabupaten/Kota di Jawa Barat</h3>
            <p><strong>Versi:</strong> 1.0.0</p>
            <p><strong>Tahun:</strong> 2026</p>
            <hr>
            
            <h4>🎯 Tujuan</h4>
            <p>
                Mengelompokkan 27 kabupaten/kota di Provinsi Jawa Barat berdasarkan 
                karakteristik perkembangan jumlah pengangguran terbuka menurut tingkat pendidikan 
                menggunakan metode <strong>K-Medoids</strong>.
            </p>
            <hr>
            
            <h4>📚 Metodologi</h4>
            <ul>
                <li><strong>Data:</strong> Open Data Jawa Barat (2017-2025)</li>
                <li><strong>Karakteristik:</strong> Mean, Slope, Coefficient of Variation</li>
                <li><strong>Standardisasi:</strong> RobustScaler (tahan outlier)</li>
                <li><strong>Clustering:</strong> K-Medoids dengan Euclidean Distance</li>
                <li><strong>Evaluasi:</strong> Silhouette Score, Davies-Bouldin Index</li>
            </ul>
            <hr>
            
            <h4>👤 Identitas</h4>
            <ul>
                <li><strong>Nama:</strong> Rakha Rizky Mahendra</li>
                <li><strong>NPM:</strong> 21083010013</li>
                <li><strong>Program Studi:</strong> Sains Data</li>
                <li><strong>Universitas:</strong> UPN Veteran Jawa Timur</li>
            </ul>
            <hr>
            
            <h4>📦 Teknologi</h4>
            <ul>
                <li><strong>Framework:</strong> Streamlit</li>
                <li><strong>Machine Learning:</strong> Scikit-learn (KMedoids Manual)</li>
                <li><strong>Visualisasi:</strong> Matplotlib, Seaborn</li>
                <li><strong>Data Processing:</strong> Pandas, NumPy</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("""
    <div class="footer">
        © 2026 Rakha Rizky Mahendra - Sains Data UPN Veteran Jawa Timur
    </div>
""", unsafe_allow_html=True)
