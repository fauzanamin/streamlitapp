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
        
        # Set random state
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
            
            # Assign setiap titik ke medoid terdekat
            distances = cdist(X, X[medoid_indices], metric=self.metric)
            labels = np.argmin(distances, axis=1)
            
            # Hitung total cost saat ini
            current_cost = np.sum(np.min(distances, axis=1))
            
            # Coba swap medoid dengan non-medoid
            improved = False
            best_medoids = medoid_indices.copy()
            best_cost = current_cost
            
            # Sampling untuk efisiensi
            non_medoid_indices = [i for i in range(n_samples) if i not in medoid_indices]
            
            # Jika terlalu banyak, ambil sample
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
        
        # Hasil akhir
        self.medoid_indices_ = medoid_indices
        distances = cdist(X, X[medoid_indices], metric=self.metric)
        self.labels_ = np.argmin(distances, axis=1)
        self.inertia_ = np.sum(np.min(distances, axis=1))
        
        return self
    
    def fit_predict(self, X):
        """Fit dan return labels"""
        self.fit(X)
        return self.labels_
    
    def _heuristic_init(self, X, k):
        """Heuristic initialization untuk medoids (Kaufman & Rousseeuw)"""
        n_samples = X.shape[0]
        
        if k >= n_samples:
            return np.arange(k)
        
        medoid_indices = []
        
        # Pilih titik dengan jarak minimum terbesar
        if n_samples > 0:
            # Pilih titik pertama secara random
            first_idx = np.random.choice(n_samples)
            medoid_indices.append(first_idx)
            
            # Pilih titik-titik berikutnya dengan jarak maksimum dari medoid yang ada
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
# CUSTOM CSS
# ==========================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        padding: 0.8rem 0;
        border-bottom: 2px solid #ecf0f1;
        margin-bottom: 1rem;
    }
    .stat-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.08);
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
    </style>
""", unsafe_allow_html=True)

# ==========================================================
# SESSION STATE
# ==========================================================
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
if 'pendidikan_order' not in st.session_state:
    st.session_state.pendidikan_order = ['SD KE BAWAH', 'SMP', 'SMA/SLTA', 'PERGURUAN TINGGI']
if 'karakteristik_columns' not in st.session_state:
    st.session_state.karakteristik_columns = None

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
    <div class="main-header">
        📊 Analisis Pengelompokan Kabupaten/Kota di Jawa Barat<br>
        <small style="font-size: 1rem; color: #555;">
            Berdasarkan Karakteristik Perkembangan Jumlah Pengangguran Terbuka<br>
            Menggunakan Metode K-Medoids
        </small>
    </div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR MENU
# ==========================================================
with st.sidebar:
    st.markdown("### 📋 MENU NAVIGASI")
    
    menu_options = [
        "🏠 Beranda",
        "📤 Upload Dataset",
        "📊 Exploratory Data Analysis",
        "⚙️ Preprocessing & Karakteristik",
        "🎯 Clustering K-Medoids",
        "📈 Evaluasi & Hasil",
        "📥 Unduh Hasil",
        "ℹ️ Tentang"
    ]
    
    selected = st.radio(
        "Pilih Menu",
        menu_options,
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
        <div style="font-size: 0.8rem; color: #7f8c8d; text-align: center;">
            <strong>Rakha Rizky Mahendra</strong><br>
            NPM 21083010013<br>
            Sains Data - UPN Veteran Jatim
        </div>
    """, unsafe_allow_html=True)

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
# MENU-MENU APLIKASI
# ==========================================================

# [SISANYA SAMA SEPERTI SEBELUMNYA, MENU BERANDA, UPLOAD, EDA, dll]
# ... (lanjutkan dengan kode menu yang sama seperti sebelumnya)

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 1rem 0;">
        © 2026 Rakha Rizky Mahendra - Sains Data UPN Veteran Jawa Timur
    </div>
""", unsafe_allow_html=True)
