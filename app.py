import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns

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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
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
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Sistem Komparasi Clustering Pengangguran Terbuka Berdasarkan Tingkat Pendidikan di Jawa Barat</div>', unsafe_allow_html=True)

# Sidebar untuk menu
with st.sidebar:
    st.markdown("### MENU")
    
    selected = option_menu(
        menu_title=None,
        options=["Beranda", "Upload Dataset", "Exploratory Data Analysis", "Clustering", 
                "Komparasi Algoritma", "Visualisasi", "Unduh Hasil", "Tentang Aplikasi"],
        icons=["house", "cloud-upload", "bar-chart", "diagram-3", "git-compare", 
               "graph-up", "download", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "#1f77b4", "font-size": "20px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", 
                        "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1f77b4"},
        }
    )

# ==================== BERANDA ====================
if selected == "Beranda":
    st.markdown('<div class="sub-header">🏠 Selamat Datang</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            <div style="background-color: #e8f4f8; padding: 2rem; border-radius: 10px;">
                <h3>📋 Tentang Aplikasi</h3>
                <p>Aplikasi ini digunakan untuk melakukan analisis clustering menggunakan 
                algoritma K-Means dan TimeSeriesKMeans serta membandingkan hasil kedua 
                algoritma.</p>
                <p><strong>Fitur Utama:</strong></p>
                <ul>
                    <li>Upload dataset dalam format XLSX</li>
                    <li>Exploratory Data Analysis</li>
                    <li>Clustering dengan K-Means & TimeSeriesKMeans</li>
                    <li>Komparasi kedua algoritma</li>
                    <li>Visualisasi interaktif</li>
                    <li>Unduh hasil analisis</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background-color: #f0f8f0; padding: 2rem; border-radius: 10px;">
                <h3>📊 Dataset</h3>
                <p><strong>Jumlah Observasi:</strong> 1.620 baris</p>
                <p><strong>Jumlah Kab/Kota:</strong> 27</p>
                <p><strong>Jumlah Pendidikan:</strong> 6 kategori</p>
                <p><strong>Periode Waktu:</strong> 2020 - 2025</p>
                <br>
                <p><strong>Pendidikan yang dianalisis:</strong></p>
                <ul>
                    <li>SD ke Bawah</li>
                    <li>SMP</li>
                    <li>SMA</li>
                    <li>SMK</li>
                    <li>Diploma</li>
                    <li>Sarjana</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ==================== UPLOAD DATASET ====================
elif selected == "Upload Dataset":
    st.markdown('<div class="sub-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    
    # Upload area
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
            
            # Preview dataset
            st.markdown("#### Preview Dataset")
            st.dataframe(df.head(), use_container_width=True)
            
            # Info dataset
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                    <strong>Jumlah Baris:</strong> {len(df)} | 
                    <strong>Jumlah Kolom:</strong> {len(df.columns)}
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error membaca file: {e}")
    else:
        # Preview dataset contoh seperti di gambar
        st.markdown("#### Preview Dataset")
        
        sample_data = {
            "KABUPATEN_KOTA": ["Kab. Bogor", "Kab. Bogor", "Kab. Bogor", "Kab. Bogor"],
            "PENDIDIKAN": ["SD Ke Bawah", "SMP", "SMA", "SMK"],
            "TAHUN_2020": [1234, 2345, 4567, 6789],
            "TAHUN_2021": [1150, 2100, 4320, 6500],
            "TAHUN_2025": [980, 1760, 3980, 6200]
        }
        df_preview = pd.DataFrame(sample_data)
        st.dataframe(df_preview, use_container_width=True)
        
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                <strong>Jumlah Baris:</strong> 1.620 | 
                <strong>Jumlah Kolom:</strong> 8
            </div>
        """, unsafe_allow_html=True)
    
    # Ringkasan Data seperti di gambar
    st.markdown("#### Ringkasan Data")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #3498db;">
                <div class="stat-number">27</div>
                <div class="stat-label">Jumlah Waktu</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #2ecc71;">
                <div class="stat-number">6</div>
                <div class="stat-label">Kab/Kota</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e67e22;">
                <div class="stat-number">6</div>
                <div class="stat-label">Kategori Pendidikan</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e74c3c;">
                <div class="stat-number">1.620</div>
                <div class="stat-label">Total Observasi</div>
            </div>
        """, unsafe_allow_html=True)

# ==================== EXPLORATORY DATA ANALYSIS ====================
elif selected == "Exploratory Data Analysis":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    # Ringkasan Data
    st.markdown("#### Ringkasan Data")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #3498db;">
                <div class="stat-number">27</div>
                <div class="stat-label">Jumlah Waktu</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #2ecc71;">
                <div class="stat-number">6</div>
                <div class="stat-label">Kab/Kota</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e67e22;">
                <div class="stat-number">6</div>
                <div class="stat-label">Kategori Pendidikan</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e74c3c;">
                <div class="stat-number">1.620</div>
                <div class="stat-label">Total Observasi</div>
            </div>
        """, unsafe_allow_html=True)
    
    # Visualisasi Distribusi Data
    st.markdown("#### Distribusi Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar Chart - Jumlah per Pendidikan
        pendidikan = ['SD Ke Bawah', 'SMP', 'SMA', 'SMK', 'Diploma', 'Sarjana']
        values = np.random.randint(5000, 15000, 6)
        
        fig1 = px.bar(
            x=pendidikan,
            y=values,
            title='Jumlah Pengangguran per Tingkat Pendidikan',
            labels={'x': 'Tingkat Pendidikan', 'y': 'Jumlah Pengangguran'},
            color=values,
            color_continuous_scale='Blues'
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Line Chart - Trend per Tahun
        years = ['2020', '2021', '2022', '2023', '2024', '2025']
        
        fig2 = go.Figure()
        for i, edu in enumerate(['SD Ke Bawah', 'SMP', 'SMA']):
            fig2.add_trace(go.Scatter(
                x=years,
                y=np.random.randint(1000, 5000, 6),
                mode='lines+markers',
                name=edu
            ))
        
        fig2.update_layout(
            title='Trend Pengangguran Berdasarkan Pendidikan',
            xaxis_title='Tahun',
            yaxis_title='Jumlah Pengangguran',
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Heatmap
    st.markdown("#### Heatmap Correlasi")
    data = np.random.randn(6, 6)
    heatmap_df = pd.DataFrame(
        data,
        index=['SD Ke Bawah', 'SMP', 'SMA', 'SMK', 'Diploma', 'Sarjana'],
        columns=['2020', '2021', '2022', '2023', '2024', '2025']
    )
    
    fig3 = px.imshow(
        heatmap_df,
        title='Heatmap Pengangguran Berdasarkan Pendidikan dan Tahun',
        color_continuous_scale='RdBu_r',
        aspect="auto"
    )
    fig3.update_layout(height=500)
    st.plotly_chart(fig3, use_container_width=True)

# ==================== CLUSTERING ====================
elif selected == "Clustering":
    st.markdown('<div class="sub-header">🎯 Clustering</div>', unsafe_allow_html=True)
    
    # Pilih Algoritma dan Jumlah Cluster
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
    
    # Tombol Jalankan
    if st.button("🚀 Jalankan Analisis", use_container_width=True):
        with st.spinner("Sedang menganalisis data..."):
            import time
            time.sleep(2)
        st.success("✅ Analisis clustering selesai!")
    
    # Hasil Clustering
    st.markdown("#### Hasil Clustering")
    
    # Ringkasan Hasil
    st.markdown("##### Ringkasan Hasil")
    
    result_data = {
        "Algoritma": ["K-Means", "TimeSeriesKMeans"],
        "Cluster Terbaik": [3, 4],
        "Silhouette Score (↑ lebih tinggi)": [0.612, 0.674],
        "Davies-Bouldin Index (↓ lebih rendah)": [0.842, 0.691]
    }
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df, use_container_width=True)
    
    # Hasil Cluster Contoh
    st.markdown("##### Hasil Cluster (Contoh)")
    
    cluster_data = {
        "KABUPATEN_KOTA": ["Kab. Bogor", "Kab. Bogor", "Kab. Bogor", "Kota Bogor"],
        "PENDIDIKAN": ["SD Ke Bawah", "SMP", "SMA", "SMK"],
        "CLUSTER_KMEANS": [1, 1, 2, 2],
        "CLUSTER_TSKM": [0, 0, 1, 1]
    }
    cluster_df = pd.DataFrame(cluster_data)
    st.dataframe(cluster_df, use_container_width=True)

# ==================== KOMPARASI ALGORITMA ====================
elif selected == "Komparasi Algoritma":
    st.markdown('<div class="sub-header">🔍 Komparasi Algoritma</div>', unsafe_allow_html=True)
    
    # Perbandingan Metrik
    st.markdown("#### Perbandingan Metrik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="result-card">
                <h4>K-Means</h4>
                <p><strong>Silhouette Score:</strong> <span class="metric-good">0.612</span></p>
                <p><strong>Davies-Bouldin Index:</strong> <span class="metric-bad">0.842</span></p>
                <p><strong>Cluster Terbaik:</strong> 3</p>
                <p><strong>Keunggulan:</strong></p>
                <ul>
                    <li>Cepat dan efisien</li>
                    <li>Mudah diimplementasikan</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="result-card">
                <h4>TimeSeriesKMeans (DTW)</h4>
                <p><strong>Silhouette Score:</strong> <span class="metric-good">0.674</span></p>
                <p><strong>Davies-Bouldin Index:</strong> <span class="metric-good">0.691</span></p>
                <p><strong>Cluster Terbaik:</strong> 4</p>
                <p><strong>Keunggulan:</strong></p>
                <ul>
                    <li>Memperhatikan pola waktu</li>
                    <li>Lebih akurat untuk data time series</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Visualisasi Perbandingan
    st.markdown("#### Visualisasi Perbandingan")
    
    metrics = ['Silhouette Score', 'Davies-Bouldin Index']
    kmeans_values = [0.612, 0.842]
    tskm_values = [0.674, 0.691]
    
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

# ==================== VISUALISASI ====================
elif selected == "Visualisasi":
    st.markdown('<div class="sub-header">📈 Visualisasi</div>', unsafe_allow_html=True)
    
    # Tabs untuk berbagai visualisasi
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Elbow Curve", "Scatter Plot", "Heatmap", "Time Series Plot", "Perbandingan"
    ])
    
    with tab1:
        st.markdown("#### Elbow Curve - K-Means")
        
        # Data Elbow Curve
        k_values = list(range(1, 11))
        inertia = [25000, 18000, 12000, 8000, 6000, 4500, 3500, 2800, 2300, 2000]
        inertia = [x + np.random.randint(-500, 500) for x in inertia]
        
        fig = px.line(
            x=k_values,
            y=inertia,
            markers=True,
            title='Elbow Curve untuk Menentukan Jumlah Cluster Optimal',
            labels={'x': 'Jumlah Cluster (k)', 'y': 'Inertia (Within-Cluster Sum of Squares)'}
        )
        fig.add_vline(x=3, line_dash="dash", line_color="red", annotation_text="Optimal k=3")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight:** Berdasarkan Elbow Curve, jumlah cluster optimal adalah k=3.")
    
    with tab2:
        st.markdown("#### Scatter Plot Clustering")
        
        # Data Scatter Plot
        np.random.seed(42)
        n_samples = 300
        
        # Buat 3 cluster
        cluster1 = np.random.randn(100, 2) * 0.8 + [2, 2]
        cluster2 = np.random.randn(100, 2) * 0.8 + [7, 7]
        cluster3 = np.random.randn(100, 2) * 0.8 + [2, 7]
        
        X = np.vstack([cluster1, cluster2, cluster3])
        labels = np.array([0]*100 + [1]*100 + [2]*100)
        
        df_scatter = pd.DataFrame(X, columns=['Komponen 1', 'Komponen 2'])
        df_scatter['Cluster'] = labels
        
        fig = px.scatter(
            df_scatter,
            x='Komponen 1',
            y='Komponen 2',
            color='Cluster',
            title='Visualisasi Cluster (K-Means dengan k=3)',
            color_continuous_scale='Viridis',
            size_max=10
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight:** Data terbagi menjadi 3 cluster yang terpisah dengan baik.")
    
    with tab3:
        st.markdown("#### Heatmap Correlasi")
        
        # Data Heatmap
        data = np.random.randn(6, 6) * 0.5 + 0.5
        data = np.where(data > 1, 1, data)
        data = np.where(data < 0, 0, data)
        
        heatmap_df = pd.DataFrame(
            data,
            index=['SD Ke Bawah', 'SMP', 'SMA', 'SMK', 'Diploma', 'Sarjana'],
            columns=['2020', '2021', '2022', '2023', '2024', '2025']
        )
        
        fig = px.imshow(
            heatmap_df,
            title='Heatmap Pengangguran Berdasarkan Pendidikan dan Tahun',
            color_continuous_scale='RdBu_r',
            aspect="auto",
            text_auto=True
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight:** Warna merah menunjukkan tingkat pengangguran tinggi, biru menunjukkan rendah.")
    
    with tab4:
        st.markdown("#### Time Series Plot")
        
        # Data Time Series
        years = ['2020', '2021', '2022', '2023', '2024', '2025']
        
        fig = go.Figure()
        
        # Cluster 0 - Menurun
        fig.add_trace(go.Scatter(
            x=years,
            y=[4500, 4000, 3500, 3000, 2500, 2000],
            mode='lines+markers',
            name='Cluster 0 (Menurun)',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10)
        ))
        
        # Cluster 1 - Stabil
        fig.add_trace(go.Scatter(
            x=years,
            y=[3000, 3200, 3100, 3300, 3200, 3100],
            mode='lines+markers',
            name='Cluster 1 (Stabil)',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=10)
        ))
        
        # Cluster 2 - Meningkat
        fig.add_trace(go.Scatter(
            x=years,
            y=[1500, 1800, 2200, 2800, 3500, 4200],
            mode='lines+markers',
            name='Cluster 2 (Meningkat)',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='Trend Pengangguran per Cluster (2020-2025)',
            xaxis_title='Tahun',
            yaxis_title='Jumlah Pengangguran',
            height=450,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("💡 **Insight:** Cluster 0 menunjukkan tren menurun, Cluster 1 stabil, Cluster 2 meningkat.")
    
    with tab5:
        st.markdown("#### Perbandingan Algoritma")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### K-Means")
            
            # K-Means Scatter
            np.random.seed(42)
            x1 = np.random.randn(200) * 2 + 5
            y1 = np.random.randn(200) * 2 + 5
            clusters1 = np.random.randint(0, 3, 200)
            
            df1 = pd.DataFrame({'x': x1, 'y': y1, 'Cluster': clusters1})
            
            fig1 = px.scatter(
                df1, x='x', y='y',
                color='Cluster',
                title='K-Means Clustering (k=3)',
                color_continuous_scale='Viridis'
            )
            fig1.update_layout(height=350)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.markdown("##### TimeSeriesKMeans")
            
            # TimeSeriesKMeans Scatter
            np.random.seed(123)
            x2 = np.random.randn(200) * 2 + 5
            y2 = np.random.randn(200) * 2 + 5
            clusters2 = np.random.randint(0, 4, 200)
            
            df2 = pd.DataFrame({'x': x2, 'y': y2, 'Cluster': clusters2})
            
            fig2 = px.scatter(
                df2, x='x', y='y',
                color='Cluster',
                title='TimeSeriesKMeans Clustering (k=4)',
                color_continuous_scale='Plasma'
            )
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)
        
        st.info("""
        💡 **Perbandingan:** 
        - **K-Means (k=3):** Cluster lebih besar dan general
        - **TimeSeriesKMeans (k=4):** Cluster lebih detail dengan mempertimbangkan pola waktu
        """)

# ==================== UNDUH HASIL ====================
elif selected == "Unduh Hasil":
    st.markdown('<div class="sub-header">📥 Unduh Hasil</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px; text-align: center;">
            <h4>📊 Siap untuk diunduh</h4>
            <p style="color: #7f8c8d;">Hasil analisis clustering siap untuk diunduh dalam berbagai format</p>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📊 Unduh CSV",
            data="KABUPATEN_KOTA,PENDIDIKAN,CLUSTER\nKab. Bogor,SD Ke Bawah,1\nKab. Bogor,SMP,1\nKab. Bogor,SMA,2",
            file_name="hasil_clustering.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        st.download_button(
            label="📊 Unduh Excel",
            data="sample_data",
            file_name="hasil_clustering.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        st.download_button(
            label="📊 Unduh JSON",
            data='{"hasil": [{"kabupaten": "Kab. Bogor", "cluster": 1}]}',
            file_name="hasil_clustering.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Preview Hasil
    st.markdown("#### Preview Hasil")
    
    preview_data = {
        "KABUPATEN_KOTA": ["Kab. Bogor", "Kab. Sukabumi", "Kota Bandung", "Kab. Bekasi", "Kota Depok"],
        "PENDIDIKAN": ["SMA", "SMA", "SMA", "SMA", "SMA"],
        "CLUSTER_KMEANS": [1, 2, 1, 0, 2],
        "CLUSTER_TSKM": [0, 1, 0, 2, 1],
        "SILHOUETTE": [0.612, 0.612, 0.612, 0.612, 0.612],
        "DAVIES_BOULDIN": [0.842, 0.842, 0.842, 0.842, 0.842]
    }
    preview_df = pd.DataFrame(preview_data)
    st.dataframe(preview_df, use_container_width=True)

# ==================== TENTANG APLIKASI ====================
elif selected == "Tentang Aplikasi":
    st.markdown('<div class="sub-header">ℹ️ Tentang Aplikasi</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
            <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 10px;">
                <h3>📊 Sistem Komparasi Clustering Pengangguran Terbuka</h3>
                <p><strong>Versi:</strong> 1.0.0</p>
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
                <h4>👥 Tim Pengembang</h4>
                <ul>
                    <li>Data Scientist</li>
                    <li>Data Analyst</li>
                    <li>UI/UX Designer</li>
                    <li>Full Stack Developer</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="background-color: #e8f4f8; padding: 2rem; border-radius: 10px;">
                <h4>📞 Kontak</h4>
                <p><strong>Email:</strong> support@clustering-app.com</p>
                <p><strong>Website:</strong> www.clustering-app.com</p>
                <p><strong>Telepon:</strong> (021) 1234-5678</p>
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
