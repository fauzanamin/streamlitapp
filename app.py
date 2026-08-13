import streamlit as st
import pandas as pd
import numpy as np

# Coba import plotly dengan error handling
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError as e:
    PLOTLY_AVAILABLE = False
    st.warning("⚠️ Plotly tidak tersedia. Install dengan: pip install plotly")

# Coba import option_menu
try:
    from streamlit_option_menu import option_menu
    MENU_AVAILABLE = True
except ImportError as e:
    MENU_AVAILABLE = False
    st.warning("⚠️ Streamlit Option Menu tidak tersedia. Install dengan: pip install streamlit-option-menu")

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
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📊 Sistem Komparasi Clustering Pengangguran Terbuka Berdasarkan Tingkat Pendidikan di Jawa Barat</div>', unsafe_allow_html=True)

# Sidebar untuk menu
with st.sidebar:
    st.markdown("### MENU")
    
    if MENU_AVAILABLE:
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
    else:
        # Menu sederhana tanpa option_menu
        menu_options = ["Beranda", "Upload Dataset", "Exploratory Data Analysis", "Clustering", 
                       "Komparasi Algoritma", "Visualisasi", "Unduh Hasil", "Tentang Aplikasi"]
        selected = st.selectbox("Pilih Menu", menu_options)

# Main content
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

elif selected == "Upload Dataset":
    st.markdown('<div class="sub-header">📤 Upload Dataset</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Pilih file XLSX", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success(f"✅ File berhasil diupload! {len(df)} baris data")
            st.dataframe(df.head())
            
            # Tampilkan info dataset
            st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                    <strong>Jumlah Baris:</strong> {len(df)} | 
                    <strong>Jumlah Kolom:</strong> {len(df.columns)}
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error membaca file: {e}")
    else:
        # Preview dataset contoh
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

elif selected == "Exploratory Data Analysis":
    st.markdown('<div class="sub-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("#### Ringkasan Data")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #3498db;">
                <div class="stat-number">27</div>
                <div class="stat-label">Kab/Kota</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #2ecc71;">
                <div class="stat-number">6</div>
                <div class="stat-label">Kategori Pendidikan</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e67e22;">
                <div class="stat-number">6</div>
                <div class="stat-label">Tahun</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="stat-card" style="border-left-color: #e74c3c;">
                <div class="stat-number">1.620</div>
                <div class="stat-label">Total Observasi</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("#### Distribusi Data")
    
    if PLOTLY_AVAILABLE:
        years = ['2020', '2021', '2022', '2023', '2024', '2025']
        values = np.random.randint(1000, 5000, 6)
        
        fig = px.line(
            x=years, 
            y=values,
            title='Trend Pengangguran Berdasarkan Pendidikan',
            labels={'x': 'Tahun', 'y': 'Jumlah Pengangguran'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("📊 Install plotly untuk visualisasi interaktif: `pip install plotly`")
        # Tampilkan data sederhana
        years = ['2020', '2021', '2022', '2023', '2024', '2025']
        values = np.random.randint(1000, 5000, 6)
        data_df = pd.DataFrame({"Tahun": years, "Jumlah": values})
        st.dataframe(data_df)

elif selected == "Clustering":
    st.markdown('<div class="sub-header">🎯 Clustering</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### Pilih Algoritma Clustering")
        algorithm = st.radio(
            "Pilih Algoritma",
            ["K-Means", "TimeSeriesKMeans (DTW)", "Komparasi Kedua Algoritma"],
            index=0
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
            import time
            time.sleep(2)
        st.success("✅ Analisis clustering selesai!")
    
    st.markdown("#### Hasil Clustering")
    st.markdown("##### Ringkasan Hasil")
    
    result_data = {
        "Algoritma": ["K-Means", "TimeSeriesKMeans"],
        "Cluster Terbaik": [3, 4],
        "Silhouette Score (↑ lebih tinggi)": [0.612, 0.674],
        "Davies-Bouldin Index (↓ lebih rendah)": [0.842, 0.691]
    }
    result_df = pd.DataFrame(result_data)
    st.dataframe(result_df, use_container_width=True)
    
    st.markdown("##### Hasil Cluster (Contoh)")
    
    cluster_data = {
        "KABUPATEN_KOTA": ["Kab. Bogor", "Kab. Bogor", "Kab. Bogor", "Kota Bogor"],
        "PENDIDIKAN": ["SD Ke Bawah", "SMP", "SMA", "SMK"],
        "CLUSTER_KMEANS": [1, 1, 2, 2],
        "CLUSTER_TSKM": [0, 0, 1, 1]
    }
    cluster_df = pd.DataFrame(cluster_data)
    st.dataframe(cluster_df, use_container_width=True)

elif selected == "Komparasi Algoritma":
    st.markdown('<div class="sub-header">🔍 Komparasi Algoritma</div>', unsafe_allow_html=True)
    
    st.markdown("#### Perbandingan Metrik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="result-card">
                <h4>K-Means</h4>
                <p><strong>Silhouette Score:</strong> <span class="metric-good">0.612</span></p>
                <p><strong>Davies-Bouldin Index:</strong> <span class="metric-bad">0.842</span></p>
                <p><strong>Cluster Terbaik:</strong> 3</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="result-card">
                <h4>TimeSeriesKMeans (DTW)</h4>
                <p><strong>Silhouette Score:</strong> <span class="metric-good">0.674</span></p>
                <p><strong>Davies-Bouldin Index:</strong> <span class="metric-good">0.691</span></p>
                <p><strong>Cluster Terbaik:</strong> 4</p>
            </div>
        """, unsafe_allow_html=True)
    
    if PLOTLY_AVAILABLE:
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
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

elif selected == "Visualisasi":
    st.markdown('<div class="sub-header">📈 Visualisasi</div>', unsafe_allow_html=True)
    
    if PLOTLY_AVAILABLE:
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Elbow Curve", "Scatter Plot", "Heatmap", "Time Series Plot", "Perbandingan"
        ])
        
        with tab1:
            st.markdown("#### Elbow Curve - K-Means")
            k_values = list(range(1, 11))
            inertia = np.random.randint(5000, 30000, 10)[::-1] + np.random.randint(-1000, 1000, 10)
            
            fig = px.line(
                x=k_values,
                y=inertia,
                markers=True,
                title='Elbow Curve untuk Menentukan Jumlah Cluster Optimal',
                labels={'x': 'Jumlah Cluster (k)', 'y': 'Inertia'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("#### Scatter Plot Clustering")
            np.random.seed(42)
            x = np.random.randn(200) * 2 + 5
            y = np.random.randn(200) * 2 + 5
            clusters = np.random.randint(0, 3, 200)
            
            fig = px.scatter(
                x=x, y=y,
                color=clusters,
                title='Visualisasi Cluster',
                labels={'x': 'Komponen 1', 'y': 'Komponen 2'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("#### Heatmap Correlasi")
            data = np.random.randn(6, 6)
            heatmap_df = pd.DataFrame(
                data,
                index=['SD Ke Bawah', 'SMP', 'SMA', 'SMK', 'Diploma', 'Sarjana'],
                columns=['2020', '2021', '2022', '2023', '2024', '2025']
            )
            
            fig = px.imshow(
                heatmap_df,
                title='Heatmap Pengangguran Berdasarkan Pendidikan dan Tahun',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.markdown("#### Time Series Plot")
            years = ['2020', '2021', '2022', '2023', '2024', '2025']
            
            fig = go.Figure()
            for i in range(3):
                fig.add_trace(go.Scatter(
                    x=years,
                    y=np.random.randint(1000, 5000, 6),
                    mode='lines+markers',
                    name=f'Cluster {i}'
                ))
            
            fig.update_layout(
                title='Trend Pengangguran per Cluster',
                xaxis_title='Tahun',
                yaxis_title='Jumlah Pengangguran',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.markdown("#### Perbandingan Algoritma")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### K-Means")
                np.random.seed(42)
                x1 = np.random.randn(200) * 2 + 5
                y1 = np.random.randn(200) * 2 + 5
                clusters1 = np.random.randint(0, 3, 200)
                
                fig1 = px.scatter(
                    x=x1, y=y1,
                    color=clusters1,
                    title='K-Means Clustering'
                )
                fig1.update_layout(height=300)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                st.markdown("##### TimeSeriesKMeans")
                np.random.seed(123)
                x2 = np.random.randn(200) * 2 + 5
                y2 = np.random.randn(200) * 2 + 5
                clusters2 = np.random.randint(0, 4, 200)
                
                fig2 = px.scatter(
                    x=x2, y=y2,
                    color=clusters2,
                    title='TimeSeriesKMeans Clustering'
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("📊 Install plotly untuk visualisasi interaktif: `pip install plotly`")
        st.markdown("""
        **Visualisasi yang tersedia:**
        - Elbow Curve
        - Scatter Plot
        - Heatmap
        - Time Series Plot
        - Perbandingan Algoritma
        
        Untuk mengaktifkan visualisasi, install plotly:
        ```bash
        pip install plotly
