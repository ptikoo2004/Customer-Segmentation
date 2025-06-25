import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .cluster-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file=None, use_default=False):
    """Load and preprocess the retail data"""
    try:
        if use_default:
            # Load default dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
            df = pd.read_excel(url)
            st.success("✅ Default dataset loaded successfully!")
        elif uploaded_file is not None:
            # Load uploaded file
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.error("Please upload a CSV or Excel file")
                return None
            st.success("✅ File uploaded successfully!")
        else:
            return None
            
        # Data cleaning
        original_shape = df.shape
        
        # Remove missing CustomerIDs
        df = df.dropna(subset=['CustomerID'])
        
        # Remove negative quantities and prices
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
        
        # Calculate total price
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        cleaned_shape = df.shape
        
        st.info(f"📊 Data cleaned: {original_shape[0]:,} → {cleaned_shape[0]:,} rows ({original_shape[0] - cleaned_shape[0]:,} removed)")
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_data
def calculate_rfm(df):
    """Calculate RFM metrics"""
    # Set snapshot date (day after last transaction)
    snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    # Calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency', 
        'TotalPrice': 'Monetary'
    })
    
    return rfm

def perform_clustering(rfm, n_clusters=4):
    """Perform K-means clustering on RFM data"""
    # Standardize the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    # Add cluster labels
    rfm_clustered = rfm.copy()
    rfm_clustered['Cluster'] = clusters
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(rfm_scaled, clusters)
    
    return rfm_clustered, kmeans, scaler, silhouette_avg

def get_cluster_insights(rfm_clustered):
    """Generate insights for each cluster"""
    cluster_summary = rfm_clustered.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean', 
        'Monetary': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Count'})
    
    # Define cluster names based on RFM characteristics
    cluster_names = {}
    cluster_descriptions = {}
    cluster_strategies = {}
    
    for cluster in cluster_summary.index:
        recency = cluster_summary.loc[cluster, 'Recency']
        frequency = cluster_summary.loc[cluster, 'Frequency']
        monetary = cluster_summary.loc[cluster, 'Monetary']
        
        if recency < 50 and frequency > 5 and monetary > 500:
            cluster_names[cluster] = "🏆 Champions"
            cluster_descriptions[cluster] = "Best customers: recent, frequent, high-value"
            cluster_strategies[cluster] = "Reward them, ask for reviews, upsell premium products"
        elif recency < 100 and frequency > 3 and monetary > 200:
            cluster_names[cluster] = "💎 Loyal Customers"
            cluster_descriptions[cluster] = "Regular buyers with good value"
            cluster_strategies[cluster] = "Offer loyalty programs, recommend new products"
        elif recency > 100 and frequency > 2:
            cluster_names[cluster] = "⚠️ At Risk"
            cluster_descriptions[cluster] = "Haven't purchased recently but were active"
            cluster_strategies[cluster] = "Send re-engagement campaigns, limited-time offers"
        else:
            cluster_names[cluster] = "😴 Lost Customers"
            cluster_descriptions[cluster] = "Inactive, low engagement customers"
            cluster_strategies[cluster] = "Win-back campaigns, surveys to understand why they left"
    
    return cluster_summary, cluster_names, cluster_descriptions, cluster_strategies

def main():
    # Header
    st.markdown('<h1 class="main-header">👥 Customer Segmentation Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze your customer data using RFM (Recency, Frequency, Monetary) analysis and machine learning clustering**")
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Data loading options
    st.sidebar.subheader("📁 Data Source")
    data_option = st.sidebar.radio(
        "Choose data source:",
        ["Upload your own file", "Use default dataset"]
    )
    
    uploaded_file = None
    if data_option == "Upload your own file":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx'],
            help="File should contain: CustomerID, InvoiceDate, Quantity, UnitPrice"
        )
    
    # Load data
    df = None
    if data_option == "Use default dataset":
        if st.sidebar.button("🚀 Load Default Dataset"):
            df = load_data(use_default=True)
    elif uploaded_file is not None:
        df = load_data(uploaded_file)
    
    if df is not None:
        # Sidebar controls for analysis
        st.sidebar.subheader("⚙️ Analysis Settings")
        n_clusters = st.sidebar.slider("Number of clusters", 2, 8, 4)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 RFM Analysis", "🎯 Clustering", "📈 Visualizations", "💾 Export"])
        
        # Calculate RFM
        rfm = calculate_rfm(df)
        rfm_clustered, kmeans_model, scaler, silhouette_score_val = perform_clustering(rfm, n_clusters)
        cluster_summary, cluster_names, cluster_descriptions, cluster_strategies = get_cluster_insights(rfm_clustered)
        
        with tab1:
            st.subheader("📊 Dataset Overview")
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
            with col2:
                st.metric("Total Transactions", f"{df.shape[0]:,}")
            with col3:
                st.metric("Date Range", f"{(df['InvoiceDate'].max() - df['InvoiceDate'].min()).days} days")
            with col4:
                st.metric("Total Revenue", f"${df['TotalPrice'].sum():,.0f}")
            
            # Data preview
            st.subheader("🔍 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Basic statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Transaction Statistics")
                stats_df = df[['Quantity', 'UnitPrice', 'TotalPrice']].describe()
                st.dataframe(stats_df, use_container_width=True)
            
            with col2:
                st.subheader("🌍 Top Countries")
                if 'Country' in df.columns:
                    country_stats = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
                    st.dataframe(country_stats, use_container_width=True)
        
        with tab2:
            st.subheader("🔍 RFM Analysis")
            
            # RFM summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Avg Recency", f"{rfm['Recency'].mean():.1f} days")
            with col2:
                st.metric("Avg Frequency", f"{rfm['Frequency'].mean():.1f} orders")
            with col3:
                st.metric("Avg Monetary", f"${rfm['Monetary'].mean():.0f}")
            
            # RFM distributions
            st.subheader("📊 RFM Distributions")
            
            fig = make_subplots(rows=1, cols=3, 
                              subplot_titles=('Recency Distribution', 'Frequency Distribution', 'Monetary Distribution'))
            
            fig.add_trace(go.Histogram(x=rfm['Recency'], name='Recency', nbinsx=30), row=1, col=1)
            fig.add_trace(go.Histogram(x=rfm['Frequency'], name='Frequency', nbinsx=30), row=1, col=2)
            fig.add_trace(go.Histogram(x=rfm['Monetary'], name='Monetary', nbinsx=30), row=1, col=3)
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # RFM correlation heatmap
            st.subheader("🔥 RFM Correlation Matrix")
            correlation_matrix = rfm[['Recency', 'Frequency', 'Monetary']].corr()
            
            fig = px.imshow(correlation_matrix, 
                          text_auto=True, 
                          aspect="auto",
                          color_continuous_scale='RdBu',
                          title="RFM Metrics Correlation")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🎯 Customer Segmentation Results")
            
            # Clustering metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Clusters", n_clusters)
            with col2:
                st.metric("Silhouette Score", f"{silhouette_score_val:.3f}")
            with col3:
                quality = "Excellent" if silhouette_score_val > 0.5 else "Good" if silhouette_score_val > 0.3 else "Fair"
                st.metric("Clustering Quality", quality)
            
            # Cluster summaries
            st.subheader("📋 Cluster Insights")
            
            for cluster in sorted(cluster_summary.index):
                with st.expander(f"{cluster_names.get(cluster, f'Cluster {cluster}')} - {cluster_summary.loc[cluster, 'Count']} customers"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Characteristics:**")
                        st.write(cluster_descriptions.get(cluster, "No description available"))
                        st.write(f"• Avg Recency: {cluster_summary.loc[cluster, 'Recency']:.1f} days")
                        st.write(f"• Avg Frequency: {cluster_summary.loc[cluster, 'Frequency']:.1f} orders")  
                        st.write(f"• Avg Monetary: ${cluster_summary.loc[cluster, 'Monetary']:.0f}")
                    
                    with col2:
                        st.write("**Recommended Strategy:**")
                        st.write(cluster_strategies.get(cluster, "No strategy defined"))
            
            # Cluster size pie chart
            st.subheader("🥧 Customer Distribution")
            cluster_counts = rfm_clustered['Cluster'].value_counts().sort_index()
            labels = [cluster_names.get(i, f'Cluster {i}') for i in cluster_counts.index]
            
            fig = px.pie(values=cluster_counts.values, 
                        names=labels,
                        title="Customer Distribution Across Segments")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("📈 Advanced Visualizations")
            
            # 3D Scatter plot
            st.subheader("🌐 3D Customer Segmentation")
            fig = px.scatter_3d(rfm_clustered.reset_index(), 
                              x='Recency', y='Frequency', z='Monetary',
                              color='Cluster',
                              title="3D Customer Segments",
                              labels={'Cluster': 'Customer Segment'},
                              opacity=0.7)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            # 2D scatter plots
            st.subheader("📊 2D Relationship Views")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(rfm_clustered.reset_index(), 
                               x='Recency', y='Monetary', 
                               color='Cluster',
                               title="Recency vs Monetary Value")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(rfm_clustered.reset_index(), 
                               x='Frequency', y='Monetary', 
                               color='Cluster',
                               title="Frequency vs Monetary Value")
                st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics heatmap
            st.subheader("🔥 Cluster Characteristics Heatmap")
            fig = px.imshow(cluster_summary[['Recency', 'Frequency', 'Monetary']].T,
                          text_auto=True,
                          aspect="auto",
                          color_continuous_scale='Viridis',
                          title="Average RFM Values by Cluster")
            fig.update_xaxes(ticktext=[cluster_names.get(i, f'Cluster {i}') for i in cluster_summary.index],
                           tickvals=list(range(len(cluster_summary.index))))
            st.plotly_chart(fig, use_container_width=True)
        
        with tab5:
            st.subheader("💾 Export Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**📋 Cluster Summary**")
                cluster_export = cluster_summary.copy()
                cluster_export['Cluster_Name'] = [cluster_names.get(i, f'Cluster {i}') for i in cluster_export.index]
                cluster_export['Description'] = [cluster_descriptions.get(i, '') for i in cluster_export.index]
                cluster_export['Strategy'] = [cluster_strategies.get(i, '') for i in cluster_export.index]
                st.dataframe(cluster_export, use_container_width=True)
                
                # Download cluster summary
                csv_cluster = cluster_export.to_csv(index=True)
                st.download_button(
                    label="📥 Download Cluster Summary",
                    data=csv_cluster,
                    file_name=f"cluster_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.write("**👥 Customer Data with Segments**")
                customer_export = rfm_clustered.copy()
                customer_export['Cluster_Name'] = customer_export['Cluster'].map(cluster_names)
                st.dataframe(customer_export.head(), use_container_width=True)
                
                # Download full customer data
                csv_customers = customer_export.to_csv(index=True)
                st.download_button(
                    label="📥 Download Customer Segments",
                    data=csv_customers,
                    file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            # Individual customer lookup
            st.subheader("🔍 Customer Lookup")
            customer_id = st.number_input("Enter Customer ID:", min_value=0, step=1)
            
            if customer_id > 0 and customer_id in rfm_clustered.index:
                customer_data = rfm_clustered.loc[customer_id]
                cluster_id = customer_data['Cluster']
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Customer Segment", cluster_names.get(cluster_id, f'Cluster {cluster_id}'))
                with col2:
                    st.metric("Recency", f"{customer_data['Recency']} days")
                with col3:
                    st.metric("Frequency", f"{customer_data['Frequency']} orders")
                with col4:
                    st.metric("Monetary", f"${customer_data['Monetary']:.0f}")
                
                st.info(f"**Strategy:** {cluster_strategies.get(cluster_id, 'No strategy defined')}")
    
    else:
        # Landing page when no data is loaded
        st.markdown("""
        ## 🚀 Welcome to Customer Segmentation Dashboard
        
        This powerful tool helps you understand your customers through **RFM Analysis** and **Machine Learning Clustering**.
        
        ### 📋 Required Data Format:
        Your data should contain these columns:
        - **CustomerID**: Unique customer identifier
        - **InvoiceDate**: Transaction date
        - **Quantity**: Number of items purchased  
        - **UnitPrice**: Price per item
        
        ### 🎯 What You'll Get:
        - **Customer Segments**: Automatic clustering of customers
        - **RFM Analysis**: Recency, Frequency, Monetary insights
        - **3D Visualizations**: Interactive charts and plots
        - **Business Strategies**: Actionable recommendations for each segment
        - **Export Options**: Download results for further analysis
        
        ### 📁 Get Started:
        1. Choose "Use default dataset" to try with sample data
        2. Or upload your own CSV/Excel file
        3. Explore the analysis tabs to discover insights!
        """)
        
        # Demo section
        if st.button("🎬 See Demo with Default Dataset"):
            st.balloons()
            st.success("Click 'Load Default Dataset' in the sidebar to get started!")

if __name__ == "__main__":
    main()