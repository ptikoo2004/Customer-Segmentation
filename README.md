# 👥 Customer Segmentation Dashboard

An interactive web application for customer segmentation using **RFM Analysis** and **Machine Learning Clustering**. Built with Streamlit and powered by scikit-learn.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)


## 🚀 Live Demo

[Insert your deployed app link here]

## 📊 Features

### 🎯 Core Functionality
- **RFM Analysis**: Automatic calculation of Recency, Frequency, and Monetary values
- **Smart Clustering**: K-means clustering with customizable cluster count
- **Interactive Visualizations**: 3D scatter plots, heatmaps, and distribution charts
- **Business Insights**: Actionable recommendations for each customer segment

### 📈 Dashboard Tabs
1. **📊 Overview**: Dataset statistics, data preview, and key metrics
2. **🔍 RFM Analysis**: Distribution plots and correlation analysis
3. **🎯 Clustering**: Customer segments with business strategies
4. **📈 Visualizations**: Interactive 3D plots and advanced charts
5. **💾 Export**: Download results and individual customer lookup

### ✨ Advanced Features
- **Default Dataset**: Built-in UCI Online Retail dataset
- **File Upload**: Support for CSV and Excel files
- **Data Validation**: Automatic data cleaning and preprocessing
- **Export Options**: Download segmentation results as CSV
- **Customer Lookup**: Search and analyze individual customers
- **Quality Metrics**: Silhouette score for clustering evaluation

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/customer-segmentation-dashboard.git
cd customer-segmentation-dashboard
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📁 Data Format

Your dataset should contain the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `CustomerID` | Unique customer identifier | Integer |
| `InvoiceDate` | Transaction date | DateTime |
| `Quantity` | Number of items purchased | Integer |
| `UnitPrice` | Price per item | Float |

### Sample Data Structure
```
CustomerID | InvoiceDate | Quantity | UnitPrice
12345      | 2023-01-15  | 2        | 15.99
12346      | 2023-01-16  | 1        | 25.50
```

## 🎯 Customer Segments

The application automatically identifies four key customer segments:

### 🏆 Champions
- **Characteristics**: Recent, frequent, high-value customers
- **Strategy**: Reward loyalty, ask for reviews, upsell premium products

### 💎 Loyal Customers  
- **Characteristics**: Regular buyers with good lifetime value
- **Strategy**: Offer loyalty programs, recommend new products

### ⚠️ At Risk
- **Characteristics**: Previously active customers who haven't purchased recently
- **Strategy**: Re-engagement campaigns, limited-time offers

### 😴 Lost Customers
- **Characteristics**: Inactive customers with low engagement
- **Strategy**: Win-back campaigns, surveys to understand churn

## 📊 RFM Metrics Explained

- **Recency (R)**: Days since last purchase
- **Frequency (F)**: Number of transactions
- **Monetary (M)**: Total amount spent

## 🔧 Technical Details

### Built With
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **Seaborn & Matplotlib**: Statistical plotting

### Machine Learning Approach
- **Algorithm**: K-means clustering
- **Preprocessing**: StandardScaler normalization
- **Evaluation**: Silhouette score for cluster quality
- **Optimization**: Configurable cluster count (2-8 clusters)

## 📷 Screenshots

### Dashboard Overview
![Dashboard Overview](screenshots/dashboard-overview.png)

### 3D Customer Segmentation
![3D Visualization](screenshots/3d-visualization.png)

### RFM Analysis
![RFM Analysis](screenshots/rfm-analysis.png)

## 🚀 Deployment

### Streamlit Cloud (Recommended)
1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy with one click!

### Local Development
```bash
streamlit run app.py --server.port 8501
```

## 📈 Usage Examples

### Using Default Dataset
1. Select "Use default dataset" in the sidebar
2. Click "Load Default Dataset"
3. Explore the different tabs to analyze customer segments

### Uploading Custom Data
1. Prepare your data in the required format
2. Select "Upload your own file" 
3. Upload your CSV or Excel file
4. Adjust clustering parameters if needed
5. Download results for further analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **Dataset**: [UCI Online Retail Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail)
- **Inspiration**: RFM analysis methodology in customer relationship management
- **Tools**: Built with amazing open-source libraries

---

⭐ **If you found this project helpful, please give it a star!** ⭐
