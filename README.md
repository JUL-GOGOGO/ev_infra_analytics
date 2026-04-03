# ⚡ EV Charging Infrastructure Analytics Dashboard

North American EV charging infrastructure startup's data-driven decision dashboard built with Streamlit.

## 🌟 Key Features
- **B2B Analytics**: 10+ KPIs for infrastructure management and gap scoring.
- **Peak Load Simulator**: Operative strategy for 20% peak demand dispersion.
- **User Segmentation**: PCA + KMeans clustering for behavior-based persona profiling.
- **Churn Signal Detection**: Real-time interval & energy usage monitoring for retention.
- **EDA Themes**: Spatial, Temporal, Vehicle, Environment, and Cost analysis.

## 🛠️ Tech Stack
- **Dashboard**: Streamlit, Plotly, Folium
- **Data Engineering**: Pandas, PyArrow
- **ML/Analytics**: Scikit-learn, Statsmodels

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- [uv](https://github.com/astral-sh/uv) (Recommended for package management)

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_GITHUB_ID/ev-charging-infra-dashboard.git
cd ev-charging-infra-dashboard

# Install dependencies using uv
uv pip install -r requirements.txt
```

### Running the Dashboard
```bash
streamlit run src/app.py
```

## 📂 Project Structure
```text
ev_infra_analytics/
├── src/
│   ├── app.py             # Main Streamlit Application
│   ├── preprocess.py      # Data Preprocessing & Churn Logic
│   └── user_clustering.py # ML Clustering Pipeline
├── data/                  # Sample Datasets (CSV, Parquet)
├── images/                # Screenshots & Visualization Assets
├── reports/               # Analysis Reports
└── requirements.txt       # Project Dependencies
```

## 📝 License
MIT License
