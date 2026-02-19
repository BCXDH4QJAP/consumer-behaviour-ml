import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, silhouette_score
import datetime
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────
st.set_page_config(page_title="Consumer Behaviour Analysis", 
                   layout="wide", page_icon="🛒")

st.title("🛒 Consumer Behaviour Analysis")
st.markdown("### Final Year Project — Machine Learning")
st.markdown("---")

# ─── File Upload ───────────────────────────────────────────
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is None:
    st.info("👈 Please upload your dataset from the sidebar to get started.")
    st.markdown("""
    ### What this app does:
    - ✅ Cleans and explores your retail data
    - ✅ Performs RFM Analysis
    - ✅ Segments customers using KMeans Clustering
    - ✅ Predicts customer loyalty using Random Forest
    """)
    st.stop()

# ─── Load Data ─────────────────────────────────────────────
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file, encoding='unicode_escape')
    else:
        df = pd.read_excel(file, sheet_name='Year 2010-2011')
    return df

df = load_data(uploaded_file)
st.success(f"✅ Dataset loaded! Shape: {df.shape}")

# ─── Raw Data Preview ──────────────────────────────────────
st.subheader("📋 Raw Data Preview")
st.dataframe(df.head(10))

# ─── Data Cleaning ─────────────────────────────────────────
st.markdown("---")
st.subheader("🧹 Data Cleaning")

df.columns = df.columns.str.strip()

df.dropna(subset=['CustomerID'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df['TotalPrice']   = df['Quantity'] * df['UnitPrice']
df['InvoiceDate']  = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')
df['Month']        = df['InvoiceDate'].dt.month
df['DayOfWeek']    = df['InvoiceDate'].dt.dayofweek

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Rows",      f"{df.shape[0]:,}")
col2.metric("Total Customers", f"{df['CustomerID'].nunique():,}")
col3.metric("Total Orders",    f"{df['InvoiceNo'].nunique():,}")
col4.metric("Total Revenue",   f"£{df['TotalPrice'].sum():,.0f}")

# ─── EDA ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Sales by Month**")
    monthly = df.groupby('Month')['TotalPrice'].sum()
    fig, ax = plt.subplots(figsize=(6,3))
    monthly.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    ax.set_title("Monthly Revenue")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**Sales by Day of Week**")
    day_names = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    daily = df.groupby('DayOfWeek')['TotalPrice'].sum().reset_index()
    daily['DayName'] = daily['DayOfWeek'].map(day_names)
    daily = daily.set_index('DayName')['TotalPrice']
    fig, ax = plt.subplots(figsize=(6,3))
    daily.plot(kind='bar', ax=ax, color='coral')
    ax.set_title("Revenue by Day of Week")
    plt.tight_layout()
    st.pyplot(fig)

col3, col4 = st.columns(2)

with col3:
    st.markdown("**Top 10 Countries**")
    top_countries = df.groupby('Country')['TotalPrice'].sum()\
                      .sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6,4))
    top_countries.plot(kind='barh', ax=ax, color='green')
    ax.set_title("Top 10 Countries by Revenue")
    plt.tight_layout()
    st.pyplot(fig)

with col4:
    st.markdown("**Top 10 Products**")
    top_products = df.groupby('Description')['Quantity'].sum()\
                     .sort_values(ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(6,4))
    top_products.plot(kind='barh', ax=ax, color='purple')
    ax.set_title("Top 10 Products by Quantity")
    plt.tight_layout()
    st.pyplot(fig)

# ─── RFM Analysis ──────────────────────────────────────────
st.markdown("---")
st.subheader("📐 RFM Analysis")

snapshot_date = df['InvoiceDate'].max() + datetime.timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate' : lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo'   : 'nunique',
    'TotalPrice'  : 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm[rfm['Monetary'] > 0]

st.write("RFM Table (first 10 rows):")
st.dataframe(rfm.head(10))

col1, col2, col3 = st.columns(3)
with col1:
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(rfm['Recency'], bins=30, color='steelblue')
    ax.set_title("Recency Distribution")
    st.pyplot(fig)
with col2:
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(rfm['Frequency'], bins=30, color='coral')
    ax.set_title("Frequency Distribution")
    st.pyplot(fig)
with col3:
    fig, ax = plt.subplots(figsize=(5,3))
    ax.hist(rfm['Monetary'], bins=30, color='green')
    ax.set_title("Monetary Distribution")
    st.pyplot(fig)

# ─── Clustering ────────────────────────────────────────────
st.markdown("---")
st.subheader("🔵 Customer Segmentation (KMeans)")

scaler     = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Elbow
inertias = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertias.append(km.inertia_)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(range(2,10), inertias, marker='o', color='steelblue')
    ax.set_title("Elbow Method — Optimal K")
    ax.set_xlabel("Number of Clusters")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

# Final clustering with K=4
km = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = km.fit_predict(rfm_scaled)

score = silhouette_score(rfm_scaled, rfm['Cluster'])
st.success(f"✅ Silhouette Score: {score:.4f}  (closer to 1 is better)")

segment_map = {0:'Champions', 1:'At Risk', 2:'New Customers', 3:'Hibernating'}
rfm['Segment'] = rfm['Cluster'].map(segment_map)

with col2:
    fig, ax = plt.subplots(figsize=(6,4))
    rfm['Segment'].value_counts().plot(
        kind='pie', autopct='%1.1f%%', ax=ax,
        colors=['#2ecc71','#e74c3c','#3498db','#f39c12'])
    ax.set_title("Customer Segments")
    ax.set_ylabel("")
    st.pyplot(fig)

st.markdown("**Cluster Summary:**")
st.dataframe(rfm.groupby('Segment')[['Recency','Frequency','Monetary']].mean().round(2))

# ─── Classification ────────────────────────────────────────
st.markdown("---")
st.subheader("🤖 Purchase Prediction (Random Forest)")

rfm['Label'] = (rfm['Frequency'] > rfm['Frequency'].median()).astype(int)
X = rfm[['Recency','Frequency','Monetary']]
y = rfm['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Classification Report:**")
    st.dataframe(report_df)

with col2:
    st.markdown("**Feature Importance:**")
    feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
    fig, ax = plt.subplots(figsize=(5,3))
    feat_imp.sort_values().plot(kind='barh', ax=ax, color='teal')
    ax.set_title("Feature Importance")
    plt.tight_layout()
    st.pyplot(fig)

# ─── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Final Year Project | Consumer Behaviour Analysis using ML**")
