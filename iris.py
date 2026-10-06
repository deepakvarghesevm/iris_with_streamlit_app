import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

# Page configuration
st.set_page_config(
    page_title="Iris Explorer",
    page_icon="🌸",
    layout="wide"
)

# Load dataset
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = [
    iris.target_names[i]
    for i in iris.target
]

# -------------------------------
# Custom CSS
# -------------------------------

st.markdown("""
<style>

.main {
    background-color: #fffaf5;
}

.hero {
    padding: 35px;
    border-radius: 20px;
    background: linear-gradient(135deg, #fff1e6, #fde2e4);
    text-align: center;
    margin-bottom: 30px;
}

.hero h1 {
    color: #7c3f58;
    font-size: 45px;
    margin-bottom: 10px;
}

.hero p {
    color: #6b4f4f;
    font-size: 19px;
}

.card {
    padding: 22px;
    border-radius: 15px;
    background-color: #fff;
    border: 1px solid #f0d9d9;
    text-align: center;
    margin-bottom: 20px;
}

.card h3 {
    color: #7c3f58;
}

.card p {
    color: #6b5b5b;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------
# Hero Section
# -------------------------------

st.markdown("""
<div class="hero">

<h1>🌸 Welcome to Iris Explorer</h1>

<p>
Explore the famous Iris dataset through interactive
visualizations and machine learning.
</p>

</div>
""", unsafe_allow_html=True)


# -------------------------------
# Introduction
# -------------------------------

st.subheader("🌿 Discover the Iris Dataset")

st.write("""
The Iris dataset is one of the most popular datasets used in
Data Science and Machine Learning. It contains measurements
of iris flowers belonging to three different species:
**Setosa, Versicolor, and Virginica.**
""")

st.write("""
Use this application to explore the data, understand relationships
between flower features, and predict the species of a flower using
a Machine Learning model.
""")


# -------------------------------
# Dataset Highlights
# -------------------------------

st.subheader("📊 Dataset at a Glance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌱 Samples", "150")

with col2:
    st.metric("📏 Features", "4")

with col3:
    st.metric("🌸 Species", "3")

with col4:
    st.metric("📈 Model", "Decision Tree")


# -------------------------------
# Features
# -------------------------------

st.subheader("🔍 What You Can Explore")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="card">

    <h3>📊 Visualize the Data</h3>

    <p>
    Explore relationships between flower measurements using
    scatter plots, histograms, boxplots and correlation heatmaps.
    </p>

    </div>
    """, unsafe_allow_html=True)


with col2:

    st.markdown("""
    <div class="card">

    <h3>🤖 Predict the Species</h3>

    <p>
    Enter the measurements of an Iris flower and use a
    Decision Tree model to predict its species.
    </p>

    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# Species
# -------------------------------

st.subheader("🌸 The Three Iris Species")

species_col1, species_col2, species_col3 = st.columns(3)

with species_col1:

    st.markdown("""
    <div class="card">

    <h3>🌷 Setosa</h3>

    <p>One of the three species in the dataset.</p>

    </div>
    """, unsafe_allow_html=True)


with species_col2:

    st.markdown("""
    <div class="card">

    <h3>🌺 Versicolor</h3>

    <p>A species with intermediate flower measurements.</p>

    </div>
    """, unsafe_allow_html=True)


with species_col3:

    st.markdown("""
    <div class="card">

    <h3>🌹 Virginica</h3>

    <p>The species with generally larger flower measurements.</p>

    </div>
    """, unsafe_allow_html=True)


# -------------------------------
# Navigation message
# -------------------------------

st.markdown("---")

st.info(
    "👈 Use the sidebar to move to **Visualization** "
    "and **Prediction**."
)