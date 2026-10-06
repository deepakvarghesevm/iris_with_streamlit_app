import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

st.title("📊 Iris Dataset Visualization")

# Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Add species column
df["species"] = [
    iris.target_names[i]
    for i in iris.target
]

# Sidebar
st.sidebar.header("Visualization Options")

# Select X and Y features
x_axis = st.sidebar.selectbox(
    "Select X-axis",
    iris.feature_names
)

y_axis = st.sidebar.selectbox(
    "Select Y-axis",
    iris.feature_names
)

# -------------------------
# Scatter Plot
# -------------------------

st.subheader("🔵 Scatter Plot")

fig, ax = plt.subplots()

sns.scatterplot(
    data=df,
    x=x_axis,
    y=y_axis,
    hue="species",
    ax=ax
)

st.pyplot(fig)


# -------------------------
# Histogram
# -------------------------

st.subheader("📈 Histogram")

selected_feature = st.selectbox(
    "Select a feature",
    iris.feature_names
)

fig, ax = plt.subplots()

sns.histplot(
    data=df,
    x=selected_feature,
    kde=True,
    ax=ax
)

st.pyplot(fig)


# -------------------------
# Boxplot
# -------------------------

st.subheader("📦 Boxplot")

fig, ax = plt.subplots()

sns.boxplot(
    data=df,
    x="species",
    y=selected_feature,
    ax=ax
)

st.pyplot(fig)


# -------------------------
# Correlation Heatmap
# -------------------------

st.subheader("🔥 Correlation Heatmap")

fig, ax = plt.subplots()

correlation = df[iris.feature_names].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)