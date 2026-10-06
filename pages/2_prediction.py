import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

st.title("🤖 Iris Species Prediction")

st.write("Enter the flower measurements to predict the Iris species.")

# Load dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = DecisionTreeClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Performance")

st.metric(
    "Model Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.markdown("---")

st.subheader("🌸 Enter Flower Measurements")

# Input columns
col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5
    )

with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2
    )

st.markdown("---")

# Prediction button
if st.button("🔮 Predict Species"):

    # Create input data
    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    # Make prediction
    prediction = model.predict(input_data)

    # Get species name
    predicted_species = iris.target_names[prediction[0]]

    st.success(
        f"🌸 Predicted Species: **{predicted_species.capitalize()}**"
    )