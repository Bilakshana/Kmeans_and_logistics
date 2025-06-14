import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import mode
import joblib

st.title("🔬 Cell Samples K-Means Clustering Classifier")

# Upload dataset
uploaded_file = st.file_uploader("Upload your cell_samples.csv file", type=["csv"])

if uploaded_file:
    cell_df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    st.write("Shape:", cell_df.shape)
    st.write("Class distribution:\n", cell_df['Class'].value_counts())

    # Clean data
    cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
    cell_df.loc[:, 'BareNuc'] = cell_df['BareNuc'].astype(int)

    features = ['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize',
                'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']

    X = cell_df[features].values
    y = cell_df['Class'].values

    # Select number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 5, 2)

    # Select features for scatter plot
    x_feature = st.selectbox("X-axis feature", features, index=0)
    y_feature = st.selectbox("Y-axis feature", features, index=1)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=4)
    kmeans.fit(X_train)

    # Predict test set clusters
    cluster_labels = kmeans.predict(X_test)

    # Map cluster to actual label
    mapped_labels = np.zeros_like(cluster_labels)
    for i in range(n_clusters):
        mask = (kmeans.labels_ == i)
        if np.sum(mask) > 0:
            common_label = mode(y_train[mask]).mode.item()
            mapped_labels[cluster_labels == i] = common_label

    # Show evaluation metrics
    accuracy = accuracy_score(y_test, mapped_labels)
    st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
    st.text("Classification Report:\n" + classification_report(y_test, mapped_labels))

    # Plot data with clusters and labels
    fig, ax = plt.subplots()
    scatter = ax.scatter(cell_df[x_feature], cell_df[y_feature], c='gray', alpha=0.3, label='All samples')

    # Plot benign and malignant samples with color
    benign = cell_df[cell_df['Class'] == 2]
    malignant = cell_df[cell_df['Class'] == 4]

    ax.scatter(benign[x_feature], benign[y_feature], c='blue', label='Benign (Class 2)', alpha=0.6)
    ax.scatter(malignant[x_feature], malignant[y_feature], c='red', label='Malignant (Class 4)', alpha=0.6)

    ax.set_xlabel(x_feature)
    ax.set_ylabel(y_feature)
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Input form for prediction
    st.write("---")
    st.header("Predict a new cell sample")

    user_input = {}
    for feat in features:
        user_input[feat] = st.number_input(f"Enter {feat}", value=float(cell_df[feat].median()))

    input_array = np.array([list(user_input.values())]).reshape(1, -1)
    pred_cluster = kmeans.predict(input_array)[0]

    # Map predicted cluster to class label
    pred_class = None
    for i in range(n_clusters):
        mask = (kmeans.labels_ == i)
        if i == pred_cluster and np.sum(mask) > 0:
            pred_class = mode(y_train[mask]).mode.item()

    if st.button("Predict Class"):
        if pred_class == 2:
            st.success("Prediction: Benign (Class 2)")
        elif pred_class == 4:
            st.error("Prediction: Malignant (Class 4)")
        else:
            st.warning("Prediction: Unknown class")

    # Save model button
    if st.button("Save Trained Model"):
        joblib.dump(kmeans, "kmeans_cell_classifier.joblib")
        st.success("💾 Model saved as 'kmeans_cell_classifier.joblib'")

else:
    st.info("Please upload the 'cell_samples.csv' dataset to get started.")
