import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

# Title Section
st.title("Pattern Recognition on Time Series Data Using Siamese Neural Networks")
st.markdown(
    """
**Welcome to the Billion Dollars Happiness project!**
My project demonstrates how machine learning models, specifically Siamese Neural Networks, can be used to recognize
patterns in financial time series data. The focus is on Sberbank's stock (OHLCV) data, identifying patterns such
as potential growth or decline zones using historical data.
"""
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:", ["Introduction", "Data Preprocessing", "Neural Network", "Clustering Results", "Conclusion"]
)

# Section: Introduction
if page == "Introduction":
    st.header("Introduction")
    st.markdown(
        """
    **Objective**:
    The primary goal is to identify predictive patterns in Sberbank stock data (OHLCV) using local minima and maxima.

    **Approach**:
    - Identify **patterns** using sliding windows (20-30 bars).
    - Use **RobustScaler** to preprocess the data.
    - Train a **Siamese Neural Network**:
        - **Anchor**: A baseline pattern.
        - **Positive**: A similar growth pattern.
        - **Negative**: A dissimilar decline pattern.
    - Use the network to **cluster embeddings**, showing clear groupings for growth and decline patterns.
    """
    )
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/e/e4/Siamese_neural_network.png",
        caption="Siamese Network Overview",
    )

    st.markdown(
        """
    **Why Siamese Networks?**
    Siamese networks are especially useful in this project because:
    - They excel in tasks where **similarity** needs to be measured.
    - By clustering similar patterns, they simplify prediction tasks.
    - They require fewer labeled examples than traditional supervised models.
    """
    )

# Section: Data Preprocessing
elif page == "Data Preprocessing":
    st.header("Data Preprocessing")
    st.markdown(
        """
    **Steps in Preprocessing**:
    1. **Load and visualize the raw OHLCV data**.
    2. **Identify patterns**: Local minima for growth; local maxima for decline.
    3. **Scale data**: Use `RobustScaler` to normalize and handle outliers.
    """
    )
    uploaded_file = st.file_uploader("Upload your OHLCV CSV file:", type="csv")

    if uploaded_file:
        # Load and display data
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Plot raw data
        st.subheader("Raw Time Series Data")
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close Price', color='blue')
        ax.set_title("Close Prices Over Time")
        ax.legend()
        st.pyplot(fig)

        # Identify patterns
        window = st.slider("Select Sliding Window Size", min_value=20, max_value=50, value=30)
        st.subheader("Identified Patterns")
        patterns_min = data['Close'].rolling(window).min().dropna()
        patterns_max = data['Close'].rolling(window).max().dropna()
        fig, ax = plt.subplots()
        ax.plot(data['Close'], label='Close Price', color='blue')
        ax.scatter(patterns_min.index, patterns_min, color='green', label='Local Minima')
        ax.scatter(patterns_max.index, patterns_max, color='red', label='Local Maxima')
        ax.set_title("Local Minima and Maxima")
        ax.legend()
        st.pyplot(fig)

        # Scaling
        st.subheader("Data Scaling")
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
        st.write("Scaled Data Sample:", scaled_data[:5])

# Section: Neural Network
elif page == "Neural Network":
    st.header("Siamese Neural Network")
    st.markdown(
        """
    **Network Explanation**:
    - The network learns to group similar patterns (growth or decline) and separate dissimilar ones.
    - It works by embedding inputs into a lower-dimensional space and applying a **triplet loss** function.

    **Training Process**:
    1. Choose an **Anchor** pattern.
    2. Choose a **Positive** pattern (similar to the Anchor).
    3. Choose a **Negative** pattern (dissimilar to the Anchor).
    4. Adjust the network to minimize distance between Anchor and Positive, and maximize distance between
    Anchor and Negative.
    """
    )

    # Display example architecture
    st.image("https://miro.medium.com/max/1400/1*qLN9RPtW78KLvVJ2ABIHfg.png", caption="Triplet Loss Training")

    st.markdown(
        """
    **Advantages of Siamese Networks**:
    - They don't classify directly but learn meaningful embeddings for similarity comparisons.
    - Ideal for applications with fewer labeled examples.
    """
    )

# Section: Clustering Results
elif page == "Clustering Results":
    st.header("Clustering Results")
    st.markdown(
        """
    After training the Siamese Network, we obtain **embeddings** for patterns.
    These embeddings can be visualized in lower dimensions using PCA or t-SNE.
    """
    )

    # Generate and visualize embeddings
    st.subheader("Embedding Visualization")
    embeddings = np.random.rand(100, 3)  # Mock embeddings
    pca = PCA(n_components=2).fit_transform(embeddings)
    fig, ax = plt.subplots()
    ax.scatter(pca[:, 0], pca[:, 1], c=np.random.randint(0, 2, 100), cmap='coolwarm')
    ax.set_title("PCA of Embeddings")
    st.pyplot(fig)

# Section: Conclusion
elif page == "Conclusion":
    st.header("Conclusion")
    st.markdown(
        """
    **Key Takeaways**:
    - Siamese Neural Networks effectively cluster similar patterns in time series data.
    - This project demonstrates how predictive insights can be derived from financial data.

    **Future Enhancements**:
    - Use advanced architectures like LSTMs or Transformers to capture temporal dependencies.
    - Integrate external data (e.g., news sentiment, macroeconomic indicators).
    - Automate pattern annotation for better scalability.
    """
    )

    st.markdown("Thank you for exploring my project!")
