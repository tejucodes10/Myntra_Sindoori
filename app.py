## Importing Packages
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Set the theme and styles to resemble Myntra's look
st.set_page_config(page_title="Style Guru", page_icon="👗", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f8f8f8;
        color: #333;
    }
    .main {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .banner {
        width: 100%;
        height: 400px;
        background-image: url("https://i.pinimg.com/originals/0b/46/ce/0b46ceb1581581d7ca5069c7120d269b.jpg");
        background-size: cover;
        background-position: center;
        margin-bottom: 2rem;
    }
    .title {
        text-align: center;
        font-size: 2.5rem;
        color: #ff3f6c;
        margin-bottom: 1rem;
    }
    .search-options {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        max-width: 600px;
        margin: 0 auto 2rem;
    }
    .search-button {
        background-color: #ff3f6c;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        cursor: pointer;
    }
    .results-section {
        margin: 2rem auto;
        max-width: 800px;
    }
    .result-item {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .result-item h3 {
        color: #ff3f6c;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Page title
st.markdown("<div class='banner'></div>", unsafe_allow_html=True)
st.markdown("<div class='title'>Meet your Style Guru</div>", unsafe_allow_html=True)

## Loading Samples of data
try:
    data = pd.read_csv("Dataset/Myntra Fasion Clothing.csv")
    data = data[:500]
    #st.success("Data Loaded")
except:
    st.error("Data Loading Exception")
    #print("Data Loading Exception")

## Encoder Loading
def encoder_search(text):
    """
    Function to encode the input text using SentenceTransformer.

    Parameters:
    - text (str): Input text to be encoded.

    Returns:
    - np.array: Encoded vector for the input text.
    """
    enocder = SentenceTransformer("all-mpnet-base-v2")
    search_vector = enocder.encode(text)
    _vector = np.array([search_vector])
    faiss.normalize_L2(_vector)
    return _vector

## Results Loader
def get_results(vector, k, index):
    """
    Function to retrieve search results based on the input vector.

    Parameters:
    - vector (np.array): Input vector for search.
    - k (int): Number of top results to retrieve.
    - index: Faiss index for similarity search.

    Returns:
    - pd.DataFrame: DataFrame containing search results.
    """
    distances, ann = index.search(vector, k=k)
    sim_scores = 1 / (1 + distances[0])
    results = pd.DataFrame({'distances': distances[0], 'ann': ann[0],'Score': sim_scores})
    merge_results_data = pd.merge(results, data, left_on='ann', right_index=True)
    return merge_results_data

## Reading the vector Index
try:  
    index = faiss.read_index("vector_store/myntra_embedding_vector_store.index")
    #st.success("Index File Loaded")
except:
    st.error("Index not loaded properly")
    #print("Index not loaded properly")

# Creating Streamlit UI
#st.markdown("<div class='search-options'>", unsafe_allow_html=True)
st.subheader("What's your outfit-mood for the day?")
search_text = st.text_input("Hukum")
k = st.slider("Number of options you would like:", min_value=1, max_value=10, value=3)
gender_filter = st.radio("Filter by Gender", ["All", "Men", "Women"])
sbub = st.button("Ask Style Guru to find your fits", key="explore", help="Click to search", use_container_width=True)

#cred_page = st.button("About This Project", key="about", help="Learn more about this project", use_container_width=True)
#st.markdown("</div>", unsafe_allow_html=True)

# Main section to display results
if sbub:
    # Perform the search and get results
    _vector = encoder_search(search_text)
    results = get_results(_vector, k, index)
    
    if gender_filter == "Men":
        results = results[results["category_by_Gender"] == "Men"]
    
    elif gender_filter == "Women":
        results = results[results["category_by_Gender"] == "Women"]

    # Display results in the main section
    #st.markdown("<div class='results-section'>", unsafe_allow_html=True)
    st.markdown("## Style Finds for You")
    st.markdown("---")
    for sample in results.itertuples():
        st.markdown("<div class='result-item'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{sample.BrandName}</h3>", unsafe_allow_html=True)
        st.write(f"Product Rating: {sample.Ratings}")
        st.write(f"Product URL: {sample.URL}")
        st.write(f"Product ID: {sample.Product_id}")
        st.write(f"Product Category: {sample.Category}")
        st.write(f"Gender: {sample.category_by_Gender}")
        st.write(f"Available Size: {sample.SizeOption}")
        st.write(f"Description: {sample.Description}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---")

    #st.markdown("</div>", unsafe_allow_html=True)


