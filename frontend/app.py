import streamlit as st
import json
import numpy as np
import pickle
import scipy.sparse
import sys

# MUST be the first Streamlit command
st.set_page_config(page_title="LightFM Recommender Dashboard", layout="wide")

# Debug output (fine after set_page_config)
st.write("✅ Python executable:", sys.executable)

# More robust LightFM check
lightfm_available = False
try:
    from lightfm import LightFM
    lightfm_available = True
except ImportError:
    pass

if not lightfm_available:
    st.warning("⚠️ LightFM is not installed. Please run: pip install lightfm")

st.title("🎯 LightFM Recommender Frontend")

# Sidebar for file uploads
st.sidebar.header("📂 Upload Model Files")
metadata_file = st.sidebar.file_uploader("Upload model_metadata.json", type="json")
mappings_file = st.sidebar.file_uploader("Upload reverse_mappings.json", type="json")
embeddings_file = st.sidebar.file_uploader("Upload model_parameters.npz", type="npz")
features_file = st.sidebar.file_uploader("Upload item_features.npz (optional)", type="npz")
interactions_file = st.sidebar.file_uploader("Upload train_interactions.npz (optional)", type="npz")
model_file = st.sidebar.file_uploader("Upload light_model.pkl", type="pkl")

# Load metadata
metadata, mappings, embeddings_data = None, None, None
n_users = n_items = 0
if metadata_file:
    try:
        metadata_raw = metadata_file.read()
        metadata_str = metadata_raw.decode('utf-8') if isinstance(metadata_raw, bytes) else metadata_raw
        metadata = json.loads(metadata_str)
    except Exception as e:
        st.error(f"Failed to read metadata JSON: {e}")
        metadata = {}

    st.header("📊 Model Metadata")
    st.json(metadata.get("model_info", {}))
    n_users = metadata.get("model_info", {}).get("n_users", 0)
    n_items = metadata.get("model_info", {}).get("n_items", 0)

# Load mappings
if mappings_file:
    try:
        mappings = json.load(mappings_file)
        st.header("🔁 User ID Mappings")
        sample_users = list(mappings["id_to_user"].items())[:10]
        st.write("Sample User IDs:", sample_users)
    except Exception as e:
        st.error(f"Failed to read mappings JSON: {e}")

# Load and show embeddings
if embeddings_file:
    try:
        embeddings_data = np.load(embeddings_file)
        st.header("🧠 Model Embeddings")
        if 'user_embeddings' in embeddings_data:
            st.write("User Embedding Shape:", embeddings_data['user_embeddings'].shape)
        if 'item_embeddings' in embeddings_data:
            st.write("Item Embedding Shape:", embeddings_data['item_embeddings'].shape)
    except Exception as e:
        st.error(f"Failed to load embeddings: {e}")

# Show item features if provided
if features_file:
    try:
        features_data = np.load(features_file)
        st.header("📄 Item Features Matrix")
        st.write("Keys in NPZ file:", features_data.files)
        for k in features_data.files:
            st.write(f"{k}: {features_data[k].shape}")
    except Exception as e:
        st.error(f"Could not load item features: {e}")

# Load train interactions if provided
interactions = None
if interactions_file:
    try:
        inter_data = np.load(interactions_file)
        st.header("📈 Train Interaction Matrix")
        st.write("Keys in NPZ file:", inter_data.files)
        if {'data', 'row', 'col', 'shape'}.issubset(inter_data.files):
            interactions = scipy.sparse.coo_matrix((inter_data['data'], (inter_data['row'], inter_data['col'])), shape=inter_data['shape'])
            st.write("Shape:", interactions.shape)
            st.write("Non-zero interactions:", interactions.nnz)
        else:
            st.warning("Expected keys 'data', 'row', 'col', 'shape' not found in uploaded file.")
    except Exception as e:
        st.error(f"Failed to load train interactions: {e}")

# Recommender system interface
if model_file and metadata:
    st.header("🤖 Make Recommendations")
    try:
        model = pickle.load(model_file)

        user_input = st.text_input("Enter User ID (as string)", "0")
        if metadata and "user_mapping" in metadata and user_input in metadata["user_mapping"]:
            user_id = metadata["user_mapping"][user_input]
            scores = model.predict(user_id, np.arange(n_items))
            top_items = np.argsort(-scores)[:10]

            st.subheader(f"Top 10 Recommendations for User {user_input}")
            st.write(top_items.tolist())

            # Show historical interactions
            if interactions is not None:
                user_interacted_items = interactions.getrow(user_id).indices
                st.caption("Previously interacted item IDs:")
                st.write(user_interacted_items.tolist())
        else:
            st.warning("Please enter a valid User ID from the mapping.")

    except Exception as e:
        st.error(f"Failed to recommend: {e}")



