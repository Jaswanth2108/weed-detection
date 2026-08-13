import streamlit as st
import os
import sys

# Ensure current directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection import (
    detect_and_plot_objects,
    load_yolo_model,
    identify_plant_species,
    PLANTNET_API_KEY
)

st.set_page_config(page_title="Weed Detection App", page_icon="🌿", layout="wide")

st.title("🌾 Weed Detection & Pl@ntNet Species Classifier")
st.markdown("Use YOLOv3 to detect weeds in crops and Pl@ntNet API to identify plant species.")

# Sidebar Settings
st.sidebar.title("Options")
api_key = st.sidebar.text_input("Pl@ntNet API Key", value=PLANTNET_API_KEY, type="password")
organ = st.sidebar.selectbox("Organ Type", ["leaf", "flower", "fruit", "bark", "auto"])

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    tab1, tab2 = st.tabs(["🎯 YOLO Object Detection", "🔬 Pl@ntNet API Species ID"])

    with tab1:
        st.subheader("YOLO Object Detection")
        try:
            m, class_names = load_yolo_model()
            output_image, count = detect_and_plot_objects(uploaded_file, m, class_names)
            st.pyplot(output_image, use_container_width=True)
            st.write(f"Detected **{count}** objects.")
        except Exception as e:
            st.error(f"Detection Error: {e}")

    with tab2:
        st.subheader("Pl@ntNet Plant Species Identification")
        uploaded_file.seek(0)
        with st.spinner("Calling Pl@ntNet API..."):
            res = identify_plant_species(uploaded_file, organ=organ, api_key=api_key)
            if res.get("success"):
                st.success(f"Best Match: {res.get('best_match')}")
                for idx, item in enumerate(res.get("results", [])):
                    st.write(f"### {idx+1}. {item['scientific_name']} ({item['score']}%)")
                    st.write(f"**Common Names:** {', '.join(item['common_names'])}")
                    st.write(f"**Family:** {item['family']}")
                    if item.get("images"):
                        st.image(item["images"][0], width=200)
            else:
                st.error(res.get("error"))
