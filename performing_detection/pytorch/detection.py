import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import io
import streamlit as st

from utils import *
from darknet import Darknet
from plantnet_api import identify_plant_species, PLANTNET_API_KEY

st.set_page_config(page_title="Crop & Weed Detection AI", page_icon="🌿", layout="wide")


def detect_and_plot_objects(image_bytes, model, class_names, iou_threshold=0.4, nms_threshold=0.6):
    """Perform YOLO Darknet object detection on image bytes."""
    img = np.array(Image.open(image_bytes))
    original_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize the image to the input width and height of the first layer of the network.
    resized_image = cv2.resize(original_image, (model.width, model.height))

    # Detect objects in the image
    boxes = detect_objects(model, resized_image, iou_threshold, nms_threshold)

    # Plot the image with bounding boxes and corresponding object class labels
    output_image = plot_boxes(original_image, boxes, class_names, plot_labels=True)
    return output_image, len(boxes)


@st.cache_resource
def load_yolo_model():
    """Load and cache YOLO model."""
    cfg_file = '../data/cfg/crop_weed.cfg'
    weight_file = '../data/weights/crop_weed_detection.weights'
    namesfile = '../data/names/obj.names'

    if not os.path.exists(cfg_file):
        cfg_file = 'performing_detection/data/cfg/crop_weed.cfg'
        weight_file = 'performing_detection/data/weights/crop_weed_detection.weights'
        namesfile = 'performing_detection/data/names/obj.names'

    m = Darknet(cfg_file)
    m.load_weights(weight_file)
    class_names = load_class_names(namesfile)
    return m, class_names


# ---------------- SIDEBAR ----------------
st.sidebar.title("🌿 Settings & Configuration")
analysis_mode = st.sidebar.radio(
    "Select Analysis Mode:",
    ["Combined (YOLO + Pl@ntNet API)", "YOLO Object Detection (Crop vs Weed)", "Pl@ntNet Species Identification API"]
)

st.sidebar.subheader("🔑 Pl@ntNet API Key")
api_key_input = st.sidebar.text_input("API Key", value=PLANTNET_API_KEY, type="password")

organ_selected = st.sidebar.selectbox(
    "Plant Organ Type (for Pl@ntNet API):",
    ["leaf", "flower", "fruit", "bark", "auto"]
)

st.sidebar.subheader("🎯 YOLO Parameters")
iou_thresh = st.sidebar.slider("IOU Threshold", 0.1, 1.0, 0.4, 0.05)
nms_thresh = st.sidebar.slider("NMS Threshold", 0.1, 1.0, 0.6, 0.05)


# ---------------- MAIN APP ----------------
st.title("🌱 Smart Agriculture: Weed Detection & Plant Identification AI")
st.markdown("""
This application combines **YOLOv3 Object Detection** for bounding box detection of crops and weeds 
with the **Pl@ntNet API** for real-time botanical species identification.
""")

# Load YOLO Model
try:
    yolo_model, class_names = load_yolo_model()
    st.success("✅ YOLOv3 Weed Detection Model Loaded Successfully!")
except Exception as e:
    st.error(f"⚠️ Could not load YOLO model weights: {e}")
    yolo_model, class_names = None, []

# Upload Image
uploaded_file = st.file_uploader("Upload a field/plant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.divider()

    # Create tabs or columns based on selected mode
    if analysis_mode == "YOLO Object Detection (Crop vs Weed)":
        st.header("🎯 YOLO Crop vs Weed Detection")
        if yolo_model:
            with st.spinner("Detecting crops and weeds..."):
                output_image, box_count = detect_and_plot_objects(
                    uploaded_file, yolo_model, class_names, iou_thresh, nms_thresh
                )
                st.pyplot(output_image, use_container_width=True)
                st.info(f"Total Detections Found: **{box_count}** objects")
        else:
            st.error("YOLO Model is not available.")

    elif analysis_mode == "Pl@ntNet Species Identification API":
        st.header("🔬 Pl@ntNet Botanical Species Identification")
        with st.spinner("Analyzing plant species via Pl@ntNet API..."):
            api_res = identify_plant_species(uploaded_file, organ=organ_selected, api_key=api_key_input)
            
            if api_res.get("success"):
                st.subheader(f"Best Match: **{api_res.get('best_match')}**")
                
                results = api_res.get("results", [])
                if results:
                    top_match = results[0]
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.metric("Scientific Name", top_match['scientific_name'])
                        st.metric("Family", top_match['family'] or "N/A")
                        st.metric("Genus", top_match['genus'] or "N/A")
                        st.write(f"**Common Names:** {', '.join(top_match['common_names']) if top_match['common_names'] else 'N/A'}")
                        st.progress(min(top_match['score'] / 100.0, 1.0), text=f"Confidence Score: {top_match['score']}%")

                    with col2:
                        if top_match.get("images"):
                            st.write("**Reference Species Images (Pl@ntNet):**")
                            st.image(top_match["images"][0], caption=top_match['scientific_name'], width=250)

                    # List other candidate species
                    if len(results) > 1:
                        st.markdown("### 📋 Alternative Candidate Matches")
                        for idx, cand in enumerate(results[1:], start=2):
                            with st.expander(f"#{idx}: {cand['scientific_name']} ({cand['score']}%)"):
                                st.write(f"**Common Names:** {', '.join(cand['common_names']) if cand['common_names'] else 'N/A'}")
                                st.write(f"**Family:** {cand['family']}")
                                if cand.get("images"):
                                    st.image(cand["images"][0], width=200)
                else:
                    st.warning("No matches found for this image.")
            else:
                st.error(f"Pl@ntNet API Error: {api_res.get('error')}")

    else:  # Combined (YOLO + Pl@ntNet API)
        st.header("🚀 Combined AI Analysis: YOLO Bounding Boxes + Pl@ntNet Species ID")
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("🎯 1. YOLO Crop & Weed Bounding Boxes")
            if yolo_model:
                with st.spinner("Running YOLO Detection..."):
                    uploaded_file.seek(0)
                    output_image, box_count = detect_and_plot_objects(
                        uploaded_file, yolo_model, class_names, iou_thresh, nms_thresh
                    )
                    st.pyplot(output_image, use_container_width=True)
                    st.info(f"Detections: **{box_count}** objects bounding-boxed")
            else:
                st.error("YOLO Model is not available.")

        with col_right:
            st.subheader("🔬 2. Pl@ntNet Species Identification")
            with st.spinner("Querying Pl@ntNet API..."):
                uploaded_file.seek(0)
                api_res = identify_plant_species(uploaded_file, organ=organ_selected, api_key=api_key_input)
                
                if api_res.get("success"):
                    st.success(f"Best Match: **{api_res.get('best_match')}**")
                    results = api_res.get("results", [])
                    if results:
                        top = results[0]
                        st.write(f"**Scientific Name:** *{top['scientific_name']}*")
                        st.write(f"**Common Names:** {', '.join(top['common_names']) if top['common_names'] else 'N/A'}")
                        st.write(f"**Family:** {top['family']}")
                        st.progress(min(top['score'] / 100.0, 1.0), text=f"Confidence: {top['score']}%")
                        if top.get("images"):
                            st.image(top["images"][0], caption="Pl@ntNet Reference Image", width=220)
                    else:
                        st.warning("No plant species match identified.")
                else:
                    st.error(f"Pl@ntNet API Error: {api_res.get('error')}")
