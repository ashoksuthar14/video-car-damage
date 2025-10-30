import streamlit as st
from car_pipeline import car_damage_pipeline  # Import your pipeline function
from PIL import Image
import os
import time

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Vehicle Damage Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- CUSTOM CSS STYLES ----------
st.markdown(
    """
    <style>
    /* Overall app background: soft gradient */
    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #fefeff 0%, #eaf2ff 100%);
        color: #333;
    }
    
    /* Make headers a nice blue color */
    h1, h2, h3, h4, h5, h6 {
        color: #0d6efd;
        font-family: "Arial", sans-serif;
    }

    /* Sidebar background: a subtle pale background */
    [data-testid="stSidebar"] {
        background-color: #f7f9fd;
        border-right: 1px solid #e6e6e6;
    }

    /* Tweak default text color, link color, etc. */
    .css-10trblm {  /* Body text in main area */
        color: #333;
    }
    
    /* Streamlit expander arrow color */
    .streamlit-expanderHeader {
        color: #0d6efd;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- FOLDERS & LAYOUT ----------
os.makedirs("static", exist_ok=True)

# --- SIDEBAR: File Uploader ---
with st.sidebar:
    st.title("Vehicle Damage Uploader")
    st.write("Upload a car image to detect damages:")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- MAIN PAGE TITLE ---
st.title("Vehicle Damage Detection & Repair Cost Estimation")
st.write(
    "This application analyzes your vehicle’s damage and provides an approximate repair cost. "
    "Simply upload an image from the sidebar and let our AI engine do the rest!"
)

# ---------- MAIN LOGIC ----------
if uploaded_file is not None:
    # Save the uploaded image
    image_path = "static/uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show spinner/animation while processing
    with st.spinner("Analyzing damage..."):
        time.sleep(1)  # A short delay so spinner is visible
        results = car_damage_pipeline(image_path)

    # Lay out results in two columns
    col1, col2 = st.columns(2, gap="large")

    # Original image
    with col1:
        st.subheader("Uploaded Image")
        st.image(image_path, caption="Original", use_container_width=True)

    # Annotated image
    annotated_image_path = results["annotated_image_path"]
    with col2:
        st.subheader("Predicted Damage")
        st.image(annotated_image_path, caption="Annotated", use_container_width=True)

    # --- Display cost estimation ---
    st.markdown("---")
    st.header("Damage & Cost Estimation Results")
    st.write(f"**Total Estimated Repair Cost:** AED {results['total_cost']:.2f}")

    # Expandable section for breakdown
    with st.expander("Cost Breakdown by Damage Type"):
        for cost in results["cost_breakdown"]:
            st.write(f"- **{cost['type'].capitalize()}**: AED {cost['estimated_cost']:.2f}")

    # Cleanup the temporary files (optional)
    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists(annotated_image_path):
        os.remove(annotated_image_path)

else:
    st.info("Please upload an image in the sidebar to proceed.")
