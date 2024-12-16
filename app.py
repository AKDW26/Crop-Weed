import streamlit as st
import cv2
import numpy as np
from streamlit_option_menu import option_menu
from roboflow import Roboflow
from detection import CropWeedDetector

api_key = "ywbRFI4K4ArkHodSLcw8"  # Replace with your actual API key

# Initialize the detector
@st.cache_resource
def get_detector():
    return CropWeedDetector(api_key)

def main():
    st.set_page_config(page_title="Crop Weed Detection", layout="wide")

    # Custom styling
    st.markdown("""
    <style>
    /* Add custom styles here */
    </style>
    """, unsafe_allow_html=True)

    # Initialize detector
    detector = get_detector()

    # Sidebar navigation
    with st.sidebar:
        page = option_menu(
            "Main Menu",
            ["Home", "About", "Contact"],
            icons=['house', 'people', 'envelope'],
            menu_icon="cast",
            default_index=0
        )

    if page == "Home":
        st.title("🌱 Crop Weed Detection")
        st.write("Upload an image to detect weeds in your crops!")

        # Image upload
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        # Detection settings
        with st.expander("Detection Settings"):
            confidence = st.slider("Confidence Threshold", 0, 100, 40)
            overlap = st.slider("Overlap Threshold", 0, 100, 30)

        if uploaded_file is not None:
            try:
                # Read and process image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if image is None:
                    st.error("Error: Could not read the image file.")
                    return

                with st.spinner('Analyzing image...'):
                    # Process the image using the detector's method
                    result = detector.process_image(image, confidence=confidence, overlap=overlap)
                    stats = detector.get_detection_stats(result)

                    # Display results
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Display annotated image
                        output_image_rgb = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
                        st.image(output_image_rgb, caption="Analyzed Image")

                    with col2:
                        # Display statistics
                        st.subheader("Detection Results")
                        st.metric("Total Detections", stats["total_detections"])
                        
                        # Class distribution
                        st.subheader("Class Distribution")
                        for class_name, count in stats["class_distribution"].items():
                            st.write(f"{class_name}: {count}")

                        # Confidence metrics
                        st.subheader("Confidence Metrics")
                        st.write(f"Mean: {stats['detection_confidence']['mean']:.2f}%")
                        st.write(f"Min: {stats['detection_confidence']['min']:.2f}%")
                        st.write(f"Max: {stats['detection_confidence']['max']:.2f}%")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif page == "About":
        st.title("About Crop Weed Detection")
        st.markdown("""
        ## 🌾 About Project

        Our Crop Weed Detection system uses advanced computer vision and machine learning to identify weeds among crops, helping farmers optimize their field management.

        ## 🚀 Impact

        - Reduced herbicide usage
        - Increased crop yield
        - Better resource management
        - Environmental sustainability

        ## 🔍 How It Works

        1. Upload an image of your crop field.
        2. Our AI model analyzes the image.
        3. Weeds and crops are identified and marked.
        4. Get detailed statistics about the detection.
        """)

    elif page == "Contact":
        st.title("Contact Us")
        st.write("📫 Get in touch with us!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name")
            email = st.text_input("Email")
        
        with col2:
            message = st.text_area("Message", height=150)
        
        if st.button("Send Message", type="primary"):
            if name and email and message:
                st.success("Thank you for your message! We'll get back to you soon.")
            else:
                st.error("Please fill in all fields.")

if __name__ == "__main__":
    main()
