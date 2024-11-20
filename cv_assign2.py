import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Function to display images side by side for comparison
def show_images_side_by_side(original, modified, title_original="Original Image", title_modified="Modified Image"):
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    modified_rgb = cv2.cvtColor(modified, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_rgb)
    ax[0].set_title(title_original)
    ax[0].axis('off')

    ax[1].imshow(modified_rgb)
    ax[1].set_title(title_modified)
    ax[1].axis('off')

    st.pyplot(fig)

# Transformation Functions
def apply_translation(image, tx, ty):
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    translated_image = cv2.warpAffine(image, M, (cols, rows))
    show_images_side_by_side(image, translated_image, "Original Image", "Translated Image")
    return translated_image

def apply_rotation(image, angle, scale=1.0):
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, M, (cols, rows))
    show_images_side_by_side(image, rotated_image, "Original Image", "Rotated Image")
    return rotated_image

def apply_scaling(image, fx, fy):
    scaled_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
    show_images_side_by_side(image, scaled_image, "Original Image", "Scaled Image")
    return scaled_image

def apply_shearing(image, shear_factor_x, shear_factor_y):
    rows, cols = image.shape[:2]
    M = np.float32([[1, shear_factor_x, 0], [shear_factor_y, 1, 0]])
    sheared_image = cv2.warpAffine(image, M, (cols + int(shear_factor_x * rows), rows + int(shear_factor_y * cols)))
    show_images_side_by_side(image, sheared_image, "Original Image", "Sheared Image")
    return sheared_image

# Streamlit App
def main():
    st.title("Image Transformation Application")

    st.markdown("""
    This application allows you to upload an image and apply various **Affine transformations**, including:
    - Translation
    - Rotation
    - Scaling
    - Shearing
    """)

    # Upload image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        # Convert the uploaded image to OpenCV format
        image = Image.open(uploaded_image)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Options for image processing
        option = st.sidebar.selectbox(
            "Choose a transformation", 
            ["None", "Translation", "Rotation", "Scaling", "Shearing"]
        )
        
        if option == "Translation":
            tx = st.slider("Translate X (pixels)", -100, 100, 0)
            ty = st.slider("Translate Y (pixels)", -100, 100, 0)
            apply_translation(image, tx, ty)
        
        elif option == "Rotation":
            angle = st.slider("Rotation Angle (degrees)", -180, 180, 0)
            scale = st.slider("Scale Factor", 0.5, 2.0, 1.0)
            apply_rotation(image, angle, scale)
        
        elif option == "Scaling":
            fx = st.slider("Scale Factor X", 0.5, 2.0, 1.0)
            fy = st.slider("Scale Factor Y", 0.5, 2.0, 1.0)
            apply_scaling(image, fx, fy)
        
        elif option == "Shearing":
            shear_factor_x = st.slider("Shear Factor X", -0.5, 0.5, 0.0)
            shear_factor_y = st.slider("Shear Factor Y", -0.5, 0.5, 0.0)
            apply_shearing(image, shear_factor_x, shear_factor_y)

if __name__ == "__main__":
    main()
