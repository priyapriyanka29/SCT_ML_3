import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# ===============================
# 🎨 PAGE CONFIG
# ===============================
st.set_page_config(page_title="Task 03 - SVM Image Classifier", layout="wide")

# ===============================
# 🏆 HEADER
# ===============================
st.markdown("""
# 🧠 MACHINE LEARNING TASK 03  
## 🐱🐶 Cat vs Dog Image Classification  
### 🔬 Model: Support Vector Machine (SVM) + HOG Features
---
""")

# ===============================
# 📌 SIDEBAR INFO
# ===============================
st.sidebar.title("📌 Project Details")
st.sidebar.markdown("""
**Track:** Machine Learning  
**Task:** Task 03  
**Model:** SVM (RBF Kernel)  
**Feature Extraction:** HOG  
**Dataset:** Cats vs Dogs  
""")

# ===============================
# 📤 UPLOAD SECTION
# ===============================
st.subheader("📤 Upload Image")

uploaded_file = st.file_uploader(
    "Upload a Cat or Dog Image", 
    type=["jpg", "png", "jpeg"]
)

# ===============================
# 🤖 LOAD MODEL
# ===============================
model = joblib.load("svm_model.pkl")

# ===============================
# 🔍 PREDICTION
# ===============================
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    # Preprocessing
    img_resized = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    features = hog(
        gray,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )

    features = features.reshape(1, -1)

    prediction = model.predict(features)

    with col2:
        st.subheader("🔍 Prediction Result")

        if prediction[0] == 0:
            st.success("🐱 CAT Detected")
        else:
            st.success("🐶 DOG Detected")

        st.info("""
        ⚙️ Model Details:
        - Algorithm: SVM (RBF Kernel)
        - Feature: HOG (Histogram of Oriented Gradients)
        - Input Size: 128x128
        """)

# ===============================
# 📄 FOOTER
# ===============================
st.markdown("""
---
### 👨‍💻 Developed by PRIYANKA R
🚀 Machine Learning Internship Project  
""")