
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# تحميل النموذج
model = load_model("eurosat_cnn_model.keras")

# أسماء الفئات
categories = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
              'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

st.title("تصنيف صور EuroSAT")

uploaded_file = st.file_uploader("اختر صورة", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_img(uploaded_file, target_size=(64, 64))
    st.image(img, caption='الصورة المدخلة', use_column_width=True)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = categories[np.argmax(prediction)]

    st.write(f"**الفئة المتوقعة:** {predicted_class}")

