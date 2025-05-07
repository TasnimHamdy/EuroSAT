
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

# إعداد صفحة Streamlit
st.set_page_config(page_title="تصنيف صور الأقمار الصناعية - EuroSAT", page_icon="🛰️", layout="centered")

# عنوان الصفحة
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🛰️ تصنيف صور الأقمار الصناعية باستخدام EuroSAT</h1>
    <p style='text-align: center;'>ارفع صورة من القمر الصناعي ليتم تصنيفها تلقائيًا باستخدام نموذج الذكاء الاصطناعي المدرب.</p>
    """,
    unsafe_allow_html=True
)

# تحميل النموذج
model_path = "eurosat_cnn_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
else:
    st.error("❌ لم يتم العثور على ملف النموذج. تأكد من أن 'eurosat_model.keras' موجود في نفس المجلد.")
    st.stop()

# إعداد فئات التصنيف
class_labels = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]

# تحميل الصورة من المستخدم
uploaded_file = st.file_uploader("📤 ارفع صورة القمر الصناعي (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 الصورة المُرفوعة", use_container_width=True)

        # تحويل الصورة لحجم الإدخال المناسب
        image_resized = image.resize((64, 64))
        img_array = np.array(image_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # التنبؤ بالفئة
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        st.success(f"✅ الفئة المتوقعة: **{predicted_class}**")

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الصورة: {e}")
else:
    st.info("👆 قم برفع صورة لتبدأ عملية التصنيف.")