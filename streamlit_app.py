
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# تحميل النموذج
model = load_model("eurosat_model.keras")
class_names = sorted(os.listdir("EuroSAT/2750"))  # تأكد من المسار الصحيح للفئات

# تهيئة الصفحة
st.set_page_config(page_title="EuroSAT Image Classifier", layout="centered")
st.title("🌍 EuroSAT Image Classification App")
st.markdown("قم برفع صورة من القمر الصناعي ليتم تصنيفها تلقائيًا باستخدام نموذج الذكاء الاصطناعي المدرب.")

# رفع الصورة
uploaded_file = st.file_uploader("📸 ارفع صورة الأقمار الصناعية (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# المعالجة والتنبؤ
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="📷 الصورة المُرفوعة", use_container_width=True)

        img = image.resize((64, 64))  # نفس الحجم المستخدم في التدريب
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = class_names[np.argmax(prediction)]

        st.success(f"✅ الفئة المتوقعة: **{predicted_class}**")
    except Exception as e:
        st.error("حدث خطأ أثناء معالجة الصورة. تأكد من أن الملف صالح.")
        st.exception(e)
else:
    st.info("يرجى رفع صورة للبدء في التصنيف.")



