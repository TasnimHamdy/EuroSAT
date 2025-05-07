
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import os

# إعداد الصفحة
st.set_page_config(page_title="تصنيف صور الأقمار الصناعية - EuroSAT", page_icon="🛰️", layout="centered")

# شريط جانبي يحتوي على اختيار الصفحة
page = st.sidebar.selectbox("🔽 اختر الصفحة", ["🏠 الصفحة الرئيسية", "ℹ️ حول التطبيق"])

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

# الصفحة الرئيسية
if page == "🏠 الصفحة الرئيسية":
    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50;'>🛰️ تصنيف صور الأقمار الصناعية باستخدام EuroSAT</h1>
        <p style='text-align: center;'>ارفع صورة من القمر الصناعي ليتم تصنيفها تلقائيًا باستخدام نموذج الذكاء الاصطناعي المدرب.</p>
        """,
        unsafe_allow_html=True
    )

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

# صفحة "حول التطبيق"
elif page == "ℹ️ حول التطبيق":
    st.markdown(
        """
        ## ℹ️ حول المشروع

        هذا المشروع يستخدم نموذج ذكاء اصطناعي مبني على شبكة عصبية (CNN) مدربة على مجموعة بيانات EuroSAT لتصنيف صور الأقمار الصناعية إلى 10 فئات مختلفة مثل:
        
        - 🌲 Forest
        - 🏘️ Residential
        - 🛣️ Highway
        - 🌾 Crop Types
        - 🌊 Sea/Lake
        - وغيرهم...

        ### 👨‍💻 كيف يعمل التطبيق؟
        1. يقوم المستخدم برفع صورة قمر صناعي.
        2. يتم معالجة الصورة وتغيير حجمها تلقائيًا.
        3. النموذج المدرب يقوم بتحليل الصورة وتصنيفها حسب الفئة الأقرب.
        4. تظهر النتيجة مباشرة.

        ### 📁 ملفات المشروع:
        - streamlit_app.py: الكود الرئيسي للتطبيق.
        - eurosat_model.keras: النموذج المدرب بصيغة Keras.
        - README.md: ملف توضيحي للمشروع.

        ### 🧠 أدوات وتقنيات مستخدمة:
        - Python & Streamlit
        - TensorFlow / Keras
        - مكتبة PIL للصور
        - NumPy

        ---
        تم تنفيذ هذا المشروع لأغراض تعليمية وعرض قدرات الذكاء الاصطناعي في مجال الرؤية الحاسوبية 🌍🛰️
        """
    )


