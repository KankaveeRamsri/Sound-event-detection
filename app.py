import streamlit as st
import random
import tensorflow as tf
import numpy as np
import librosa
import io
import joblib


# ✅ ตั้งค่าพื้นหลังสีให้ดูสวยขึ้น
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
}
[data-testid="stSidebar"] {
    background-color: #1e293b;
}
h1 {
    background: -webkit-linear-gradient(45deg, #38bdf8, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ ตั้งชื่อแอป
st.markdown(
    """
    <h1 style='text-align: center; font-size: 50px;'>
        🎵 Sound Event Detection 🎶
    </h1>
    <p style='text-align: center; font-size: 18px; color: white;'> 
        🔍 อัปโหลดไฟล์เสียง และให้ AI ทายว่าเป็นเสียงอะไร!  
    </p>
    """,
    unsafe_allow_html=True,
)

# ✅ เปลี่ยนสี `st.info()` ให้ดูเข้ากับธีม
st.markdown(
    "<h3 style='color: white;'>🎧 เกล็ดความรู้เล็กน้อยเกี่ยวกับเสียง</h3>",
    unsafe_allow_html=True,
)
fun_facts = [
    "🔍 เสียงของ 'siren' สามารถเดินทางได้ไกลขึ้นในอากาศเย็น!",
    "🎶 คลื่นเสียงเดินทางเร็วขึ้นในน้ำมากกว่าในอากาศถึง 4 เท่า!",
    "📢 หูมนุษย์สามารถได้ยินเสียงที่มีความถี่ตั้งแต่ 20Hz ถึง 20,000Hz!",
    "🦇 ค้างคาวใช้คลื่นเสียงที่สูงเกินกว่าที่มนุษย์จะได้ยิน (Ultrasonic) เพื่อนำทาง!",
    "🚗 เสียงเครื่องยนต์สามารถช่วยวิเคราะห์ความผิดปกติของรถได้!",
]
st.success(random.choice(fun_facts))

# ✅ อธิบายกระบวนการก่อนที่ AI จะเข้าใจเสียง (เปลี่ยนสีให้ดูน่าสนใจขึ้น)
st.markdown(
    "<h2 style='color: #38bdf8;'>🔬 ก่อนที่ AI จะเข้าใจเสียง ต้องผ่านขั้นตอนอะไรบ้าง?</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    """
<p style='color: white; font-size: 16px;'>
ก่อนที่เสียงจะถูกนำไปใช้ใน AI เราไม่สามารถป้อนเสียงดิบเข้าไปได้โดยตรง  
แต่ต้องผ่านกระบวนการแปลงให้อยู่ในรูปแบบที่ AI สามารถเข้าใจได้ ดังนี้:
</p>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<ul style='color: white; font-size: 16px;'>
<li>📂 <b>โหลดไฟล์เสียง</b> → AI ต้องโหลดไฟล์เสียง เช่น <code>.wav</code></li>
<li>🎛 <b>แปลงเสียงให้อยู่ในรูปของตัวเลข</b> → AI ใช้ <code>MFCC</code>, <code>Mel Spectrogram</code></li>
<li>🤖 <b>ส่งเข้าโมเดล AI</b> → ให้โมเดลพยากรณ์ประเภทของเสียง</li>
<li>📊 <b>แสดงผลลัพธ์ให้ผู้ใช้ดู</b> → โมเดลบอกว่าเสียงที่ได้ยินเป็นเสียงอะไร 🎯</li>
</ul>
""",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='color:#38bdf8; font-size:20px;'>📢 อัปโหลดไฟล์เสียง แล้วให้โมเดลทายว่าเป็นเสียงอะไร!</p>",
    unsafe_allow_html=True,
)

# ✅ โหลดโมเดล
model = tf.keras.models.load_model(
    "models/Model_sound_event.h5"
)  # ✅ เปลี่ยน path ให้ถูกต้อง
# scaler = StandardScaler()
scaler = joblib.load("models/scaler.pkl")  # ✅ โหลด Scaler ที่ใช้ตอนเทรน

# ✅ Class Labels
class_labels = {
    0: "Air Conditioner",
    1: "Car Horn",
    2: "Children Playing",
    3: "Dog Bark",
    4: "Engine idling",
    5: "Jackhammer",
    6: "Siren",
}


# ✅ ฟังก์ชันดึง MFCC และแปลงเป็นเวกเตอร์
def extract_mfcc(file, sr=22050, n_mfcc=40, min_n_fft=1024):
    y, sr = librosa.load(io.BytesIO(file.read()), sr=sr)
    if len(y) < min_n_fft:
        return None
    n_fft_value = min(2048, len(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft_value)
    return mfcc.flatten()


# ✅ อัปโหลดไฟล์เสียง
st.warning("📂 กรุณาอัปโหลดไฟล์เสียง (.wav) ที่ต้องการให้ AI วิเคราะห์")
uploaded_file = st.file_uploader("", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.write("✅ **ไฟล์ที่คุณอัปโหลด:**", uploaded_file.name)

    mfcc_features = extract_mfcc(uploaded_file)
    if mfcc_features is not None:
        mfcc_features = mfcc_features.reshape(1, -1)

        target_features = 6960  # ✅ จำนวนฟีเจอร์ที่โมเดลต้องการ
        if mfcc_features.shape[1] < target_features:
            pad_width = target_features - mfcc_features.shape[1]
            mfcc_features = np.pad(
                mfcc_features,
                ((0, 0), (0, pad_width)),
                mode="constant",
                constant_values=-100,
            )
        elif mfcc_features.shape[1] > target_features:
            mfcc_features = mfcc_features[:, :target_features]

        # ✅ ใช้ `scaler.transform()` ที่โหลดมาแทน fit_transform()
        print(mfcc_features)
        print("X_test ก่อน Scaling:", mfcc_features[:5])

        mfcc_features = scaler.transform(mfcc_features)
        print("X_test หลัง Scaling:", mfcc_features[:5])

        X_test = np.expand_dims(mfcc_features, axis=1)  # ✅ เปลี่ยนให้เป็น (1, 1, 6960)

        # ✅ ให้โมเดลพยากรณ์
        prediction = model.predict(X_test)
        print(prediction)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

    if st.button("🔍 วิเคราะห์เสียง"):
        st.success(f"✅ การพยากรณ์เสร็จสิ้น! เสียงนี้คือ **{predicted_label}**")
        st.balloons()
