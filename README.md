Sound Event Detection

📌 คำอธิบายโปรเจค

Sound Event Detection เป็นระบบที่สามารถจำแนกเสียงจากไฟล์เสียงที่อัปโหลด โดยใช้โมเดล Deep Learning ซึ่งได้รับการฝึกฝนให้จำแนกเสียงประเภทต่างๆ เช่น เสียงแตรรถยนต์, เด็กเล่น, สุนัขเห่า และอื่นๆ โดยใช้ MFCC (Mel-Frequency Cepstral Coefficients) เป็นตัวแปลงเสียงให้อยู่ในรูปของฟีเจอร์ก่อนนำเข้าโมเดล

🚀 วิธีใช้งาน (Run Project)

🔹 1. ติดตั้ง Dependencies

- รันคำสั่งนี้เพื่อดาวน์โหลดไลบรารีที่จำเป็น: pip install -r requirements.txt

🔹 2. รันแอปพลิเคชัน Streamlit: streamlit run main.py

🔹 3. การอัปโหลดไฟล์เสียง

- เลือกไฟล์ .wav ที่ต้องการทดสอบ (สามารถโหลดไฟล์เสียงทดสอบได้จากโฟลเดอร์ sound_test)
- คลิก "🔍 วิเคราะห์เสียง" เพื่อให้โมเดลพยากรณ์ประเภทของเสียง

📊 การเทรนโมเดล

🔹 1. เตรียมข้อมูลเสียง

- ข้อมูลเสียงถูกแบ่งเป็น 7 คลาส ได้แก่:
  Air Conditioner, Car Horn, Children Playing, Dog Bark, Engine Idling, Jackhammer, Siren
- ทำการโหลดไฟล์เสียงและดึงคุณลักษณะ MFCC โดยใช้ librosa

🔹 2. Data Preprocessing

- ทำ Padding ให้ข้อมูลเสียงมีขนาดเท่ากัน (n_mfcc = 40)
- ใช้ StandardScaler() ในการทำ Scaling
- แบ่งข้อมูลเป็น Train (80%), Validation (10%), Test (10%)

🔹 3. สร้างโมเดล Deep Learning

- ใช้โมเดล RNN (LSTM) หรือ CNN + RNN
- มีโครงสร้างหลักดังนี้:
  model = Sequential([
  LSTM(2048, return_sequences=True, input_shape=(1, 6960)),
  LSTM(1024, return_sequences=False),
  Dense(7, activation='softmax')
  ])
- ใช้ categorical_crossentropy เป็น Loss Function
- ใช้ Optimizer RMSProp และวัดผลด้วย accuracy

🔹 4. การฝึกสอนโมเดล (Training)

- history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=1, callbacks = [early_stopping])

🔹 5. การบันทึกโมเดล

- หลังจากเทรนเสร็จแล้ว บันทึกโมเดลเพื่อใช้ใน Streamlit:
  model.save('models/Model_sound_event.h5')
  joblib.dump(scaler, 'models/scaler.pkl')

🛠 รายชื่อ Dependencies
Python 3.11, TensorFlow, Streamlit, NumPy, Librosa, Joblib, Scikit-Learn

❌ ปัญหาที่พบเจอ และแนวทางแก้ไข

1. ค่า Prediction ของโมเดลไม่เปลี่ยนแปลง (Predict ค่าคลาสเดียวตลอด)

ปัญหา: พบว่าโมเดลทำนายค่าเป็นคลาสเดียวกันทุกครั้ง ไม่ว่าไฟล์เสียงที่ป้อนเข้าจะเป็นเสียงประเภทใดก็ตาม
แนวทางแก้ไข: ตรวจสอบการทำ Scaling ของข้อมูล และตรวจสอบการกระจายของชุดข้อมูลว่าไม่เกิด Data Imbalance

2. StandardScaler ไม่ทำงานขณะ Deploy

ปัญหา: ค่า Scaling ที่ได้ใน Streamlit ไม่ตรงกับตอนเทรนโมเดล
แนวทางแก้ไข: บันทึก scaler.pkl หลังจาก fit กับ X_train และโหลดมาใช้แทนการ fit_transform() ใหม่ใน Streamlit

3. Memory Error บน Google Colab

ปัญหา: การเทรนโมเดลบน Google Colab ทำให้ RAM เต็ม เนื่องจากขนาดข้อมูลใหญ่เกินไป
แนวทางแก้ไข: ใช้ batch_size ที่เล็กลง และลดจำนวน LSTM Units เพื่อลดการใช้หน่วยความจำ

📌 หมายเหตุ

- โมเดลนี้ออกแบบมาเพื่อทำงานกับข้อมูลเสียงที่ผ่านการ Preprocessing เท่านั้น
- หากต้องการเทรนโมเดลใหม่ ควรใช้ scaler.pkl เดิม เพื่อให้ข้อมูลที่เข้าโมเดลอยู่ในช่วงเดียวกัน
- หากต้องการปรับแต่งโมเดล สามารถแก้ไขโค้ดที่ training_model.ipynb ได้

🚀 โปรเจคนี้ช่วยให้เข้าใจการประมวลผลเสียงด้วย Deep Learning และสามารถนำไปพัฒนาเพิ่มเติมได้!
