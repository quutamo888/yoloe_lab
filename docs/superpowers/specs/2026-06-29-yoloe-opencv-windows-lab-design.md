# Design Spec: YOLOE OpenCV Windows Lab Jupyter Notebook

**Date:** 2026-06-29  
**Status:** Approved  
**Topic:** Interactive Jupyter Notebook for YOLOE (Ultralytics) on Windows using OpenCV (Webcam & Video) with line-by-line Thai explanations.

---

## 1. Goal Description
The objective is to create a new Jupyter Notebook `yoloe_opencv_lab.ipynb` in the workspace directory. This notebook will guide the user through setting up, configuring, and running the open-vocabulary object detection and segmentation model **YOLOE-26** (or other YOLOE-based models) using a PC/Windows environment with standard OpenCV webcam and video capture inputs. 

All code blocks will include detailed, line-by-line explanations in clear, easy-to-understand Thai language.

---

## 2. Notebook Structure & Content

The notebook will be structured into 6 sequential sections:

### Section 1: Installation & Setup (ติดตั้งไลบรารีที่จำเป็น)
Installs `ultralytics`, `onnxruntime`, `opencv-python`, `numpy`, `matplotlib`.

### Section 2: Import & Check Model (นำเข้าโมเดลและตรวจสอบคลาส YOLOE)
Imports the YOLOE class and loads the `yoloe-26s-seg.pt` model.

### Section 3: YOLOE Prompt-Free Mode (การตรวจจับแบบอัตโนมัติ - Prompt-Free)
Runs the model on a sample video/webcam feed without any target prompts. Displays the real-time feed in an OpenCV window with bounding boxes/segmentation masks and FPS.

### Section 4: Custom Text Prompting (ตรวจจับวัตถุตามสั่งด้วยคำค้นหา - Text Prompts)
Applies `model.set_classes()` to dynamically set custom categories (e.g. `"coffee cup"`, `"keypad"`, `"water bottle"`) and runs real-time inference on the webcam or a video file.

### Section 5: YOLOE Visual Prompting (ตรวจจับวัตถุด้วยภาพตัวอย่าง - Visual Prompts)
Demonstrates cross-image prompting by taking a reference image of a target object, specifying its bounding box, and using the `YOLOEVPSegPredictor` to segment and locate visually similar objects in the live feed/target image.

### Section 6: Practical Applications (การประยุกต์ใช้งานจริง - Counting & Tracking)
- **Object Counter**: Triggers an action when the number of target objects exceeds a threshold.
- **Location Tracker**: Computes X, Y coordinates (0.0 to 1.0) on the screen to trigger region-based actions (e.g., detecting if a hand is in the bottom-right corner).

---

## 3. Detailed Proposed Changes

### [NEW] [yoloe_opencv_lab.ipynb](file:///f:/test/yoloe_lab/yoloe_opencv_lab.ipynb)
A complete Jupyter Notebook containing Markdown cells explaining concepts and Code cells implementing YOLOE with line-by-line Thai comments.

Below is the conceptual content and code design of the notebook cells:

#### Cell 1 (Markdown) - Title & Intro
Explanation of YOLOE's zero-shot open-vocabulary capabilities and what the user will achieve in this lab.

#### Cell 2 (Code) - Installation
```python
# อัปเดตไลบรารี ultralytics และติดตั้งโมดูลที่เกี่ยวข้องสำหรับประมวลผลและการแสดงผล
!pip install -U ultralytics onnxruntime opencv-python numpy matplotlib
```

#### Cell 3 (Code) - Imports
```python
import cv2  # นำเข้าไลบรารี OpenCV สำหรับจัดการกล้อง วิดีโอ และการแสดงผลหน้าต่างภาพ
import numpy as np  # นำเข้า NumPy สำหรับประมวลผลข้อมูลโครงสร้างอาร์เรย์ (เช่น พิกัดกล่อง)
import matplotlib.pyplot as plt  # นำเข้า Matplotlib สำหรับการแสดงผลภาพนิ่งใน Notebook
import os  # นำเข้าโมดูล os สำหรับจัดการกับไฟล์และเส้นทางโฟลเดอร์
from ultralytics import YOLOE  # นำเข้าคลาส YOLOE เพื่อดึงโมเดลสำหรับทำคำสั่งภาษาอังกฤษ/ภาพ
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor  # นำเข้าตัวทำนายสำหรับการส่งภาพอ้างอิง (Visual Prompt)

print("นำเข้าไลบรารีที่จำเป็นทั้งหมดเรียบร้อยแล้ว!")
```

#### Cell 4 (Code) - Model Loading
```python
model_name = "yoloe-26s-seg.pt"  # กำหนดชื่อไฟล์โมเดล YOLOE-26 ขนาดเล็กสำหรับเซกเมนต์
model = YOLOE(model_name)  # โหลดโมเดลเข้ามาในตัวแปร model (ถ้าไม่มีโปรแกรมจะดาวน์โหลดอัตโนมัติ)
print("ดาวน์โหลดและโหลดโมเดล YOLOE สำเร็จ!")
```

#### Cell 5 (Code) - Section 1: Prompt-Free Mode
```python
# ส่วนทดสอบการจับภาพสดจากกล้องเว็บแคม (ID: 0) ในโหมดทั่วไป (Prompt-Free)
cap = cv2.VideoCapture(0)  # เปิดกล้องเว็บแคมหลักผ่านระบบ OpenCV

if not cap.isOpened():  # ตรวจสอบว่าเปิดกล้องเว็บแคมสำเร็จหรือไม่
    print("ไม่สามารถเชื่อมต่อกับกล้องเว็บแคมได้!")
else:
    cv2.namedWindow("YOLOE Prompt-Free Detection", cv2.WINDOW_NORMAL)  # สร้างหน้าต่างแสดงผลที่ยืดหยุ่นขนาดได้
    print("เริ่มตรวจจับวัตถุ กดปุ่ม 'q' หรือ 'Esc' ในหน้าต่างกล้องเพื่อออกจากการทำงาน")
    
    while True:
        ret, frame = cap.read()  # อ่านเฟรมล่าสุดจากกล้อง (ret เป็น True หากมีเฟรมเข้ามา)
        if not ret:  # หากอ่านเฟรมไม่สำเร็จ ให้หยุดการทำซ้ำ
            break
            
        results = model.predict(frame, verbose=False)  # ส่งเฟรมไปให้โมเดลทำนาย (แบบเงียบไม่พิมพ์ logs)
        annotated_frame = results[0].plot(boxes=True, masks=True)  # วาดกล่อง (boxes) และหน้ากากแบ่งส่วน (masks) ลงบนภาพ
        
        # แสดงผลภาพที่ผ่านการระบุวัตถุลงบนหน้าต่างกล้อง
        cv2.imshow("YOLOE Prompt-Free Detection", annotated_frame)
        
        key = cv2.waitKey(1) & 0xFF  # เช็กการกดปุ่มบนคีย์บอร์ด
        if key == ord('q') or key == 27:  # ถ้ากดปุ่ม 'q' หรือปุ่ม Esc (27) ให้หยุดการทำงาน
            break
            
    cap.release()  # ปล่อยการเชื่อมต่อกล้องเว็บแคมเพื่อประหยัดทรัพยากร
    cv2.destroyAllWindows()  # ปิดหน้าต่างแสดงผลของ OpenCV ทั้งหมด
    print("ปิดการใช้งานกล้องเว็บแคมเรียบร้อยแล้ว")
```

#### Cell 6 (Code) - Section 2: Custom Text Prompting
```python
# ตั้งค่าคลาสที่คุณอยากตรวจจับในรูปแบบข้อความ (Text Prompts) โดยไม่ต้องทำการเทรนใหม่
prompts = ["coffee cup", "cell phone", "keyboard", "human face"]  # กำหนดคำภาษาอังกฤษของสิ่งของที่ต้องการจับ
model.set_classes(prompts)  # ป้อนคำอธิบายเหล่านี้ให้ระบบโมเดลเข้าใจ
print(f"ตั้งค่าคลาสตรวจจับเรียบร้อย: {prompts}")

cap = cv2.VideoCapture(0)  # เปิดกล้องเว็บแคมอีกครั้ง
if not cap.isOpened():
    print("ไม่สามารถเชื่อมต่อกับกล้องเว็บแคมได้!")
else:
    cv2.namedWindow("YOLOE Text Prompt Detection", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, verbose=False)  # ทำนายเฟรมภาพโดยค้นหาเฉพาะเป้าหมายใน prompts
        annotated_frame = results[0].plot(boxes=True, masks=True)  # วาดกล่องและไฮไลต์หน้ากากแบ่งส่วน
        
        cv2.imshow("YOLOE Text Prompt Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
```

#### Cell 7 (Code) - Section 3: Visual Prompting (Cropping reference and predicting)
```python
# ส่วนการทำ Visual Prompting (การใช้รูปตัวอย่างในการตามหาวัตถุ)
# 1. ถ่ายภาพหรือดึงภาพที่มีวัตถุเพื่อใช้เป็นภาพอ้างอิง (Reference)
# ในขั้นตอนการเขียนโปรแกรม เราจะดึงภาพจากกล้องมาบันทึกเป็น ref.jpg เพื่อดึงกล่องพิกัดมาสอนโมเดล

cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("จัดวางวัตถุไว้หน้ากล้อง แล้วกดปุ่ม Spacebar เพื่อบันทึกภาพอ้างอิง")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Press Space to capture Reference Image", frame)
        if cv2.waitKey(1) & 0xFF == 32:  # 32 คือปุ่ม Spacebar
            cv2.imwrite("ref.jpg", frame)  # บันทึกภาพอ้างอิงลงเครื่องเป็นไฟล์ ref.jpg
            print("บันทึกภาพ ref.jpg สำเร็จ!")
            break
    cap.release()
    cv2.destroyAllWindows()

# 2. จำลองการเลือกพิกัดสี่เหลี่ยมรอบวัตถุในภาพอ้างอิง (Bounding Box)
# สมมติพิกัดเป็น [x1, y1, x2, y2]
# ตัวอย่าง: [100, 100, 300, 300] (ผู้ใช้สามารถกำหนดจุดกล่องพิกัดเพื่อนำมาสอนโมเดล)
# เราจะเขียนให้ผู้ใช้ป้อนพิกัดเพื่อตรวจจับ หรือแสดงภาพอ้างอิงพร้อมพิกัด
ref_img = cv2.imread("ref.jpg")
h, w, _ = ref_img.shape
# กำหนดจุดกึ่งกลางของภาพอ้างอิงเพื่อเป็นตัวอย่างกล่องพิกัดเบื้องต้น
box_coords = [int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.75)]

print(f"พิกัดกล่องอ้างอิงเบื้องต้น: {box_coords} (กว้าง x สูง: {w}x{h})")

# 3. ส่งข้อมูลรูปอ้างอิงและกล่องวัตถุให้ YOLOEVPSegPredictor ค้นหาในภาพจริงสดๆ จากกล้อง
visual_prompts = dict(
    bboxes=np.array([box_coords]),  # พิกัดกล่องของวัตถุบนภาพอ้างอิง
    cls=np.array([0])  # กำหนดให้เป็นวัตถุคลาสหลัก (หมายเลข 0)
)

cap = cv2.VideoCapture(0)
if cap.isOpened():
    cv2.namedWindow("YOLOE Visual Prompt Detection", cv2.WINDOW_NORMAL)
    print("กำลังทำงานในโหมดตรวจจับด้วยภาพตัวอย่าง...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # ส่ง target เป็นเฟรมกล้อง, refer_image เป็นภาพอ้างอิง, visual_prompts เป็นจุดพิกัดวัตถุในภาพอ้างอิง
        results = model.predict(
            source=frame,
            refer_image="ref.jpg",
            visual_prompts=visual_prompts,
            predictor=YOLOEVPSegPredictor,
            verbose=False
        )
        
        annotated_frame = results[0].plot(boxes=True, masks=True)
        cv2.imshow("YOLOE Visual Prompt Detection", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
```

#### Cell 8 (Code) - Section 4: Object Counter & Location Tracking
```python
# ส่วนการนับจำนวนวัตถุและการตรวจสอบตำแหน่งพิกัด X, Y บนหน้าจอ
TARGET_OBJECT = "coffee cup"  # กำหนดชนิดวัตถุที่เราเพิ่งตั้งค่าไว้ใน Text Prompts
CONFIDENCE_THRESHOLD = 0.25  # กำหนดระดับความมั่นใจขั้นต่ำที่ยอมรับได้ (0.0 ถึง 1.0)

# คืนค่าโมเดลกลับมาใช้ Text Prompts ของเรา
model.set_classes([TARGET_OBJECT])

cap = cv2.VideoCapture(0)
if cap.isOpened():
    cv2.namedWindow("YOLOE Counter & Location Tracking", cv2.WINDOW_NORMAL)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model.predict(frame, verbose=False)
        annotated_frame = results[0].plot(boxes=True, masks=False)
        
        # ตรวจสอบผลลัพธ์การตรวจจับ
        boxes = results[0].boxes
        object_count = 0
        
        for box in boxes:
            confidence = float(box.conf[0])  # ดึงค่าความมั่นใจของกล่องนั้นๆ
            if confidence >= CONFIDENCE_THRESHOLD:
                object_count += 1  # นับจำนวนวัตถุที่ผ่านเกณฑ์
                
                # พิกัดกล่องแบบมุม: [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0]
                # คำนวณหาจุดศูนย์กลางของวัตถุเทียบเป็นสัดส่วน (0.0 ถึง 1.0)
                cx = ((x1 + x2) / 2) / frame.shape[1]
                cy = ((y1 + y2) / 2) / frame.shape[0]
                
                # แสดงพิกัด X, Y ในหน้าต่าง Console
                print(f"พบ {TARGET_OBJECT}! พิกัดกึ่งกลาง: X={cx:.2f}, Y={cy:.2f} (ความมั่นใจ: {confidence:.2f})")
                
                # ตัวอย่างเงื่อนไข: หากอยู่บริเวณมุมล่างขวา (X > 0.5 และ Y > 0.5)
                if cx > 0.5 and cy > 0.5:
                    cv2.putText(annotated_frame, "TRIGGER: BOTTOM RIGHT!", (10, 80), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # พิมพ์และแสดงจำนวนวัตถุบนภาพ
        count_text = f"Count: {object_count}"
        cv2.putText(annotated_frame, count_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("YOLOE Counter & Location Tracking", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
```

---

## 4. Verification Plan

### Manual Verification
1. Run Jupyter Notebook and execute each cell sequentially on a Windows system.
2. Confirm the installation of libraries succeeds.
3. Test **Cell 5 (Prompt-Free)**: Verify that the external window pops up, captures frame from webcam, and highlights common objects automatically.
4. Test **Cell 6 (Custom Text Prompting)**: Set target prompt list (e.g. `"cell phone"`) and verify YOLOE only segments the requested items.
5. Test **Cell 7 (Visual Prompting)**: Position a custom object, capture it, and verify that the target object is segmented in subsequent video stream frames based on the visual example.
6. Test **Cell 8 (Counter & Tracking)**: Bring the target object to the bottom-right quadrant of the camera frame, verify that the console logs coordinates, and ensure that the "TRIGGER: BOTTOM RIGHT!" text overlay is rendered.
