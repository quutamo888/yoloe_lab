# YOLOE OpenCV Windows Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a comprehensive, interactive Jupyter Notebook (`yoloe_opencv_lab.ipynb`) for YOLOE object detection and segmentation on Windows using OpenCV, with line-by-line Thai code comments and explanations.

**Architecture:** A step-by-step Jupyter Notebook detailing installation, prompt-free inference, text prompt inference, visual prompt inference (using `YOLOEVPSegPredictor`), and practical counting and tracking applications.

**Tech Stack:** Python, Jupyter Notebook (.ipynb), Ultralytics YOLOE, OpenCV (cv2), NumPy, Matplotlib.

---

## User Review Required

> [!IMPORTANT]
> The notebook relies on `cv2.VideoCapture(0)` to access your default webcam. If you do not have a webcam connected, you can change `0` to a video file path (such as `"sample_video.mp4"`) in the respective code cells.

---

## Open Questions

None. The user has approved the design spec.

---

## Proposed Changes

### Jupyter Notebook

#### [NEW] [yoloe_opencv_lab.ipynb](file:///f:/test/yoloe_lab/yoloe_opencv_lab.ipynb)
Create a new Jupyter Notebook containing the markdown explanations and executable code cells for the lab.

---

## Tasks

### Task 1: Initialize Notebook and Setup/Installation Cell

**Files:**
- Create: `f:/test/yoloe_lab/yoloe_opencv_lab.ipynb`

- [ ] **Step 1: Write the Jupyter Notebook structure with the installation cell**
  Create `yoloe_opencv_lab.ipynb` containing a Markdown header explaining the lab and a Code cell to install the required libraries.
  
  *Code content for installation cell:*
  ```python
  # อัปเดตไลบรารี ultralytics และติดตั้งโมดูลที่เกี่ยวข้องสำหรับประมวลผลและการแสดงผล
  !pip install -U ultralytics onnxruntime opencv-python numpy matplotlib
  ```

- [ ] **Step 2: Commit**
  ```bash
  git add yoloe_opencv_lab.ipynb
  git commit -m "lab: initialize notebook and add installation cell"
  ```

---

### Task 2: Import & Check Model Cell

**Files:**
- Modify: `f:/test/yoloe_lab/yoloe_opencv_lab.ipynb`

- [ ] **Step 1: Add imports and model loading cells**
  Add a Markdown cell explaining the imports and a Code cell that imports dependencies and downloads/loads `yoloe-26s-seg.pt`.
  
  *Code content for imports cell:*
  ```python
  import cv2  # นำเข้าไลบรารี OpenCV สำหรับจัดการกล้อง วิดีโอ และการแสดงผลหน้าต่างภาพ
  import numpy as np  # นำเข้า NumPy สำหรับประมวลผลข้อมูลโครงสร้างอาร์เรย์ (เช่น พิกัดกล่อง)
  import matplotlib.pyplot as plt  # นำเข้า Matplotlib สำหรับการแสดงผลภาพนิ่งใน Notebook
  import os  # นำเข้าโมดูล os สำหรับจัดการกับไฟล์และเส้นทางโฟลเดอร์
  from ultralytics import YOLOE  # นำเข้าคลาส YOLOE เพื่อดึงโมเดลสำหรับทำคำสั่งภาษาอังกฤษ/ภาพ
  from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor  # นำเข้าตัวทำนายสำหรับการส่งภาพอ้างอิง (Visual Prompt)

  print("นำเข้าไลบรารีที่จำเป็นทั้งหมดเรียบร้อยแล้ว!")
  ```

  *Code content for model loading cell:*
  ```python
  model_name = "yoloe-26s-seg.pt"  # กำหนดชื่อไฟล์โมเดล YOLOE-26 ขนาดเล็กสำหรับเซกเมนต์
  model = YOLOE(model_name)  # โหลดโมเดลเข้ามาในตัวแปร model (ถ้าไม่มีโปรแกรมจะดาวน์โหลดอัตโนมัติ)
  print("ดาวน์โหลดและโหลดโมเดล YOLOE สำเร็จ!")
  ```

- [ ] **Step 2: Commit**
  ```bash
  git add yoloe_opencv_lab.ipynb
  git commit -m "lab: add imports and model loading cells"
  ```

---

### Task 3: Prompt-Free Inference Cell

**Files:**
- Modify: `f:/test/yoloe_lab/yoloe_opencv_lab.ipynb`

- [ ] **Step 1: Add Prompt-Free mode cell**
  Add a Markdown cell explaining the prompt-free detection capability (working with around 4,800 predefined categories), and a Code cell that runs a webcam capture loop displaying YOLOE detections in real-time.
  
  *Code content for Prompt-Free cell:*
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

- [ ] **Step 2: Commit**
  ```bash
  git add yoloe_opencv_lab.ipynb
  git commit -m "lab: add prompt-free mode cell"
  ```

---

### Task 4: Custom Text Prompting Cell

**Files:**
- Modify: `f:/test/yoloe_lab/yoloe_opencv_lab.ipynb`

- [ ] **Step 1: Add Custom Text Prompting cell**
  Add a Markdown cell explaining zero-shot open-vocabulary prompting, and a Code cell setting classes dynamically using `set_classes` and running real-time webcam inference.
  
  *Code content for Text Prompting cell:*
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

- [ ] **Step 2: Commit**
  ```bash
  git add yoloe_opencv_lab.ipynb
  git commit -m "lab: add text prompting cell"
  ```

---

### Task 5: Visual (Image) Prompting Cell

**Files:**
- Modify: `f:/test/yoloe_lab/yoloe_opencv_lab.ipynb`

- [ ] **Step 1: Add Visual Prompting cell**
  Add a Markdown cell explaining cross-image prompting, and a Code cell that captures `ref.jpg` using standard webcam input, defines a target bounding box, and runs `YOLOEVPSegPredictor` to predict live frames using the captured template.
  
  *Code content for Visual Prompting cell:*
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

- [ ] **Step 2: Commit**
  ```bash
  git add yoloe_opencv_lab.ipynb
  git commit -m "lab: add visual prompting cell"
  ```

---

### Task 6: Object Counter & Location Tracker Cell

**Files:**
- Modify: `f:/test/yoloe_lab/yoloe_opencv_lab.ipynb`

- [ ] **Step 1: Add Object Counter & Tracking cell**
  Add a Markdown cell explaining target-counting and screen quadrant coordinate-based triggers, and a Code cell capturing webcam frames, counting confident target detections, and drawing trigger text when coordinates hit the bottom-right quadrant.
  
  *Code content for Counter & Tracking cell:*
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

- [ ] **Step 2: Commit**
  ```bash
  git add yoloe_opencv_lab.ipynb
  git commit -m "lab: add counter and location tracker cell"
  ```

---

## Verification Plan

### Automated Tests
*None. This is an interactive notebook designed for real-time video/webcam data and GUI output.*

### Manual Verification
1. Run each cell of the notebook in Jupyter sequentially.
2. Verify that Cell 5 launches the prompt-free camera window, showing generic object overlays.
3. Verify that Cell 6 only overlays text-prompt target labels.
4. Verify that Cell 7 captures `ref.jpg` on spacebar, print coordinates, and segments target object in real-time.
5. Verify that Cell 8 tracks coordinates and triggers "TRIGGER: BOTTOM RIGHT!" when object is positioned in the bottom-right of the frame.
