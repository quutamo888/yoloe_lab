import cv2
from ultralytics import YOLO

lane_points = []

# ฟังก์ชัน Callback สำหรับการคลิกเมาส์หาพิกัดคำนวณเส้นแบ่งเลน
def mouse_callback(event, x, y, flags, param):
    global lane_points
    if event == cv2.EVENT_LBUTTONDOWN:
        # แปลงพิกัดกลับตามสเกลรูปภาพจริง
        orig_x, orig_y = int(x / ratio), int(y / ratio)
        print(f"Mouse clicked at: X={orig_x}, Y={orig_y}")
        lane_points.append((orig_x, orig_y))

        # เมื่อคลิกครบ 2 จุด ให้คำนวณความชัน (slope) และจุดตัดแกน (intercept)
        if len(lane_points) == 2:
            (x1, y1), (x2, y2) = lane_points
            if y2 != y1:  
                slope = (x2 - x1) / (y2 - y1)
                intercept = x2 - slope * y2
                print(f"\nLane divider equation: x = {slope:.3f} * y + {intercept:.1f}")
            lane_points.clear()

# หาพิกัด X ของเส้นแบ่งเลนที่ตำแหน่ง Y
def get_lane_divider_x(y):
    return int(lane_divider_slope * y + lane_divider_intercept)

# ponytail: shrink - combined identical line_y_in/out and divider_x variables
ratio = 1.0  # สเกลของภาพแสดงผล
line_y = 550  # เส้นตรวจจับรถวิ่งข้าม

lane_divider_slope = 0.106
lane_divider_intercept = 613.5
divider_x = get_lane_divider_x(line_y)  # พิกัด X ของเส้นแบ่งเลนที่จุดตัดแกน Y

name = "YOLO car count"

# ตัวแปรสำหรับนับจำนวนและเก็บ ID รถแยกฝั่ง ขาเข้า (IN) และขาออก (OUT)
class_count_in, class_count_out = {}, {}
crossed_in_ids, crossed_out_ids = set(), set()

# เซตสำหรับจำกัดเฉพาะรถที่เริ่มวิ่งจากฝั่งก่อนข้ามเส้น เพื่อให้เริ่มต้นทุกเคาน์เตอร์เป็น 0
active_in_ids, active_out_ids = set(), set()

# โหลดโมเดล YOLO และเตรียมไฟล์วิดีโอ
model = YOLO('yolo12l.pt')
class_list = model.names
cap = cv2.VideoCapture('Road-traffic-video_2m.mp4')

cv2.namedWindow(name)
cv2.setMouseCallback(name, mouse_callback)

# วนลูปอ่านและประมวลผลเฟรมวิดีโอ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]

    # วาดเส้นแบ่งเลน (เฉียงจากแนวเส้นขอบฟ้า y=220 ลงมาถึงด้านล่าง)
    cv2.line(frame, (get_lane_divider_x(220), 220), (get_lane_divider_x(h), h), (255, 255, 255), 3)

    # วาดเส้นตรวจจับ (เลนซ้าย OUT สีแดง, เลนขวา IN สีเขียว)
    cv2.line(frame, (0, line_y), (divider_x, line_y), (0, 0, 255), 3)
    cv2.putText(frame, "OUT (left)", (200, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    cv2.line(frame, (divider_x, line_y), (w, line_y), (0, 255, 0), 3)
    cv2.putText(frame, "IN (right)", (divider_x + 200, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    
    # รัน YOLO Tracking (ตรวจจับและติดตามรถยนต์กับรถบรรทุก)
    results = model.track(frame, persist=True, classes=[2,7], device='cpu', verbose=False)
    
    # ตรวจสอบว่าโมเดลมีการตรวจจับและระบุ Track ID หรือไม่
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()
        
        for (x1, y1, x2, y2), track_id, class_idx in zip(boxes, track_ids, class_indices):
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2 
            class_name = class_list[class_idx]

            # วาดจุดศูนย์กลาง, กล่องรอบรถ และแสดงรายละเอียด ID
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # ตรวจสอบการข้ามเส้นตรวจจับ
            divider_x_at_vehicle = get_lane_divider_x(cy)

            # เลนขวาขาเข้า (IN): รถในเลนขวาวิ่งเข้ามาหาทางกล้อง (cy เพิ่มขึ้น)
            if cx > divider_x_at_vehicle:
                if cy < line_y:  # อยู่ก่อนข้ามเส้น (อยู่ครึ่งบนของภาพ)
                    active_in_ids.add(track_id)
                elif track_id in active_in_ids and track_id not in crossed_in_ids:  # ข้ามเส้นแล้ว (อยู่ครึ่งล่าง)
                    crossed_in_ids.add(track_id)
                    class_count_in[class_name] = class_count_in.get(class_name, 0) + 1
            # เลนซ้ายขาออก (OUT): รถในเลนซ้ายวิ่งออกห่างจากกล้อง (cy ลดลง)
            elif cx < divider_x_at_vehicle:
                if cy > line_y:  # อยู่ก่อนข้ามเส้น (อยู่ครึ่งล่างของภาพ)
                    active_out_ids.add(track_id)
                elif track_id in active_out_ids and track_id not in crossed_out_ids:  # ข้ามเส้นแล้ว (อยู่ครึ่งบน)
                    crossed_out_ids.add(track_id)
                    class_count_out[class_name] = class_count_out.get(class_name, 0) + 1
                
    # แสดงสถิติการนับจำนวนรถบนหน้าจอ
    y_offset = 30
    cv2.putText(frame, "VEHICLES IN (Right Lane):", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for class_name, count in class_count_in.items():
        y_offset += 25
        cv2.putText(frame, f"{class_name}: {count}", (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    y_offset += 35
    cv2.putText(frame, "VEHICLES OUT (Left Lane):", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for class_name, count in class_count_out.items():
        y_offset += 25
        cv2.putText(frame, f"{class_name}: {count}", (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # ปรับสเกลภาพแสดงผลตามค่า ratio
    if ratio != 1.0:
        frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
    cv2.imshow(name, frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Button q pressed")
        break

cap.release()
cv2.destroyAllWindows()