import cv2
import numpy as np
import json
import torch
from ultralytics import YOLO
import requests

# Load YOLOv8 model
model = YOLO(r"C:\Users\1503\Desktop\University\Flask_Server\best.pt")  # Custom YOLOv8 model path

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.eval()
midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# 칼만 필터q
dt = 1  # time interval
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
P = np.eye(4)
R = np.array([[5, 0],
              [0, 5]])
Q = np.array([[0.1, 0, 0, 0],
              [0, 0.1, 0, 0],
              [0, 0, 0.1, 0],
              [0, 0, 0, 0.1]])

# Video stream
# cap = cv2.VideoCapture(r"C:\Users\Study\Videos\Captures\Tokyo, Japan 4K Walking Tour - Captions & Immersive Sound [4K Ultra HD_60fps] - YouTube - Whale 2024-11-04 22-36-27.mp4")
image = cv2.imread(r"C:\Users\1503\Desktop\University\Flask_Server\uploads\1.jpg")
frame_height, frame_width, _ = image.shape
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
base_zone_top = frame_height - 1000
# Frame dimensions and base zone

frame_count = 0
previous_sizes = {}
boxes_to_draw = []
previous_sizes_interval = {}
highest_risk_label = None
highest_risk_score = -float('inf')  # 초기값 설정
highest_risk_update_interval = 10
depth_map = None  # Depth map is updated only when necessary
depth_map_saved_count = 0  # Counter for saved depth maps

# 프레임 간 깊이 값 변화를 추적하기 위한 딕셔너리
previous_depths = {}
# 이전 위치 정보를 저장하는 딕셔너리
previous_positions = {}
# 이전 코드에서 누적된 size_change_rate 값을 저장할 딕셔너리
human_size_change_rates = []
file_path = r"C:\Users\1503\Desktop\University\Flask_Server\uploads\1.jpg"
# Video stream loop
# while cap.isOpened():
url = "http://59.19.113.82:3003/tts"
json_data = '{"label": "None", "tts": "아무것도없음"}'
while True:

        # ret, frame = cap.read()
        # print(f'프레임: {frame}')
        # if not ret:
        #     break
        try:
            data = np.fromfile(file_path, np.uint8)
            if data.size > 0:
                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                print("이미지 로드 성공!")
            else:
                print("파일이 비어 있습니다.")
        except Exception as e:
            print("파일 로드 중 오류 발생:", e)
        frame_count += 1
        # fps = cap.get(cv2.CAP_PROP_FPS)
        fps = 20
        frame_time = 1 / fps
        pedestrian_speed = 0.68  # 보행자 속도 (m/s)

        if depth_map is None:
            input_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_tensor = midas_transform(input_rgb).unsqueeze(0)
            if input_tensor.dim() == 5:
                input_tensor = input_tensor.squeeze(1)

            with torch.no_grad():
                depth_map = midas(input_tensor).squeeze().cpu().numpy()

            depth_map = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX)

        if frame_count % 3 == 0:
            results = model(frame)

            if frame_count % highest_risk_update_interval == 0:
                highest_risk_score = -float('inf')

            boxes_to_draw = []
            class_counts = {}

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.conf[0] >= 0.7:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width, height = x2 - x1, y2 - y1

                        current_position = np.array([[center_x], [center_y]])
                        current_size = width * height

                        if base_zone_top <= center_y <= frame_height:
                            cls_id = int(box.cls[0])
                            label = model.names[cls_id] if cls_id < len(model.names) else "Unknown"

                            if label not in class_counts:
                                class_counts[label] = 1
                            else:
                                class_counts[label] += 1
                            unique_label = f"{label}{class_counts[label]}"

                            depth_map_resized = cv2.resize(depth_map, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)
                            depth_area = depth_map_resized[y1:y2, x1:x2]
                            avg_depth = np.mean(depth_area) if depth_area.size > 0 else 1

                            x = np.array([[center_x], [center_y], [0], [0]])
                            x = F @ x
                            P = F @ P @ F.T + Q

                            measurement = np.array([[center_x], [center_y]])
                            S = H @ P @ H.T + R
                            K = P @ H.T @ np.linalg.inv(S)
                            y = measurement - (H @ x)
                            x = x + K @ y
                            P = (np.eye(4) - K @ H) @ P

                            vx, vy = x[2, 0], x[3, 0]
                            object_speed = np.sqrt(vx ** 2 + vy ** 2)

                            direction_vector = current_position - previous_positions.get(unique_label, current_position)
                            camera_center = np.array([[frame_width // 2], [frame_height // 2]])
                            to_camera_vector = camera_center - current_position

                            direction_magnitude = np.linalg.norm(direction_vector)
                            to_camera_magnitude = np.linalg.norm(to_camera_vector)

                            if direction_magnitude > 0 and to_camera_magnitude > 0:
                                cos_theta = np.dot(direction_vector.ravel(), to_camera_vector.ravel()) / (
                                        direction_magnitude * to_camera_magnitude)
                                if cos_theta > 0:
                                    speed_risk_factor = cos_theta * 0.0835
                                else:
                                    speed_risk_factor = 0
                            else:
                                speed_risk_factor = 0

                            # 상대 속도 계산 부분에 이어서
                            size_change_rate = 0
                            if unique_label in previous_sizes_interval and frame_count % 4 == 0:
                                interval_size = previous_sizes_interval[unique_label]
                                size_change_rate = (current_size - interval_size) / (
                                            interval_size + 1e-5) if current_size > interval_size else 0

                            # size_change_rate가 0.3 이상일 때만 업데이트
                            if size_change_rate >= 0.3:
                                previous_sizes_interval[unique_label] = current_size

                            # 위험 점수 계산
                            depth_change_rate = 0
                            if unique_label in previous_depths:
                                previous_depth = previous_depths[unique_label]
                                depth_change_rate = abs(avg_depth - previous_depth) / (previous_depth + 1e-5) if avg_depth > previous_depth else 0
                                # depth_change_rate 출력 (디버깅용)
                                print(f"Depth Change Rate ({unique_label}): {depth_change_rate:.4f}")

                            # 이전 평균 깊이 업데이트
                            previous_depths[unique_label] = avg_depth

                            # 'human' 클래스만 size_change_rate 저장
                            if label == "Human":
                                human_size_change_rates.append(size_change_rate)
                                print(f'size change rate: {size_change_rate}')

                            # 위험 점수 계산
                            risk_score = (
                                    speed_risk_factor +
                                    avg_depth * size_change_rate * 0.5 +
                                    current_size * avg_depth * 0.333 +
                                    depth_change_rate * 0.0835  # 깊이 변화율 가중치 추가
                            )
                            # 위험 점수 임계값 비교
                            risk_threshold = 1000
                            if risk_score > risk_threshold and risk_score > highest_risk_score:
                                highest_risk_score = risk_score
                                highest_risk_label = unique_label

                        previous_positions[unique_label] = current_position
                        boxes_to_draw.append((x1, y1, x2, y2, unique_label, avg_depth))

        for (x1, y1, x2, y2, unique_label, avg_depth) in boxes_to_draw:
            color = (0, 0, 255) if unique_label == highest_risk_label else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{unique_label} - Depth: {avg_depth:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if highest_risk_label:
            cv2.putText(frame, f"Highest Risk: {highest_risk_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if highest_risk_label.startswith('Human'):
                json_data = {"label": "Human", "tts": "사람이 다가오고 있습니다."}
            elif highest_risk_label.startswith('Motorcycle'):
                json_data = {"label": "Motorcycle", "tts": "오토바이가 다가오고 있습니다."}
            elif highest_risk_label.startswith('Bicycle'):
                json_data = {"label": "Bicycle", "tts": "자전거가 다가오고 있습니다."}
            elif highest_risk_label.startswith('Car'):
                json_data = {"label": "Car", "tts": "차량이 다가오고 있습니다."}
            elif highest_risk_label.startswith('Kickboard'):
                json_data = {"label": "Kickboard", "tts": "킥보드가 다가오고 있습니다."}

            headers = {'Content-Type': 'application/json'}
            response = requests.post(url, json=json_data, headers=headers)  # json= 사용
            print(f'리스폰스: {response.text}')


        if frame is None or frame.size == 0:
            print("유효하지 않은 frame입니다.")
        else:
            cv2.imshow("Risk Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            depth_filename = f"depth_map_{depth_map_saved_count}.png"
            cv2.imwrite(depth_filename, (depth_map * 255).astype(np.uint8))
            depth_map_saved_count += 1
            print(f"Depth map saved as {depth_filename}")

# Calculate and print average size_change_rate for human class
if human_size_change_rates:
    avg_size_change_rate = sum(human_size_change_rates) / len(human_size_change_rates)
    print(f"Average size_change_rate for human class: {avg_size_change_rate:.4f}")
else:
    print(f"No size_change_rate recorded for human class")


# 
# # 크기 변화율 데이터
# size_change_rates = human_size_change_rates
# 
# import matplotlib.pyplot as plt
# import numpy as np
# 
# # 누적 평균 계산을 위해 size_change_rates 데이터 사용
# cumulative_avg = np.cumsum(size_change_rates) / np.arange(1, len(size_change_rates) + 1)
# 
# # 그래프 생성
# plt.figure(figsize=(10, 5))
# plt.plot(cumulative_avg, label="Cumulative Average Size Change Rate", color="purple")
# plt.axhline(0.3, color="red", linestyle="--", label="Average")
# plt.xlabel("Frame", fontsize=14)
# plt.ylabel("Cumulative Average Size Change Rate", fontsize=14)
# plt.title("Cumulative Average Size Change Rate Convergence", fontsize=16)
# plt.legend()
# plt.ylim(0, 1)  # 그래프 y축을 0~1 범위로 설정
# plt.show()




# cap.release()
cv2.destroyAllWindows()