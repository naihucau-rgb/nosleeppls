import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Tắt log TensorFlow
os.environ['GLOG_minloglevel'] = '3'       # Tắt log hệ thống GLOG

import cv2
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import math

class Thread1_DataCollection(threading.Thread):
    def __init__(self, data_queue, camera_index=0):
        super().__init__()
        self.data_queue = data_queue
        self.camera_index = camera_index
        self.running = True
        
        # Khởi tạo MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Lấy thêm điểm chi tiết ở mắt/môi
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Chỉ mục các điểm mốc (Landmarks) của MediaPipe
        # Mắt trái và Mắt phải
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Môi (trên, dưới, trái, phải)
        self.LIP_TOP = 13
        self.LIP_BOTTOM = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        
        
        # Tự động nhận diện: Nếu truyền vào string ("video.mp4") -> Là file video
        # Nếu truyền vào số (0, 1, 2...) -> Là webcam/điện thoại
        is_video_file = isinstance(self.camera_index, str)
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                if is_video_file:
                    # Nếu là video có sẵn -> Tua lại từ đầu để lặp vô hạn
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    # Nếu là webcam trực tiếp -> Mất kết nối thì thoát vòng lặp
                    print("⚠️ Mất kết nối với Camera điện thoại!")
                    break

            # Tối ưu hiệu suất: chuyển màu sang RGB để đưa vào mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True

            img_h, img_w, _ = frame.shape

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Lấy tọa độ 2D của 468 điểm
                    landmarks_2d = []
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        landmarks_2d.append((x, y))

                    # 1. Tính EAR (Eye Aspect Ratio) trung bình 2 mắt
                    left_ear = self.calculate_ear(landmarks_2d, self.LEFT_EYE)
                    right_ear = self.calculate_ear(landmarks_2d, self.RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # 2. Tính MAR (Mouth Aspect Ratio)
                    mar = self.calculate_mar(landmarks_2d)

                    # 3. Tính Góc Cúi (Pitch) bằng thuật toán SolvePnP
                    pitch = self.calculate_head_pitch(face_landmarks.landmark, img_w, img_h)

                    # Đóng gói dữ liệu thô
                    raw_data = {
                        "EAR": round(avg_ear, 3),
                        "MAR": round(mar, 3),
                        "PITCH": round(pitch, 2),
                        "timestamp": time.time(),
                        "frame": frame 
                    }

                    if self.data_queue.full():
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.data_queue.put(raw_data)

        cap.release()
        self.face_mesh.close()

    def stop(self):
        """Hàm gọi từ bên ngoài để dừng luồng an toàn"""
        self.running = False

    # --- CÁC HÀM TÍNH TOÁN TOÁN HỌC ---

    def euclidean_distance(self, p1, p2):
        return math.dist(p1, p2)

    def calculate_ear(self, landmarks, eye_indices):
        """Tính Eye Aspect Ratio dựa trên 6 điểm của mắt"""
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        # Công thức EAR
        vertical_1 = self.euclidean_distance(p2, p6)
        vertical_2 = self.euclidean_distance(p3, p5)
        horizontal = self.euclidean_distance(p1, p4)
        
        if horizontal == 0: return 0.0
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def calculate_mar(self, landmarks):
        """Tính Mouth Aspect Ratio"""
        top_lip = landmarks[self.LIP_TOP]
        bottom_lip = landmarks[self.LIP_BOTTOM]
        left_mouth = landmarks[self.MOUTH_LEFT]
        right_mouth = landmarks[self.MOUTH_RIGHT]

        vertical = self.euclidean_distance(top_lip, bottom_lip)
        horizontal = self.euclidean_distance(left_mouth, right_mouth)
        
        if horizontal == 0: return 0.0
        mar = vertical / horizontal
        return mar

    def calculate_head_pitch(self, landmarks, img_w, img_h):
        """
        Sử dụng cv2.solvePnP để chuyển đổi các điểm 2D sang 3D
        và lấy ra góc cúi (Pitch) của đầu.
        """
        # Lấy 6 điểm mốc chuẩn trên mặt: Chóp mũi, Cằm, Đuôi mắt trái, Đuôi mắt phải, Mép trái, Mép phải
        face_2d = []
        face_3d = []
        
        for idx, lm in enumerate(landmarks):
            if idx in [1, 152, 226, 259, 57, 287]:
                if idx == 1: nose_2d = (lm.x * img_w, lm.y * img_h)
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([lm.x, lm.y, lm.z])
                
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Trọng tâm camera (Camera Matrix) giả định
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Tính toán ma trận xoay
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        # Chuyển đổi vector xoay thành các góc Euler
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Cắt góc Pitch (trục X), biến đổi hệ số cho dễ hiểu
        pitch = angles[0] * 360
        return pitch

import cv2        
import mediapipe as mp
import numpy as np
import threading
import queue
import time
import math

class Thread1_DataCollection(threading.Thread):
    def __init__(self, data_queue, camera_index=0):
        super().__init__()
        self.data_queue = data_queue
        self.camera_index = camera_index
        self.running = True
        
        # Khởi tạo MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True, # Lấy thêm điểm chi tiết ở mắt/môi
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Chỉ mục các điểm mốc (Landmarks) của MediaPipe
        # Mắt trái và Mắt phải
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Môi (trên, dưới, trái, phải)
        self.LIP_TOP = 13
        self.LIP_BOTTOM = 14
        self.MOUTH_LEFT = 78
        self.MOUTH_RIGHT = 308

    def run(self):
        cap = cv2.VideoCapture(self.camera_index)
        
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            # Tối ưu hiệu suất: chuyển màu sang RGB để đưa vào mediapipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = self.face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True

            img_h, img_w, _ = frame.shape

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Lấy tọa độ 2D của 468 điểm
                    landmarks_2d = []
                    for lm in face_landmarks.landmark:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        landmarks_2d.append((x, y))

                    # 1. Tính EAR (Eye Aspect Ratio) trung bình 2 mắt
                    left_ear = self.calculate_ear(landmarks_2d, self.LEFT_EYE)
                    right_ear = self.calculate_ear(landmarks_2d, self.RIGHT_EYE)
                    avg_ear = (left_ear + right_ear) / 2.0

                    # 2. Tính MAR (Mouth Aspect Ratio)
                    mar = self.calculate_mar(landmarks_2d)

                    # 3. Tính Góc Cúi (Pitch) bằng thuật toán SolvePnP
                    pitch = self.calculate_head_pitch(face_landmarks.landmark, img_w, img_h)

                    # Đóng gói dữ liệu thô
                    raw_data = {
                        "EAR": round(avg_ear, 3),
                        "MAR": round(mar, 3),
                        "PITCH": round(pitch, 2),
                        "timestamp": time.time(),
                        # Gửi kèm frame để Luồng 3 (Giao diện) dùng để vẽ lên Tkinter
                        "frame": frame 
                    }

                    # Ném vào Queue cho Luồng 2 và 3 xử lý
                    # Nếu queue đầy (chưa kịp xử lý), loại bỏ frame cũ nhất để lấy real-time
                    if self.data_queue.full():
                        try:
                            self.data_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.data_queue.put(raw_data)

        cap.release()
        self.face_mesh.close()

    def stop(self):
        """Hàm gọi từ bên ngoài để dừng luồng an toàn"""
        self.running = False

    # --- CÁC HÀM TÍNH TOÁN TOÁN HỌC ---

    def euclidean_distance(self, p1, p2):
        return math.dist(p1, p2)

    def calculate_ear(self, landmarks, eye_indices):
        """Tính Eye Aspect Ratio dựa trên 6 điểm của mắt"""
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
        # Công thức EAR
        vertical_1 = self.euclidean_distance(p2, p6)
        vertical_2 = self.euclidean_distance(p3, p5)
        horizontal = self.euclidean_distance(p1, p4)
        
        if horizontal == 0: return 0.0
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    def calculate_mar(self, landmarks):
        """Tính Mouth Aspect Ratio"""
        top_lip = landmarks[self.LIP_TOP]
        bottom_lip = landmarks[self.LIP_BOTTOM]
        left_mouth = landmarks[self.MOUTH_LEFT]
        right_mouth = landmarks[self.MOUTH_RIGHT]

        vertical = self.euclidean_distance(top_lip, bottom_lip)
        horizontal = self.euclidean_distance(left_mouth, right_mouth)
        
        if horizontal == 0: return 0.0
        mar = vertical / horizontal
        return mar

    def calculate_head_pitch(self, landmarks, img_w, img_h):
        """
        Sử dụng cv2.solvePnP để chuyển đổi các điểm 2D sang 3D
        và lấy ra góc cúi (Pitch) của đầu.
        """
        # Lấy 6 điểm mốc chuẩn trên mặt: Chóp mũi, Cằm, Đuôi mắt trái, Đuôi mắt phải, Mép trái, Mép phải
        face_2d = []
        face_3d = []
        
        for idx, lm in enumerate(landmarks):
            if idx in [1, 152, 226, 259, 57, 287]:
                if idx == 1: nose_2d = (lm.x * img_w, lm.y * img_h)
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])
                face_3d.append([lm.x, lm.y, lm.z])
                
        face_2d = np.array(face_2d, dtype=np.float64)
        face_3d = np.array(face_3d, dtype=np.float64)

        # Trọng tâm camera (Camera Matrix) giả định
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)

        # Tính toán ma trận xoay
        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
        
        # Chuyển đổi vector xoay thành các góc Euler
        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        # Cắt góc Pitch (trục X), biến đổi hệ số cho dễ hiểu
        pitch = angles[0] * 360
        return pitch