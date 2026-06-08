import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Tắt log TensorFlow
os.environ['GLOG_minloglevel'] = '3'       # Tắt log hệ thống GLOG
os.environ['ABSL_MIN_LOG_LEVEL'] = '3'     # Tắt log Abseil
import queue
import cv2
import time  # Thêm thư viện time để làm bộ đếm Timeout
from luong1 import Thread1_DataCollection 

if __name__ == "__main__":
    # Khởi tạo Queue và Luồng 1
    q = queue.Queue(maxsize=10)
    
    # Chỉ số camera để test (Thử đổi thành 0, 1, 2... để dò đúng OBS)
    CAMERA_INDEX = 0 
    
    thread_1 = Thread1_DataCollection(data_queue=q, camera_index=CAMERA_INDEX)
    thread_1.start()
    
    print(f"🔄 Đang kết nối tới Camera/OBS ở index: {CAMERA_INDEX}...")
    print("⏳ Đang đợi tín hiệu hình ảnh (Tối đa 6 giây)...")
    
    window_name = "Test Luong 1 - Visual Debugging"
    
    # Cấu hình bộ đếm lỗi kết nối
    start_time = time.time()
    has_received_frame = False 

    try:
        while True:
            # Nếu quá 6 giây từ lúc chạy mà Queue vẫn trống rỗng
            # Chứng tỏ hệ thống nhận diện được Driver nhưng bạn đang TẮT Virtual Camera trên OBS
            if not has_received_frame and (time.time() - start_time > 6.0):
                print("\n" + "="*60)
                print("❌ LỖI KHÔNG NHẬN ĐƯỢC HÌNH ẢNH!")
                print(f"👉 Nguyên nhân: Máy tính nhận được thiết bị số {CAMERA_INDEX} nhưng không có hình.")
                print("   Chắc chắn bạn chưa bấm nút 'Start Virtual Camera' trên phần mềm OBS!")
                print("👉 Khắc phục: Hãy bật Virtual Camera trên OBS lên rồi chạy lại file test này nhé.")
                print("="*60 + "\n")
                break # Chủ động thoát chương trình một cách an toàn
                
            # KIỂM TRA: Nếu Luồng 1 tự sập hoàn toàn (do sai index phần cứng) -> Dừng file test ngay
            if not thread_1.is_alive() and q.empty():
                print("\n[HỆ THỐNG] Luồng 1 đã tự đóng do không kết nối được Camera.")
                break
                
            # Liên tục lấy dữ liệu từ Queue
            if not q.empty():
                if not has_received_frame:
                    print("🎉 Kết nối thành công! Đang hiển thị camera...")
                    has_received_frame = True # Xác nhận đã thông luồng hình ảnh thành công
                
                data = q.get()
                frame = data["frame"]
                ear = data["EAR"]
                mar = data["MAR"]
                pitch = data["PITCH"]
                
                # --- GIẢI PHÁP CHỐNG BÓP HÌNH CHO CỬA SỔ TEST ---
                h, w, _ = frame.shape
                aspect_ratio = w / h  
                
                target_width = 480
                target_height = int(target_width / aspect_ratio) 
                
                # Tạo bản sao thu nhỏ để hiển thị
                display_frame = cv2.resize(frame, (target_width, target_height))
                
                # --- PHẦN VẼ LÊN MÀN HÌNH (Đã sửa: Vẽ trực tiếp lên display_frame) ---
                ear_color = (0, 0, 255) if ear < 0.2 else (0, 255, 0)
                
                # Điều chỉnh kích thước chữ (scale từ 1 xuống 0.6) để vừa vặn với màn hình 480p
                cv2.putText(display_frame, f"EAR (Mat): {ear}", (20, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
                            
                cv2.putText(display_frame, f"MAR (Mieng): {mar}", (20, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                            
                cv2.putText(display_frame, f"Pitch (Cui): {pitch}", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                
                cv2.putText(display_frame, 'Press "Esc" to close this window', (20, display_frame.shape[0] - 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Hiển thị cửa sổ camera bằng ảnh đã xử lý chống bóp hình
                cv2.imshow(window_name, display_frame)
                
            # Đưa lệnh waitKey ra rìa vòng lặp giúp cửa sổ OpenCV mượt mà, không bị hiện tượng "Not Responding"
            key = cv2.waitKey(1) & 0xFF
            if key == 27: # Nhấn phím Esc để thoát
                break
                
            # Kiểm tra nếu người dùng dùng chuột bấm nút [X] màu đỏ để tắt cửa sổ
            if has_received_frame and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
                
            time.sleep(0.001) # Khóa nhẹ vòng lặp để bảo vệ CPU không bị quá tải 100%
                    
    except KeyboardInterrupt:
        print("Đã nhận lệnh ngắt từ bàn phím...")
    finally:
        print("Đang dừng hệ thống...")
        thread_1.stop()
        thread_1.join()
        cv2.destroyAllWindows()
        print("Tắt thành công!")