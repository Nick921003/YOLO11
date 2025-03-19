# import cv2
# from ultralytics import YOLO

# # 載入模型
# model = YOLO("runs/detect/train6/weights/best.pt")

# # 讀取影像
# img = cv2.imread("data/images/test/badminton_1.png")
# results = model(img)

# # 繪製標註
# for result in results:
#     result.show()  # 這行會彈出視窗顯示偵測結果
from ultralytics import YOLO
import cv2
import os

# 載入YOLO模型 - 使用您訓練好的權重或預訓練模型
model = YOLO('C:/Users/pjw92/Desktop/Programming/YOLO11/yolo11n.pt')  

# 圖片辨識函數
def detect_image(image_path, save_dir='results'):
    # 確保結果目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 進行辨識
    results = model(image_path)
    
    # 獲取原始圖片用於顯示
    img = cv2.imread(image_path)
    
    # 處理結果
    for result in results:
        boxes = result.boxes  # 獲取偵測到的邊界框
        
        # 在圖片上繪製偵測結果
        for box in boxes:
            # 獲取座標
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 獲取置信度和類別
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = model.names[cls]
            
            # 繪製邊界框和標籤
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{cls_name} {conf:.2f}', (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        result.show()
    # 保存結果
    output_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    
    # 返回結果圖片路徑
    return output_path 

# 影片辨識函數
def detect_video(video_path, save_dir='results'):
    # 確保結果目錄存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 開啟影片
    cap = cv2.VideoCapture(video_path)
    
    # 獲取影片屬性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 設置輸出影片
    output_path = os.path.join(save_dir, f"detected_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # 處理每一幀
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        # 每隔幾幀進行一次偵測可以提高速度
        if frame_count % 1 == 0:  # 可以調整為每隔多少幀檢測一次
            # 進行偵測
            results = model(frame)
            
            # 在幀上繪製偵測結果
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # 獲取座標
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # 獲取置信度和類別
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    cls_name = model.names[cls]
                    
                    # 繪製邊界框和標籤
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{cls_name} {conf:.2f}', (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 顯示處理後的幀（可選）
        cv2.imshow('YOLO Detection', frame)
        
        # 寫入輸出影片
        out.write(frame)
        
        # 按'q'鍵退出
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 釋放資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    return output_path

# 使用範例
if __name__ == "__main__":
    # 圖片辨識
    # image_result = detect_image('C:/Users/pjw92/Desktop/badminton_3.png')
    # print(f"Image detection completed. Result saved to: {image_result}")
    
    # 影片辨識
    video_result = detect_video('C:/Users/pjw92/Desktop/羽球.mp4')
    print(f"Video detection completed. Result saved to: {video_result}")