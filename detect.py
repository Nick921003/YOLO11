import cv2
from ultralytics import YOLO

# 載入模型
model = YOLO("runs/detect/train6/weights/best.pt")

# 讀取影像
img = cv2.imread("data/images/test/badminton_1.png")
results = model(img)

# 繪製標註
for result in results:
    result.show()  # 這行會彈出視窗顯示偵測結果
