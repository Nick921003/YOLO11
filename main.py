from ultralytics import YOLO
from multiprocessing import freeze_support

# 確保多進程運算中不重複執行
if __name__ == '__main__':
    freeze_support()
# 載入 YOLOv11 預訓練模型
    model = YOLO("yolo11n.pt")  

# 訓練模型
    model.train(
        data="data.yaml",  # 數據集配置
        epochs=50,         # 訓練回合數（根據數據集大小調整）
        batch=8,           # 批次大小（依據顯存大小調整）
        imgsz=640,         # 圖片尺寸
        workers=2,         # 資料加載的線程數
        device="cuda"      # 使用 GPU 訓練（如果沒有 GPU，改成 "cpu"）
    )

# 訓練結果會儲存在 `runs/detect/train/`
