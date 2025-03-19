from ultralytics import YOLO
from multiprocessing import freeze_support

# 確保多進程運算中不重複執行
if __name__ == '__main__':
    freeze_support()
    
    # 載入 YOLOv11 預訓練模型
    model = YOLO("yolo11n.pt")  # 修正模型名稱
    
    # 訓練模型
    model.train(
        data="data.yaml",      # 數據集配置
        epochs=50,             # 訓練回合數
        batch=8,               # 批次大小
        imgsz=640,             # 圖片尺寸
        workers=4,             # 增加資料加載的線程數
        device="cuda",         # 使用 GPU 訓練
        save=True,             # 保存最佳模型
    )
    
    # 訓練完成後進行模型評估
    metrics = model.val()
    print(f"Model validation metrics: {metrics}")