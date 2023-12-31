from ultralytics import YOLO

model = YOLO("yolov8l.pt")
# model = YOLO("runs/detect/train7/weights/last.pt")
model.train(data = "basketballDetection-27/data.yaml", 
            epochs = 500, 
            batch = 64, 
            # imgsz = 640, 
            save_period=50,
            # lr0 = 0.001,
            lrf = 0.01,
            workers = 10,
            device = [0, 1],
            patience = 30,
            # resume = True
            mosaic = 0.3
            )