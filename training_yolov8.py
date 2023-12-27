from ultralytics import YOLO

model = YOLO("yolov8l.pt")
# model = YOLO("runs/detect/train7/weights/last.pt")
model.train(data = "basketballDetection-24/data.yaml", 
            epochs = 200, 
            batch = 64, 
            imgsz = 720, 
            save_period=20,
            lr0 = 0.001,
            lrf = 0.01,
            workers = 4,
            #device = [0, 1],
            patience = 30,
            # resume = True
            )