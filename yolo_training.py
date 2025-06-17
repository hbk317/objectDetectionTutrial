import os

from ultralytics import YOLO


project = f"runs/detect/n"
model = YOLO("yolov8n.pt")
model.train(data="data/dataset/training.yaml",
            epochs=100,
            batch=2,
            imgsz=(1920, 1088),
            project=project,  # 学習結果を保存するディレクトリを指定
            name="train_hagiharaDataOnly"  # これを設定しないと，projectで指定したディレクトリに，今回の学習結果がtrain*として保存される．同じproject内で複数回学習させる場合は区別が付く名前にした方が良いかも
            )
metrics = model.val(project=project, name="val_hagiharaDataOnly")
