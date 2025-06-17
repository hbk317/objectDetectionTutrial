# https://qiita.com/yuki5130/items/269f227c7b1571529d40#%E3%82%B3%E3%83%BC%E3%83%89 を改変
import os
import cv2

from ultralytics import YOLO


# 動画キャプチャの初期化
cap = cv2.VideoCapture(0)  # PCの内カメラから画像を取得

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
resize_scale = 1.0

# 推論に用いるモデルを読み込む
model_path = "yolov8n.pt"  # 事前学習時点のモデル
# model_path = "runs/detect/n/train_includeSubjectsData/weights/best.pt"  # ぬいぐるみしか検出できないので使わない
model = YOLO(model_path)

# 保存する動画の設定
output_filename = "result/yolo_realtime.mp4"
if not os.path.exists("result"):
    os.makedirs("result")

fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * resize_scale)  # 縮小後の幅
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * resize_scale)  # 縮小後の高さ
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 動画のコーデック

out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: フレームを取得できませんでした。")
        break

    # Mediapipeで骨格検出を実行
    results = model.track(frame, persist=True)

    # 検出結果を描画
    frame = results[0].plot()

    # 縮小されたフレームを保存
    out.write(frame)

    # 縮小されたフレームを表示
    cv2.imshow('Object Detection', frame)

    # 映像が表示されているウィンドウを選んだ状態で'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()  # 保存用のVideoWriterを解放
cv2.destroyAllWindows()
print(f"保存された動画ファイル: {output_filename}")
