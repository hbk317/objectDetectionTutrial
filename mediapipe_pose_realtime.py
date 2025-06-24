# https://qiita.com/yuki5130/items/269f227c7b1571529d40#%E3%82%B3%E3%83%BC%E3%83%89 を改変
import os

import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=2,
    min_detection_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils
# mp_drawing._VISIBILITY_THRESHOLD = 0  # 描画するvisibilityの閾値

# 動画キャプチャの初期化
cap = cv2.VideoCapture(0)  # PCの内カメラから画像を取得

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
resize_scale = 1.0

# 保存する動画の設定
output_filename = "result/realtime/mediapipe_pose.mp4"
if not os.path.exists("result/realtime"):
    os.makedirs("result/realtime")
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

    # フレームの高さと幅を取得
    height, width, _ = frame.shape

    # フレームサイズを縮小
    small_frame = cv2.resize(frame, (int(width * resize_scale), int(height * resize_scale)))

    # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Mediapipeで骨格検出を実行
    result = pose.process(frame_rgb)

    # 検出結果を描画
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            small_frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # 縮小されたフレームを保存
    out.write(small_frame)

    # 縮小されたフレームを表示
    cv2.imshow('Pose Detection', small_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()  # 保存用のVideoWriterを解放
cv2.destroyAllWindows()
print(f"保存された動画ファイル: {output_filename}")
