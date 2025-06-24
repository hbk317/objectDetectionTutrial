# https://qiita.com/yuki5130/items/269f227c7b1571529d40#%E3%82%B3%E3%83%BC%E3%83%89 を改変
import os

import cv2
from retinaface import RetinaFace


# retinafaceの推論結果を描画する関数
def draw_result(image, results):
    facial_area, right_eye, left_eye, nose, mouth_right, mouth_left = results
    cv2.rectangle(image, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (255, 255, 255), thickness=5)
    cv2.circle(image, (int(right_eye[0]), int(right_eye[1])), radius=5, color=(255, 255, 255), thickness=-1)  # thickness=-1の場合塗りつぶす
    cv2.circle(image, (int(left_eye[0]), int(left_eye[1])), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (int(nose[0]), int(nose[1])), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (int(mouth_right[0]), int(mouth_right[1])), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(image, (int(mouth_left[0]), int(mouth_left[1])), radius=5, color=(255, 255, 255), thickness=-1)
    return image


# 動画キャプチャの初期化
cap = cv2.VideoCapture(0)  # PCの内カメラから画像を取得

if not cap.isOpened():
    print("Error: カメラまたは動画を開けませんでした。")
    exit()

# ウィンドウサイズを変更するスケール（例: 0.5 で半分の大きさ）
resize_scale = 1.0

# 保存する動画の設定
output_filename = "result/realtime/retinaface.mp4"
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

    # BGRからRGBに変換（Mediapipeが必要とするフォーマット）
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # retinafaceで骨格検出を実行
    results = RetinaFace.detect_faces(frame_rgb)

    # 顔が複数検出された場合は全て描画
    for key in list(results.keys()):
        identity = results[key]
        facial_area = identity["facial_area"]
        if not facial_area:
            continue
        right_eye = identity["landmarks"]["right_eye"]
        left_eye = identity["landmarks"]["left_eye"]
        nose = identity["landmarks"]["nose"]
        mouth_right = identity["landmarks"]["mouth_right"]
        mouth_left = identity["landmarks"]["mouth_left"]
        landmarks = (facial_area, right_eye, left_eye, nose, mouth_right, mouth_left)

        # 検出結果を描画
        frame = draw_result(frame, landmarks) if landmarks else frame

    # フレームを保存
    out.write(frame)

    # フレームを表示
    cv2.imshow('Pose Detection', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
out.release()  # 保存用のVideoWriterを解放
cv2.destroyAllWindows()
print(f"保存された動画ファイル: {output_filename}")
