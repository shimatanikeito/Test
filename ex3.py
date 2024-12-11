import cv2
from ultralytics import YOLO
import numpy as np
import math

# YOLOv8モデルをロード
model = YOLO("yolov8x-pose.pt")

# ビデオファイルを開く
video_path = "ex3b.mp4"
cap = cv2.VideoCapture(video_path)

#骨格座標用の配列初期化
p = []
def position(x, y):
    p.append([int(x), int(y)])
    return 0

#角度を求める関数
def cos_formula(p1, p2, p3):
    a = math.dist(p1, p2)
    b = math.dist(p1, p3)
    c = math.dist(p2, p3)
    cos_c = (a**2 + b**2 - c**2)/(2*a*b)
    acos_c = math.acos(cos_c)
    return math.degrees(acos_c)

# ビデオフレームをループする
while cap.isOpened():
    # ビデオからフレームを読み込む
    success, frame = cap.read()

    if success:
        #Keypointsを取得
        results = model(frame, save=True, save_txt=True, save_conf=True)
        keypoints = results[0].keypoints

        #骨格データを座標配列pに格納
        p = []
        for x, y, _ in keypoints.data[0]:
            position(x, y)

        #ボーン描画
        skeleton = np.array([
			[16,14],[14,12],[15,13],[13,11],[12,11],[12,6],[11,5],
			[6,5],[6,8],[5,7],[8,10],[7,9]
            ])
        for a, b in skeleton:
            cv2.line(frame,
            tuple(p[a]),
            tuple(p[b]),
            (255, 0, 0),
            thickness=3
            )

        angle = cos_formula(p[6], p[8], p[12])
        #角度判定（赤色に変化）
        if(80 <= angle <= 100):
            cv2.line(frame, 
                tuple(p[6]), 
                tuple(p[8]), 
                (0, 0, 255), 
                thickness=3
            )
            cv2.line(frame, 
                tuple(p[8]), 
                tuple(p[10]), 
                (0, 0, 255), 
                thickness=3
            )

        #骨格の点描画
        for i, j in  enumerate(p):
            if (i > 4):
                cv2.circle(frame,
                tuple(j),
                10,
                (0,225,255),
                -1
                )

        # 注釈付きのフレームを表示
        cv2.imshow("YOLOv8トラッキング", frame)

        # 'q'が押されたらループから抜ける
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # ビデオの終わりに到達したらループから抜ける
        break
cap.release()
cv2.destroyAllWindows()