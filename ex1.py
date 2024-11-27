import cv2
from ultralytics import YOLO
import numpy as np

#画像の読み込み
image = cv2.imread('ex1.jpg')
#画像読み込み確認
if image is not None:
    print("画像が正常に読み込まれました。")
else:
    print("画像の読み込みに失敗しました。")

#YOLOのモデル選択
model = YOLO("yolov8x-pose.pt")

#選択したモデルに画像を突っ込み、結果を格納
results = model("ex1.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data)

#骨格のペアラベルの配列
skeleton = np.array([
			[16,14],[14,12],[15,13],[13,11],[12,11],[12,6],[11,5],
			[6,5],[6,8],[5,7],[8,10],[7,9]
            ])

#線を描く
for i in range(len(skeleton)):
    cv2.line(image,
    (int(keypoints.data[0][skeleton[i][0]][0]), int(keypoints.data[0][skeleton[i][0]][1])),
    (int(keypoints.data[0][skeleton[i][1]][0]), int(keypoints.data[0][skeleton[i][1]][1])),
    (0, 0, 255),
    thickness=3
    )

#点を描く
for i in range(keypoints.data[0].size(0)-5):
    cv2.circle(
        image,
        (int(keypoints.data[0][i+5][0]), int(keypoints.data[0][i+5][1])),
        10,
        (0,225,255),
        -1
    )

#openCVで画像表示
cv2.imshow('ex1.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

