from ultralytics import YOLO
import cv2
import numpy as np

# 画像の読み込み
image = cv2.imread('ex2.jpg')
# 画像読み込み確認
if image is not None:
    print("画像が正常に読み込まれました。")
else:
    print("画像の読み込みに失敗しました。")

#HSV色空間へ変換
Himg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# YOLOモデルの読み込み
model = YOLO("yolov8x.pt")

# 人物検出
results = model(Himg, save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes

#青色の閾値
lower = np.array([90,64,0])
upper = np.array([150,255,255])

position = []
times = len(results[0].boxes)

#物体の座標抽出
position = [[int(x1), int(y1), int(x2), int(y2)] for x1, y1, x2, y2, _, _ in boxes.data]

#監督のy座標を求める
notPlayer_y = -1
for _,  y1, _, _ in position:
    if y1 > notPlayer_y:
        notPlayer_y = y1
        
#青ユニフォームを認識
for x1, y1, x2, y2 in position:
    crip = Himg[y1:y2, x1:x2]
    mask = cv2.inRange(crip, lower, upper)
    #青が含まれている割合を求める
    brate = cv2.countNonZero(mask)/(crip.shape[0]*crip.shape[1])
    if (brate > 0 and y1 < notPlayer_y):
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

# 結果を表示
cv2.imshow('ex2.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
