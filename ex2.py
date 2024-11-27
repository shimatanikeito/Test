from ultralytics import YOLO
import cv2
import numpy as np

#画像の読み込み
image = cv2.imread('ex2.jpg')
#画像読み込み確認
if image is not None:
    print("画像が正常に読み込まれました。")
else:
    print("画像の読み込みに失敗しました。")
model = YOLO("yolov8x.pt")

results = model("ex2.jpg", save=True, save_txt=True, save_conf=True)
boxes = results[0].boxes
for box in boxes:
    print(box.data)
print(results[0].names)
print(results[0].boxes[0].data[0][2])

for i in range(len(results[0].boxes)):
    cv2.rectangle(image,
    (int(results[0].boxes[i].data[0][0]), int(results[0].boxes[i].data[0][1])),
    (int(results[0].boxes[i].data[0][2]), int(results[0].boxes[i].data[0][3])),
    (0, 0, 255),
    thickness=2)
#openCVで画像表示
cv2.imshow('ex2.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()