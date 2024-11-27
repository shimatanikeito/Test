from ultralytics import YOLO

model = YOLO("yolov8x-pose.pt")

results = model("https://ultralytics.com/images/bus.jpg", save=True, save_txt=True, save_conf=True)
keypoints = results[0].keypoints
print(keypoints.data)