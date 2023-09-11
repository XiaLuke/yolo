from ultralytics import YOLO
import cv2
import cvzone
import math
# 引入sort，使用前提：filterpy==1.4.5;scikit-image==0.19.3;lap==0.4.0
# sort在使用中，可能需要一条线，用于标记物体经过该线时，将其视为已计算
# 物体跟踪问题，需要保证一个物体在连续帧中额能够保持相同的id（在数组中）
from sort import *

cap = cv2.VideoCapture("D:\\File\\Project\\py\\yolov8\\1.mp4")
# model = YOLO("D:\\File\\Project\\py\\models\\yolov8m.pt")
model = YOLO("D:\\File\\Project\\py\\yolov8\\container_best.pt")

#className = ["person","bicycle","car","motorbick","aeroplane","bus","train","truck","boat"]
className = ["container"]

# 遮罩层
canvas = cv2.imread("../canvas.png")

# 创建实例
# 使用sort，三个值表示对象在画面存活多少帧max_age，最少出现多少帧就检测min_hits，最小检测的置信度iou_threshold
sortTracker = Sort(max_age=300,min_hits=3,iou_threshold=0.3)


# 在适当位置定义扎线，当物体超过这条线时进行计数


CurrentCount = 0

while True:
    _, frame = cap.read()

    # 设置遮罩层
    imgRegion = cv2.bitwise_and(frame,canvas)

    results = model(imgRegion, stream=True)

    # 创建一个空数组，用于存放之后的每个对象，数字表示其中的参数数量
    detections = np.empty((0, 5))

    for item in results:
        boxes = item.boxes
        for box in boxes:

            # 计数从哪里开始


            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)

            w, h = x2 - x1, y2 - y1

            cvzone.cornerRect(frame ,(x1, y1, w, h), l=9)

            # 置信度
            conf = math.ceil((box.conf[0] * 100)) / 100

            cls = int(box.cls[0])
            # 如果存在多种类型，可声明只检测部分内容，为声明的内容将不展示置信度和内容名字

            # cvzone.putTextRect(frame, '{} {}'.format(className[cls],conf),
            # (max(0, x1), max(35, y1)), scale=1, thickness=1,offset=6)

            # 将对象的坐标，置信度放入numpy数组中
            currentArray = np.array([x1,y1,x2,y2,conf])
            # 将当前对象放入一个垂直堆栈
            detections = np.vstack((detections,currentArray))


    # 调用sort中的update方法
    results = sortTracker.update(detections)
 
    # for result in results:
    for i in range(len(results)):
        result = results[i]
        x1,y1,x2,y2,Id = result
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w, h = x2-x1,y2-y1
        # cvzone.cornerRect(frame,(x1,y1,w,h),l=9,rt=2,colorR=(255,0,0))
        print(results[0][4])
        CurrentCount = results[0][4]
        # print(type(result[4]))
        cvzone.putTextRect(frame, '{}'.format(int(Id)),
            (max(0, x1), max(35, y1)), scale=2, thickness=3,offset=10)

        cvzone.putTextRect(frame, 'COUNT:{}'.format(CurrentCount),(50,50))


    if frame is not None:
        # 显示图像
        cv2.imshow('Image', frame)
        # 显示遮罩层
        # cv2.imshow("Canvas",imgRegion)
        cv2.waitKey(0)
