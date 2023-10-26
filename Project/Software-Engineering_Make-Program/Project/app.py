import os
from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

# YOLOv4 설정 파일과 가중치 파일 경로
yolo_cfg = 'yolov4.cfg'
yolo_weights = 'yolov4.weights'
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# YOLO 클래스 이름 파일 경로
classes_file = 'coco.names'
with open(classes_file, 'r') as f:
    classes = f.read().strip().split('\n')

def load_and_analyze_image(image_path):
    # 이미지 열기
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    # YOLOv4 입력 이미지에 대한 전처리
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 이미지에 객체 그리기
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

@app.route('/', methods=['GET', 'POST'])
def upload_and_analyze():
    if request.method == 'POST':
        # 업로드된 파일 저장
        image = request.files['image']
        image.save('static/uploaded_image.jpg')  # 이미지를 'static' 폴더에 저장

        # 이미지 분석
        result_image = load_and_analyze_image('static/uploaded_image.jpg')
        
        # 분석 결과 이미지를 'static' 폴더에 저장
        cv2.imwrite('static/result.jpg', result_image)
        
        return render_template('result.html', image='result.jpg')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
