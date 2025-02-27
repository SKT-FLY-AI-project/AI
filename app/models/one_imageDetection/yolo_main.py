from ultralytics import YOLO
import cv2

# YOLOv8 모델 로드
model = YOLO("yolov8n.pt")


# COCO 데이터셋 클래스 확인
print(model.names)  # YOLO가 학습한 클래스 목록 출력


# # COCO 데이터셋에서 "그림"과 관련된 클래스 ID (예제, 정확한 ID 확인 필요)
# PAINTING_CLASS_ID = 85  # 정확한 ID는 YOLO의 클래스 리스트에서 확인 필요

# def detect_painting(image_path):
#     results = model(image_path)

#     # 원본 이미지 로드
#     image = cv2.imread(image_path)

#     for r in results:
#         for box in r.boxes.data.tolist():
#             x1, y1, x2, y2, score, class_id = box
#             if class_id == PAINTING_CLASS_ID and score > 0.5:
#                 # 그림인 경우 사각형 그리기
#                 cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

#     # 결과 이미지 표시
#     cv2.imshow("Detected Paintings", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # 테스트 실행
# detect_painting(r"app/models/one_imageDetection/GardenatSainte-Adresse_monet.png")