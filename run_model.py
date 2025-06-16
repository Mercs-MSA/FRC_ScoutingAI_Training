import cv2
from ultralytics import YOLO

# Load the model (replace with your own model if needed)
model = YOLO('runs2/detect/train/weights/best.pt')  # or 'yolov8s.pt', custom_model.pt, etc.

# Optional: move to GPU for performance
model.to('cuda:0')
print(model.device)

# Open the camera (0 = default camera)
cap = cv2.VideoCapture("/run/media/kevin/frcRecordings/AiTrain/2023/Waco 2023 Q1.mkv")
if not cap.isOpened():
    raise RuntimeError("Failed to open camera.")

# (Optional) Set resolution
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference, set stream=True to avoid blocking
    results = model.predict(frame, imgsz=640, conf=0.5, stream=False, verbose=True, device='cpu')

    # Visualize results
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("YOLOv8 Live", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
