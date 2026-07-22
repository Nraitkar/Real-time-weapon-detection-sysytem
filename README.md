import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("yolov8n.pt")

# Read image
img_path = "test.jpg"   # change to your image path
img = cv2.imread(img_path)

# Run detection
results = model(img, conf=0.25)

# Draw results
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        label = f"Weapon {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

# Show output
cv2.imshow("Weapon Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

model.predict(
    source="test.jpg",   # image / video / folder path
    conf=0.25,
    save=True
)
