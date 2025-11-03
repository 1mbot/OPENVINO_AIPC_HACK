import cv2
from ultralytics import YOLO
import time
import os

# Suppress warnings (optional)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Set the capture interval (in seconds)
CAPTURE_INTERVAL = 1 

print("[INFO] Loading custom YOLOv8 model (best.pt)...")
try:
    model = YOLO("best.pt")
    print("[INFO] ✅ Model loaded successfully")
except Exception as e:
    print(f"[ERROR] ❌ Failed to load 'best.pt'. Make sure it's in the same folder.\n{e}")
    exit()

print("[INFO] Initializing webcam...")
try:
    cam = cv2.VideoCapture(0) # 0 is usually the built-in webcam
    if not cam.isOpened():
        raise Exception("Could not open webcam.")
    print("[INFO] ✅ Webcam initialized. Press 'q' in the video window to quit.")
except Exception as e:
    print(f"[ERROR] ❌ {e}")
    exit()

# === Main Loop ===
try:
    while True:
        # 1. Capture Image
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] ❌ Failed to capture frame from webcam.")
            break

        # 2. Run Inference
        # We give the raw frame directly to the model
        results = model(frame, conf=0.25)
        
        detected_objects = []
        face_result = "No Face Detected"

        # 3. Process Results
        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                # Get the text label (e.g., "Car", "Human face")
                label = model.names[int(cls)]
                
                # Separate face from other objects
                if label == "Human face":
                    face_result = "Human face"
                    color = (0, 255, 0) # Green for face
                else:
                    detected_objects.append(label)
                    color = (255, 0, 0) # Blue for objects

                # 4. Draw Boxes on the Frame
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. Print Summary to Terminal
        response = f"{face_result}. Objects detected: {', '.join(detected_objects) if detected_objects else 'None'}"
        print(f"[RESULT] 📥 {response}")
        
        # 6. Show the Live Video Window
        cv2.imshow("Live Detection (Press 'q' to quit)", frame)

        # 7. Handle Quit Key and Interval
        # cv2.waitKey(1) is needed to refresh the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 🛑 'q' pressed. Stopping...")
            break
        
        # Wait for the next interval
        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\n[INFO] 🛑 Stopping...")
finally:
    # 8. Clean up
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] 🔌 Disconnected and windows closed.")