import cv2
from ultralytics import YOLO
import time
import os
import pyttsx3
from collections import defaultdict

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

# === Text-to-Speech (using pyttsx3) ===
def speak(text):
    """
    Speaks the given text by creating a NEW engine instance.
    This is a robust, blocking call.
    """
    try:
        engine = pyttsx3.init('sapi5')
        engine.say(text)
        engine.runAndWait() # Blocks until speaking is finished
        engine.stop()
    except Exception as e:
        print(f"[TTS] ❌ pyttsx3 Error: {e}")
# === END OF NEW FUNCTION ===


# === Main Loop ===
try:
    # Keep track of the last thing we said
    last_spoken_response = None 

    while True:
        # 1. Capture Image
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] ❌ Failed to capture frame from webcam.")
            break

        # 2. Run Inference
        # === ⬇️ MODIFIED: Lowered confidence, set image size ⬇️ ===
        results = model(frame, conf=0.15, imgsz=640)
        # === ⬆️ END OF MODIFIED SECTION ⬆️ ===
        
        object_counts = defaultdict(int)

        # 3. Process Results
        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                label = model.names[int(cls)]
                
                object_counts[label] += 1
                
                # 4. Draw Boxes on the Frame (Color-coded)
                if label == "Human face":
                    color = (0, 255, 0) # Green for face
                else:
                    color = (255, 0, 0) # Blue for objects

                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                # Display the label on the box
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 5. Create Response String
        
        # Handle faces first
        face_count = object_counts.get("Human face", 0)
        if face_count == 0:
            face_result_str = "No Face Detected"
        elif face_count == 1:
            face_result_str = "1 Human face"
        else:
            face_result_str = f"{face_count} Human faces" # Handle plural
            
        # Handle other objects
        other_objects_list = []
        for label, count in object_counts.items():
            if label == "Human face":
                continue # Skip, we already handled it

            plural = "s" if count > 1 else ""
            if label.endswith('s') or label.endswith('x') or label.endswith('ch'):
                 plural = "es"

            other_objects_list.append(f"{count} {label}{plural}")

        # Combine the strings
        other_objects_str = ", ".join(other_objects_list) if other_objects_list else "None"
        response = f"{face_result_str}. Objects detected: {other_objects_str}"

        print(f"[RESULT] 📥 {response}")
        
        # 6. Show the Live Video Window
        cv2.imshow("Live Detection (Press 'q' to quit)", frame)

        # 7. Handle Quit Key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] 🛑 'q' pressed. Stopping...")
            break
        
        # 8. Speak the Result ONLY IF IT'S NEW
        if response != last_spoken_response:
            speak(response)
            last_spoken_response = response # Update the tracker
        
        # 9. Sleep for the next interval
        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    print("\n[INFO] 🛑 Stopping...")
finally:
    # 10. Clean up
    cam.release()
    cv2.destroyAllWindows()
    print("[INFO] 🔌 Disconnected and windows closed.")