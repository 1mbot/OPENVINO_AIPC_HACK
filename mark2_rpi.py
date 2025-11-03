import os
import time
import pickle
from collections import defaultdict
import numpy as np
import cv2
import pyttsx3
from ultralytics import YOLO
from deepface import DeepFace

# ----------------- CONFIG -----------------
CAPTURE_INTERVAL = 1.0
KNOWN_FACES_FILE = "known_faces.pkl"
RECOGNITION_THRESHOLD = 0.40
YOLO_MODEL_PATH = "best.pt"
MODEL_NAME = "VGG-Face"
WEBCAM_INDEX = 0
SPEAK_ENABLED = True
VERBOSE = True
# ------------------------------------------

def log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

# ----------------- YOLO LOAD -----------------
log("[INFO] Loading YOLO model...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    log("[INFO] ✅ YOLO model loaded successfully.")
except Exception as e:
    log(f"[ERROR] Could not load YOLO model ({YOLO_MODEL_PATH}): {e}")
    raise SystemExit(1)

# ----------------- KNOWN FACES -----------------
def load_known_faces(path=KNOWN_FACES_FILE):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            converted = {name: [np.array(e, dtype=np.float32) for e in embeds] for name, embeds in data.items()}
            log(f"[INFO] ✅ Loaded {len(converted)} known individuals.")
            return converted
        except Exception as e:
            log(f"[WARN] Failed to load known faces ({e}). Starting new.")
            return {}
    else:
        log("[INFO] No known_faces.pkl found. Starting fresh.")
        return {}

def save_known_faces(faces, path=KNOWN_FACES_FILE):
    data = {name: [emb.tolist() for emb in embeds] for name, embeds in faces.items()}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    log("[INFO] ✅ Known faces saved to disk.")

known_faces = load_known_faces()

# ----------------- RECOGNITION HELPERS -----------------
def euclidean_distance(a, b):
    if a.shape != b.shape:
        log(f"[WARN] Skipping mismatched embeddings: {a.shape} vs {b.shape}")
        return float("inf")
    return np.linalg.norm(a - b)

def recognize_face(embedding):
    if not known_faces:
        return None, None
    best_name, best_dist = None, float("inf")
    for name, embeds in known_faces.items():
        for e in embeds:
            dist = euclidean_distance(embedding, e)
            if dist < best_dist:
                best_dist = dist
                best_name = name
    if best_dist < RECOGNITION_THRESHOLD:
        return best_name, best_dist
    return None, best_dist

def register_face_interactive(embedding, face_crop):
    """Ask user if they want to register new face and update memory immediately."""
    window_name = "Register New Face (Y/N)"
    cv2.imshow(window_name, face_crop)
    log("[REGISTER] Press 'y' to register, 'n' to skip.")

    registered = False
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('y'):
            cv2.destroyWindow(window_name)
            name = input("Enter name for this person: ").strip()
            if not name:
                log("[REGISTER] Skipped (empty name).")
                break
            if name in known_faces:
                known_faces[name].append(embedding)
            else:
                known_faces[name] = [embedding]
            save_known_faces(known_faces)
            log(f"[REGISTER] ✅ New face '{name}' added to memory.")
            registered = True
            break
        elif key in [ord('n'), ord('q'), 27]:
            cv2.destroyWindow(window_name)
            log("[REGISTER] Skipped registration.")
            break
    return registered

# ----------------- TEXT TO SPEECH -----------------
def init_tts():
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty("rate")
        engine.setProperty("rate", max(120, rate - 10))
        return engine
    except Exception as e:
        log(f"[TTS] Init failed: {e}")
        return None

tts_engine = init_tts() if SPEAK_ENABLED else None

def speak(text):
    """Fully blocking TTS — waits for completion before returning."""
    if not SPEAK_ENABLED or tts_engine is None:
        return
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()  # should block
        tts_engine.endLoop()     # force close TTS loop cleanly
        # small wait to ensure buffer flush
        time.sleep(0.2)
    except Exception as e:
        log(f"[TTS] Error: {e}")

# ----------------- CAMERA -----------------
log("[INFO] Initializing webcam...")
cam = cv2.VideoCapture(WEBCAM_INDEX)
if not cam.isOpened():
    log("[ERROR] Could not open webcam.")
    raise SystemExit(1)
log("[INFO] ✅ Webcam ready. Press 'q' to quit.")

# ----------------- MAIN LOOP -----------------
last_spoken_response = None
try:
    while True:
        ret, frame = cam.read()
        if not ret:
            log("[ERROR] Failed to capture frame.")
            break

        results = model(frame, conf=0.25, imgsz=640)
        object_counts = defaultdict(int)
        face_counts = defaultdict(int)
        unknown_face = None

        for r in results:
            if not hasattr(r, "boxes"):
                continue
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]

                # Detect faces or humans
                if "face" in label.lower() or "person" in label.lower():
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    try:
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        rep = DeepFace.represent(face_rgb, model_name=MODEL_NAME, enforce_detection=True)
                        if isinstance(rep, list) and len(rep) > 0:
                            embedding = np.array(rep[0]["embedding"], dtype=np.float32)
                        elif isinstance(rep, dict):
                            embedding = np.array(rep["embedding"], dtype=np.float32)
                        else:
                            embedding = np.array(rep, dtype=np.float32).flatten()
                    except Exception as e:
                        log(f"[DeepFace] ❌ {e}")
                        continue

                    name, dist = recognize_face(embedding)
                    if name:
                        face_counts[name] += 1
                        color, text = (0, 255, 0), name
                    else:
                        face_counts["Unknown"] += 1
                        color, text = (0, 0, 255), "Unknown"
                        unknown_face = {"embedding": embedding, "crop": face_crop}

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                else:
                    object_counts[label] += 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Prepare results text
        face_str = ", ".join([f"{k} ({v})" for k, v in face_counts.items()]) or "No faces"
        obj_str = ", ".join([f"{k} ({v})" for k, v in object_counts.items()]) or "No objects"
        response = f"Faces: {face_str}. Objects: {obj_str}"
        print(f"[RESULT] {response}")

        # Show window
        cv2.imshow("Smart Glasses", frame)

        # Handle unknown face registration
        if unknown_face:
            registered = register_face_interactive(unknown_face["embedding"], unknown_face["crop"])
            if registered:
                speak("Face registered successfully")
            unknown_face = None

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log("[INFO] Exit requested by user.")
            break

        # Speak results (fully blocking)
        if response != last_spoken_response:
            log("[INFO] Speaking results...")
            speak(response)
            log("[INFO] Speech finished.")
            last_spoken_response = response

        # Only now start next capture
        time.sleep(CAPTURE_INTERVAL)

except KeyboardInterrupt:
    log("\n[INFO] Keyboard interrupt, closing.")

finally:
    cam.release()
    cv2.destroyAllWindows()
    log("[INFO] ✅ Clean exit.")
