import os
import time
import pickle
from collections import defaultdict, Counter
import numpy as np
import cv2
import pyttsx3
from ultralytics import YOLO
from deepface import DeepFace
import urllib.request  # Keep for future downloads if needed

# ----------------- CONFIG -----------------
KNOWN_FACES_FILE = "known_faces.pkl"  # file to persist embeddings
RECOGNITION_THRESHOLD = 0.40          # euclidean threshold for matching
YOLO_MODEL_PATH = "best.pt"           # path to YOLO .pt
MODEL_NAME = "VGG-Face"               # DeepFace model name (VGG-Face, Facenet, OpenFace, DeepFace, DeepID)
WEBCAM_INDEX = 0
SPEAK_ENABLED = True
VERBOSE = True

# --- Config from reference script ---
ENROLL_RECORD_DURATION = 60.0  # seconds to record during video enrollment
ENROLL_SKIP_FRAMES = 3        # take embedding every N frames during enroll
# ------------------------------------------

# --- REMOVED ---
# HAAR_CASCADE_PATH is no longer needed.
# download_haar_cascade_if_needed() is no longer needed.
# ---

def log(*args, **kwargs):
    """Prints only if VERBOSE is True."""
    if VERBOSE:
        print(*args, **kwargs)

# ----------------- YOLO LOAD & FIND FACE/PERSON IDs -----------------
log("[INFO] Loading YOLO model...")
try:
    model = YOLO(YOLO_MODEL_PATH)
    log("[INFO] ✅ YOLO model loaded successfully.")

    # --- NEW: Find the class IDs for 'Human face' and 'Person' ---
    HUMAN_FACE_CLASS_ID = -1
    PERSON_CLASS_ID = -1
    FACE_CLASS_IDS = []
    
    if hasattr(model, 'names'):
        for class_id, name in model.names.items():
            if name.lower() == "human face": # Use lowercase for safety
                HUMAN_FACE_CLASS_ID = class_id
            elif name.lower() == "person":
                PERSON_CLASS_ID = class_id
            # Also check for a generic 'face'
            elif name.lower() == "face":
                if HUMAN_FACE_CLASS_ID == -1: # Only use if "Human face" isn't found
                    HUMAN_FACE_CLASS_ID = class_id

        # Create a list of all class IDs we'll use for face rec
        if HUMAN_FACE_CLASS_ID != -1:
            FACE_CLASS_IDS.append(HUMAN_FACE_CLASS_ID)
            log(f"[INFO] Found 'Human face' (or 'face') at class ID: {HUMAN_FACE_CLASS_ID}")
        if PERSON_CLASS_ID != -1:
            FACE_CLASS_IDS.append(PERSON_CLASS_ID)
            log(f"[INFO] Found 'Person' at class ID: {PERSON_CLASS_ID}")

        if not FACE_CLASS_IDS:
            log("[ERROR] Could not find 'Human face' or 'Person' in your YOLO model. Face recognition will not work.")
            raise SystemExit(1)
    else:
        log("[ERROR] YOLO model has no 'names' attribute. Cannot find face classes.")
        raise SystemExit(1)
        
except Exception as e:
    log(f"[ERROR] Could not load YOLO model ({YOLO_MODEL_PATH}): {e}")
    raise SystemExit(1)

# ----------------- KNOWN FACES -----------------
def load_known_faces(path=KNOWN_FACES_FILE):
    """Loads the known face embeddings from a pickle file."""
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            # Convert lists back to numpy arrays
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
    """Saves the known face embeddings to a pickle file."""
    # Convert numpy arrays to lists for pickling
    data = {name: [emb.tolist() for emb in embeds] for name, embeds in faces.items()}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    log("[INFO] ✅ Known faces saved to disk.")

known_faces = load_known_faces()

# ----------------- RECOGNITION HELPERS -----------------
def euclidean_distance(a, b):
    """Calculates the Euclidean distance between two embedding vectors."""
    if a.shape != b.shape:
        log(f"[WARN] Skipping mismatched embeddings: {a.shape} vs {b.shape}")
        return float("inf")
    return np.linalg.norm(a - b)

def recognize_face(embedding):
    """
    Compares a new embedding to all known faces.
    Returns (name, distance) if matched, else (None, best_dist).
    """
    if not known_faces:
        return None, None
    best_name, best_dist = None, float("inf")
    for name, embeds in known_faces.items():
        for e in embeds:
            dist = euclidean_distance(embedding, e)
            if dist < best_dist:
                best_dist = dist
                best_name = name
    
    # Check if the best match is within the acceptance threshold
    if best_dist < RECOGNITION_THRESHOLD:
        return best_name, best_dist
    return None, best_dist

# ----------------- TTS -----------------
def init_tts():
    """Initializes the pyttsx3 engine."""
    try:
        engine = pyttsx3.init()
        rate = engine.getProperty("rate")
        engine.setProperty("rate", max(120, rate - 20)) # Slow down speech a bit
        return engine
    except Exception as e:
        log(f"[TTS] Init failed: {e}")
        return None

tts_engine = init_tts() if SPEAK_ENABLED else None

def speak(text):
    """Blocking TTS: speak and return after completion."""
    if not SPEAK_ENABLED or tts_engine is None or not text:
        return
    log(f"[SPEAK] {text}")
    try:
        tts_engine.say(text)
        tts_engine.runAndWait()
        time.sleep(0.1) # Small pause after speaking
    except Exception as e:
        log(f"[TTS] Error: {e}")

# ----------------- ENROLLMENT (VIDEO) --- FIXED -----------------
def enroll_via_video(yolo_model, face_class_ids_list):
    """
    Open webcam, record ENROLL_RECORD_DURATION seconds,
    take embeddings every ENROLL_SKIP_FRAMES frames, and
    ask for a name to store them.
    
    --- FIXED: This function now uses the YOLO model ---
    """
    # 1. Get user's name
    speak("Please state the name of the person you are adding.")
    name = input("[Enroll] Enter the name of the person you are adding: ")
    if not name:
        log("[Enroll] No name entered. Cancelling enrollment.")
        speak("Cancelling enrollment.")
        return False

    log(f"[Enroll] Adding '{name}' to the face database...")
    speak(f"Adding {name} to the face database.")

    # 2. Initialize embedding list
    collected_embeddings = []
    
    # 3. Start a *new* camera capture
    cap_enroll = cv2.VideoCapture(WEBCAM_INDEX)
    if not cap_enroll.isOpened():
        log("[Enroll] Error: Could not open camera.")
        speak("Error: Could not open camera.")
        return False

    # 4. Give user a countdown (from reference script)
    log(f"[Enroll] Starting {ENROLL_RECORD_DURATION}-second recording...")
    speak(f"Starting {ENROLL_RECORD_DURATION} second recording in 3...")
    log("[Enroll] ... 3"); time.sleep(1)
    speak("2")
    log("[Enroll] ... 2"); time.sleep(1)
    speak("1")
    log("[Enroll] ... 1"); time.sleep(1)
    
    speak("Recording! Please move your head slowly: up, down, left, and right.")
    log("[Enroll] --- RECORDING --- (Move your head slowly!)")

    # 5. Start the recording loop
    start_time = time.time()
    frame_count = 0
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Stop if duration is over
        if elapsed > ENROLL_RECORD_DURATION:
            break
            
        ret, frame = cap_enroll.read()
        if not ret:
            log("[Enroll] Error: Could not read frame.")
            break
            
        # We only process every Nth frame to get variety
        if frame_count % ENROLL_SKIP_FRAMES == 0:
            
            # --- FIXED: Use YOLO to find faces, not Haar Cascade ---
            # Run YOLO on the frame, filtering *only* for faces/people
            results = yolo_model(
                frame,
                classes=face_class_ids_list, # Filter for faces/people
                verbose=False
            )
            
            boxes_xyxy = []
            if hasattr(results[0], "boxes"):
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            
            if len(boxes_xyxy) > 0:
                # Find the largest face in the frame
                areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
                largest_box_index = np.argmax(areas)
                
                # Get coordinates for the largest face
                x1, y1, x2, y2 = boxes_xyxy[largest_box_index].astype(int)
                
                # Crop the face from the *original* frame
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    try:
                        # DeepFace expects RGB format
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        rep = DeepFace.represent(face_rgb, model_name=MODEL_NAME, enforce_detection=False)
                        
                        # Handle DeepFace's inconsistent return types
                        if isinstance(rep, list) and len(rep) > 0:
                            emb = np.array(rep[0].get("embedding"), dtype=np.float32)
                        elif isinstance(rep, dict):
                            emb = np.array(rep.get("embedding"), dtype=np.float32)
                        else:
                            emb = np.array(rep, dtype=np.float32).flatten()

                        if emb.size > 0:
                            collected_embeddings.append(emb)
                            # Draw the YOLO box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            log(f"[ENROLL] Captured embedding #{len(collected_embeddings)}")
                            
                    except Exception as ex:
                        # This usually happens if DeepFace gets a bad crop (e.g., side of head)
                        log(f"[ENROLL] embedding capture failed: {ex}")
            # --- END OF YOLO FIX ---

        # Show visual feedback (from reference script)
        time_left = ENROLL_RECORD_DURATION - elapsed
        cv2.putText(frame, "Recording... Move your head.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Time left: {time_left:.1f}s", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Embeddings captured: {len(collected_embeddings)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Enrollment (press q to cancel)", frame)
        frame_count += 1
        
        # Allow user to cancel enrollment
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log("[Enroll] User cancelled recording.")
            speak("Cancelled recording.")
            break
            
    cap_enroll.release()
    cv2.destroyAllWindows()
    log(f"[Enroll] Recording complete. Captured {len(collected_embeddings)} embeddings.")
    speak("Recording complete.")

    # 6. Save the data
    if not collected_embeddings:
        log("[ENROLL] No embeddings were captured during enrollment.")
        speak("No new embeddings were captured. Exiting.")
        return False
        
    if name in known_faces:
        log(f"[Enroll] Updating existing entry for '{name}'.")
        known_faces[name].extend(collected_embeddings)
    else:
        log(f"[Enroll] Creating new entry for '{name}'.")
        known_faces[name] = collected_embeddings
        
    save_known_faces(known_faces)
    log(f"[ENROLL] ✅ Saved {len(collected_embeddings)} embeddings for '{name}'")
    speak(f"Successfully saved {name}.")
    return True

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":

    log("--- Initializing Smart Glasses ---")
    
    # --- Initialize Camera ---
    log("[INFO] Initializing webcam...")
    cam = cv2.VideoCapture(WEBCAM_INDEX)
    if not cam.isOpened():
        log("[ERROR] Could not open webcam.")
        raise SystemExit(1)
        
    log("[INFO] ✅ Webcam ready. Press 'q' to quit.")
    log("     *** Press 'a' to add a new person (video) ***")
    log("     *** Press 's' to save the last unknown person ***")

    # ----------------- AUDIO/ANNOUNCE MEMORY (from reference script) -----------------
    last_spoken_objects_str = None
    last_spoken_persons_str = None
    last_unknown_embedding = None  # <-- This will store the embedding for 's'
    last_asked_to_save_unknown = False

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                log("[ERROR] Failed to capture frame.")
                break

            # Run YOLOv8 inference
            results = model(frame, conf=0.25, imgsz=640, verbose=False)
            
            object_counts = defaultdict(int)
            face_counts = defaultdict(int)

            # iterate model outputs
            for r in results:
                if not hasattr(r, "boxes"):
                    continue
                
                for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls)
                    label = model.names[class_id]

                    # --- FIXED: Check if the detected class_id is one of our face/person IDs ---
                    if class_id in FACE_CLASS_IDS:
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue
                        try:
                            # DeepFace expects RGB
                            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            # Get embedding
                            rep = DeepFace.represent(face_rgb, model_name=MODEL_NAME, enforce_detection=False)
                            
                            # Handle DeepFace's return types
                            if isinstance(rep, list) and len(rep) > 0:
                                embedding = np.array(rep[0].get("embedding"), dtype=np.float32)
                            elif isinstance(rep, dict):
                                embedding = np.array(rep.get("embedding"), dtype=np.float32)
                            else:
                                embedding = np.array(rep, dtype=np.float32).flatten()
                            
                            if embedding.size == 0:
                                raise ValueError("Empty embedding returned")

                        except Exception as e:
                            log(f"[DeepFace] ❌ {e}")
                            continue

                        # Compare embedding to known faces
                        name, dist = recognize_face(embedding)
                        if name:
                            face_counts[name] += 1
                            color, text = (0, 255, 0), name
                        else:
                            face_counts["Unknown Person"] += 1
                            color, text = (0, 0, 255), "Unknown"
                            # Store for 's' key
                            last_unknown_embedding = embedding 

                        # Draw box and label for face
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, text, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    else:
                        # This is a regular object
                        object_counts[label] += 1
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, label, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # --- SMART BALANCING (from reference script) ---
            if "Person" in object_counts:
                total_specific_faces = sum(face_counts.values()) # Count all faces (known + unknown)
                if total_specific_faces > 0:
                    # Subtract the *specific* faces found from the *generic* 'Person' count
                    object_counts["Person"] = max(0, object_counts["Person"] - total_specific_faces)
                    if object_counts["Person"] == 0:
                        del object_counts["Person"]

            # --- Build announcement strings independently ---
            object_parts = []
            person_parts = []
            
            for k, v in sorted(object_counts.items()):
                s = "s"
                if v == 1:
                    s = ""
                # Handle words ending in 's', 'x', or 'Bus'
                elif k.endswith("s") or k.endswith("x") or k.endswith("Bus"):
                     s = "es"
                object_parts.append(f"{v} {k}{s}")
                
            for k, v in sorted(face_counts.items()):
                if k == "Unknown Person":
                    person_parts.append(f"{v} unknown person{'s' if v>1 else ''}")
                else:
                    person_parts.append(f"{k}") # Simpler: "I see Bob" not "I see 1 Bob"

            current_objects_str = ""
            if object_parts:
                current_objects_str = "I see " + ", ".join(object_parts)
            current_persons_str = ""
            if person_parts:
                current_persons_str = "I see " + ", ".join(person_parts)

            # --- DECOUPLED AUDIO LOGIC (from reference script) ---
            final_speech_to_make = []
            
            # Announce objects if they change
            if current_objects_str != last_spoken_objects_str:
                if current_objects_str:
                    final_speech_to_make.append(current_objects_str)
                last_spoken_objects_str = current_objects_str

            # Announce people if they change
            if current_persons_str != last_spoken_persons_str:
                if current_persons_str:
                    final_speech_to_make.append(current_persons_str)
                last_spoken_persons_str = current_persons_str

            # Announce prompt for unknown person (only once)
            if "Unknown Person" in face_counts and not last_asked_to_save_unknown:
                final_speech_to_make.append("I see an unknown person. Press 's' to save this person.")
                last_asked_to_save_unknown = True
            elif "Unknown Person" not in face_counts and last_asked_to_save_unknown:
                last_asked_to_save_unknown = False # Reset prompt

            # Speak the final combined string
            speak_string = ". ".join([p for p in final_speech_to_make if p])
            speak(speak_string) # speak() already handles empty strings

            # Draw UI and show frame
            fps_text = f"Objects: {sum(object_counts.values())}  People: {sum(face_counts.values())}"
            cv2.putText(frame, fps_text, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("Smart Glasses (YOLO+DeepFace)", frame)

            # --- Key handling ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log("[INFO] Exit requested by user.")
                break
            
            # --- 's' key logic (from reference script) ---
            elif key == ord('s'):
                if last_unknown_embedding is None:
                    log("[Save] No 'Unknown Person' has been seen to save.")
                    speak("No unknown person to save.")
                else:
                    speak("Saving last unknown person. Please state their name.")
                    log("\n[Save] --- ADDING NEW FACE ---")
                    name = input("[Save] Enter the name for this person (or press Enter to cancel): ").strip()
                    
                    if name:
                        if name in known_faces:
                            known_faces[name].append(last_unknown_embedding)
                            log(f"[Save] Added new embedding to '{name}'.")
                        else:
                            known_faces[name] = [last_unknown_embedding]
                            log(f"[Save] Created new entry for '{name}'.")
                        
                        save_known_faces(known_faces)
                        log(f"[Save] Successfully saved '{name}' to {KNOWN_FACES_FILE}")
                        speak(f"Successfully saved {name}.")
                    else:
                        log("[Save] Save cancelled.")
                        speak("Save cancelled.")
                
                last_unknown_embedding = None # Clear after attempt
                last_asked_to_save_unknown = False # Reset prompt

            # --- 'a' key logic (FIXED) ---
            elif key == ord('a'):
                log("[INFO] User requested full enrollment via video ('a' pressed).")
                speak("Pausing to add a new person.")
                
                # Release camera so enroll function can use it
                cam.release()
                cv2.destroyAllWindows()

                # FIXED: Pass the YOLO model and face IDs
                enrolled = enroll_via_video(model, FACE_CLASS_IDS)
                
                if enrolled:
                    speak("Enrollment complete. Resuming detection.")
                else:
                    speak("Enrollment cancelled. Resuming detection.")
                
                # Reset announcement memory
                last_spoken_objects_str = None
                last_spoken_persons_str = None
                last_asked_to_save_unknown = False
                
                # Re-initialize camera
                cam = cv2.VideoCapture(WEBCAM_INDEX)
                if not cam.isOpened():
                    log("[ERROR] Could not re-open camera. Exiting.")
                    speak("Error: Could not re-open camera.")
                    break

    except KeyboardInterrupt:
        log("\n[INFO] Keyboard interrupt, closing.")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        log("[INFO] ✅ Clean exit.")