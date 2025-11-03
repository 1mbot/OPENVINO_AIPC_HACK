import cv2
import numpy as np
import openvino.runtime as ov
import os
import sys
import time
import pickle
import numpy.linalg as LA
import pygame
from gtts import gTTS

# --- Configuration ---
# File to store the video-cluster embeddings
EMBEDDINGS_FILE = "face_embeddings_video.pkl"
# Number of samples to take during video enrollment
NUM_SAMPLES_TO_TAKE = 5
# Face recognition confidence threshold
FACE_REC_THRESHOLD = 0.5 
# Audio cooldown
SPEAK_COOLDOWN = 3.0
# --- End Configuration ---


# --- Helper Function: Text-to-Speech ---
def speak(text):
    """Uses gTTS and pygame to speak the given text aloud."""
    print(f"[Audio] Speaking: {text}")
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("output.mp3")
        pygame.mixer.init()
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.unload()
        pygame.mixer.quit()
        os.remove("output.mp3")
    except Exception as e:
        print(f"[Audio] Error in text-to-speech: {e}")

# --- Helper Function: Preprocess for YOLOv8 ---
def preprocess_for_inference(frame):
    """Preprocesses a single image frame for OpenVINO model inference."""
    try:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_chw = img_resized.transpose(2, 0, 1)
        img_normalized = img_chw.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error preprocessing frame for inference: {e}")
        return None

# --- Helper Function: Preprocess for Face Recognition ---
def preprocess_for_face_rec(face_crop):
    """Preprocesses a cropped face for the ArcFace recognition model."""
    try:
        resized_face = cv2.resize(face_crop, (112, 112))
        img_chw = resized_face.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_chw, axis=0)
        return img_batch
    except Exception as e:
        print(f"Error preprocessing face crop: {e}")
        return None

# --- Helper Function: Run OpenVINO Inference ---
def run_openvino_inference(compiled_model, image_input):
    """Runs inference on the preprocessed image."""
    try:
        output_layer = compiled_model.output(0)
        start_time = time.perf_counter()
        result = compiled_model([image_input])[output_layer]
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        return result, time_taken
    except Exception as e:
        print(f"Error during OpenVINO inference: {e}")
        return None, 0.0

# --- Helper Function: Post-process YOLOv8 Output ---
def postprocess_yolov8_output(output, conf_threshold=0.5, iou_threshold=0.5):
    """Post-processes the raw output of a YOLOv8 ONNX/OpenVINO model."""
    try:
        if output.shape[1] > output.shape[2]: pass
        else: output = output.transpose(0, 2, 1)
    except Exception as e:
        print(f"Error transposing output. Shape was {output.shape}. Error: {e}")
        return np.array([]), np.array([]), np.array([])
    output_batch = output[0]
    boxes_xywh = output_batch[:, :4]; class_scores_all = output_batch[:, 4:]
    scores = np.max(class_scores_all, axis=1); class_ids = np.argmax(class_scores_all, axis=1)
    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]; scores = scores[mask]; class_ids = class_ids[mask]
    if len(boxes_xywh) == 0: return np.array([]), np.array([]), np.array([])
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2; y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    widths = boxes_xywh[:, 2]; heights = boxes_xywh[:, 3]
    boxes_for_nms = np.stack([x1, y1, widths, heights], axis=1)
    indices = cv2.dnn.NMSBoxes(boxes_for_nms.tolist(), scores.tolist(), conf_threshold, iou_threshold)
    if len(indices) == 0: return np.array([]), np.array([]), np.array([])
    indices = indices.flatten()
    boxes_xywh = boxes_xywh[indices]; scores = scores[indices]; class_ids = class_ids[indices]
    final_boxes_xyxy = np.copy(boxes_xywh)
    final_boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    final_boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    final_boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    final_boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    return final_boxes_xyxy, scores, class_ids

# --- Helper Function: Draw Detections ---
def draw_detections_on_frame(frame, detections, labels, colors):
    """Draws bounding boxes and labels on a single image frame."""
    frame_width = int(frame.shape[1]); frame_height = int(frame.shape[0])
    scale_x = frame_width / 640.0; scale_y = frame_height / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX; font_scale = 0.7; font_thickness = 2
    
    for label, score, box in detections:
        x1_scaled = int(box[0] * scale_x); y1_scaled = int(box[1] * scale_y)
        x2_scaled = int(box[2] * scale_x); y2_scaled = int(box[3] * scale_y)
        try:
            class_index = labels.index(label); color = colors[class_index]
        except ValueError:
            if label == "Unknown Person": color = (0, 165, 255) # Orange
            else: color = (0, 0, 255)   # Red for recognized faces
        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        text = f"{label}: {score:.2f}"
        if label == "Unknown Person": text = "Unknown Person"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_bg_y1 = max(0, y1_scaled - text_height - 10)
        cv2.rectangle(frame, (x1_scaled, text_bg_y1), (x1_scaled + text_width, y1_scaled), color, -1)
        cv2.putText(frame, text, (x1_scaled, y1_scaled - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return frame

# --- Helper Function: Cosine Similarity ---
def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two 1D vectors."""
    vec1 = vec1.flatten(); vec2 = vec2.flatten()
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = LA.norm(vec1); norm_vec2 = LA.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0: return 0.0
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity

# ---
# --- NEW FUNCTION: FACE ENROLLMENT SUB-ROUTINE ---
# ---
def run_face_enrollment_video(compiled_model_rec, face_cascade, known_faces_db_video):
    """
    Called to enroll a new user. Opens its own camera loop,
    collects samples, and saves them to the .pkl file.
    """
    
    # 1. Get user's name
    speak("Please state the name of the person you are adding.")
    name = input("[Enroll] Enter the name of the person you are adding: ")
    if not name:
        print("[Enroll] No name entered. Cancelling enrollment.")
        speak("Cancelling enrollment.")
        return

    print(f"[Enroll] Adding '{name}' to the face database...")
    speak(f"Adding {name} to the face database.")

    # 2. Initialize embedding list for the new user
    new_embeddings_list = []
    
    # 3. Start a *new* camera capture
    cap_enroll = cv2.VideoCapture(0)
    if not cap_enroll.isOpened():
        print("[Enroll] Error: Could not open camera.")
        speak("Error: Could not open camera.")
        return

    speak(f"Starting capture for {name}. We will take {NUM_SAMPLES_TO_TAKE} snapshots. Please move your head slowly.")
    
    for i in range(NUM_SAMPLES_TO_TAKE):
        sample_num = i + 1
        print(f"\n[Enroll] Getting sample {sample_num}/{NUM_SAMPLES_TO_TAKE}...")
        
        # Countdown with audio
        speak("3")
        print("[Enroll] ... 3"); time.sleep(1)
        speak("2")
        print("[Enroll] ... 2"); time.sleep(1)
        speak("1")
        print("[Enroll] ... 1"); time.sleep(1)
        speak("Capturing!")
        print("[Enroll] ... CAPTURING!")
        
        ret, frame = cap_enroll.read()
        if not ret:
            print("[Enroll] Error: Could not read frame. Skipping sample.")
            speak("Error reading frame.")
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("[Enroll] Error: No face detected. Please make sure your face is visible.")
            speak("Error: No face detected. Please try again.")
            continue
            
        # Find the largest face
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        
        # Preprocess and get embedding
        preprocessed_face = preprocess_for_face_rec(face_crop)
        if preprocessed_face is None:
            print("[Enroll] Error preprocessing face. Skipping sample.")
            continue
            
        result = run_openvino_inference(compiled_model_rec, preprocessed_face)[0]
        embedding = result.flatten()
        
        # Add embedding to our list
        new_embeddings_list.append(embedding)
        print(f"[Enroll] Success! Sample {sample_num} captured.")
        speak(f"Sample {sample_num} captured.")
        
        # Show the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Sample {sample_num}/{NUM_SAMPLES_TO_TAKE} captured!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enrolling Face...", frame)
        cv2.waitKey(500) # Show frame for 0.5 sec
        
    cap_enroll.release()
    cv2.destroyAllWindows()
    print("[Enroll] Capture complete.")
    speak("Capture complete.")

    # 4. Save the data
    if not new_embeddings_list:
        print("[Enroll] No new embeddings were captured. Exiting.")
        speak("No new embeddings were captured. Exiting.")
        return
        
    if name in known_faces_db_video:
        print(f"[Enroll] Updating existing entry for '{name}'.")
        known_faces_db_video[name].extend(new_embeddings_list)
    else:
        print(f"[Enroll] Creating new entry for '{name}'.")
        known_faces_db_video[name] = new_embeddings_list
        
    # 5. Save back to the file
    try:
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_faces_db_video, f)
        print(f"[Enroll] Successfully saved {len(new_embeddings_list)} new embeddings for '{name}'.")
        speak(f"Successfully saved {name}.")
    except Exception as e:
        print(f"[Enroll] Error saving file: {e}")
        speak("Error saving file.")

# ---
# --- MAIN FUNCTION: LIVE DETECTION LOOP ---
# ---
def run_camera_detection(model_xml_path, face_rec_model_xml_path, haar_cascade_path, device="AUTO"):
    """
    Main function to run live detection on the PC camera feed.
    """
    
    # --- Labels ---
    labels = [
        "Person", "Car", "Bus", "Bicycle", "Motorcycle", "Traffic light",
        "Stop sign", "Chair", "Box", "Fire hydrant", "Door", "Window",
        "Laptop", "Mobile phone", "Book", "Human face"
    ]
    try:
        HUMAN_FACE_CLASS_ID = labels.index("Human face")
        print(f"Class 'Human face' found at index {HUMAN_FACE_CLASS_ID}.")
    except ValueError:
        print("Error: 'Human face' is not in your labels list. Face recognition will be disabled.")
        HUMAN_FACE_CLASS_ID = -1
        
    colors = []
    np.random.seed(42)
    for i in range(len(labels)):
        colors.append(tuple(np.random.randint(0, 256, 3).tolist()))

    # --- OpenVINO Core Init ---
    core = ov.Core()
    available_devices = core.available_devices
    print(f"Available devices for OpenVINO: {available_devices}")
    if device == "AUTO":
        if "NPU" in available_devices: device_to_use = "NPU"
        elif "GPU" in available_devices: device_to_use = "GPU"
        else: device_to_use = "CPU"
    else: device_to_use = device
    device_name = core.get_property(device_to_use, "FULL_DEVICE_NAME")
    print(f"Running main inference on device: {device_to_use} ({device_name})")
    
    # --- 1. Load YOLOv8 Model ---
    normalized_model_xml_path = os.path.normpath(model_xml_path)
    if not os.path.exists(normalized_model_xml_path):
        print(f"Error: YOLOv8 model XML file not found at {normalized_model_xml_path}"); sys.exit(1)
    model_yolo = core.read_model(model=normalized_model_xml_path)
    compiled_model_yolo = core.compile_model(model_yolo, device_to_use)
    
    # --- 2. Load Face Recognition Model ---
    normalized_face_rec_path = os.path.normpath(face_rec_model_xml_path)
    if not os.path.exists(normalized_face_rec_path):
        print(f"Error: Face Recognition model XML file not found at {normalized_face_rec_path}"); sys.exit(1)
    model_face_rec = core.read_model(model=normalized_face_rec_path)
    compiled_model_face_rec = core.compile_model(model_face_rec, "CPU")
    
    # --- 3. Load Haar Cascade (for enrollment) ---
    if not os.path.exists(haar_cascade_path):
        print(f"Error: Haar Cascade file not found at {haar_cascade_path}"); sys.exit(1)
    face_cascade = cv2.CascadeClassifier(haar_cascade_path)
    print("All models loaded.")

    # --- 4. Load Known Face Embeddings ---
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Warning: Embeddings file not found at {EMBEDDINGS_FILE}.")
        print("Press 'a' to add a new person.")
        known_faces_db_video = {}
    else:
        with open(EMBEDDINGS_FILE, "rb") as f:
            known_faces_db_video = pickle.load(f)
        print(f"Loaded {len(known_faces_db_video)} known face(s): {list(known_faces_db_video.keys())}")
    
    # --- 5. Camera Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera."); return
    
    print("\nStarting camera feed... Press 'q' to quit.")
    print("      *** Press 's' to save a new unknown face ***")
    print("      *** Press 'a' to add a new person (video) ***")
    
    last_spoken_time = 0; last_spoken_labels = set()
    last_unknown_embedding = None
    
    # --- 6. Main Detection Loop ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera."); break
        
        frame_height, frame_width = frame.shape[:2]
        
        img_np_for_inference = preprocess_for_inference(frame)
        
        if img_np_for_inference is not None:
            output, time_taken = run_openvino_inference(compiled_model_yolo, img_np_for_inference)
            if output is None: continue

            boxes, scores, class_ids = postprocess_yolov8_output(output)
            
            current_detected_labels = set(); processed_detections = [] 
            
            if len(boxes) > 0:
                scale_x = frame_width / 640.0; scale_y = frame_height / 640.0

                for i in range(len(class_ids)):
                    if class_ids[i] < len(labels):
                        label_name = labels[class_ids[i]]; box = boxes[i]; score = scores[i]
                        
                        # --- Face Recognition Logic ---
                        if class_ids[i] == HUMAN_FACE_CLASS_ID:
                            x1_scaled = int(box[0] * scale_x); y1_scaled = int(box[1] * scale_y)
                            x2_scaled = int(box[2] * scale_x); y2_scaled = int(box[3] * scale_y)
                            padding = 10
                            x1_crop = max(0, x1_scaled - padding); y1_crop = max(0, y1_scaled - padding)
                            x2_crop = min(frame_width, x2_scaled + padding); y2_crop = min(frame_height, y2_scaled + padding)
                            face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                            if face_crop.size > 0:
                                preprocessed_face = preprocess_for_face_rec(face_crop)
                                if preprocessed_face is not None:
                                    rec_result = run_openvino_inference(compiled_model_face_rec, preprocessed_face)[0]
                                    current_embedding = rec_result.flatten()
                                    
                                    found_match = False; best_match_name = ""
                                    best_match_similarity = 0.0
                                    
                                    for name, known_embeddings_list in known_faces_db_video.items():
                                        best_similarity_for_this_person = 0
                                        for known_embedding in known_embeddings_list:
                                            similarity = cosine_similarity(current_embedding, known_embedding)
                                            if similarity > best_similarity_for_this_person:
                                                best_similarity_for_this_person = similarity
                                        if best_similarity_for_this_person > best_match_similarity:
                                            best_match_similarity = best_similarity_for_this_person
                                            best_match_name = name
                                    
                                    if best_match_similarity > FACE_REC_THRESHOLD:
                                        label_name = best_match_name
                                        score = best_match_similarity
                                        found_match = True
                                    
                                    if not found_match:
                                        label_name = "Unknown Person"
                                        last_unknown_embedding = current_embedding
                                        
                        current_detected_labels.add(label_name)
                        processed_detections.append((label_name, score, box))

            # --- Audio Logic ---
            current_time = time.time()
            if current_detected_labels != last_spoken_labels and (current_time - last_spoken_time) > SPEAK_COOLDOWN:
                if len(current_detected_labels) > 0:
                    spoken_labels = [l for l in current_detected_labels if l != "Unknown Person"]
                    if spoken_labels:
                        spoken_text = "I see " + ", ".join(spoken_labels)
                        speak(spoken_text)
                last_spoken_time = current_time; last_spoken_labels = current_detected_labels
            elif len(current_detected_labels) == 0 and len(last_spoken_labels) > 0:
                last_spoken_labels = set()
            
            # --- Drawing Logic ---
            annotated_frame = draw_detections_on_frame(frame, processed_detections, labels, colors)
            fps = 1.0 / time_taken if time_taken > 0 else 0
            inference_text = f"Inference: {time_taken*1000:.2f} ms ({fps:.1f} FPS)"
            cv2.putText(annotated_frame, inference_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('PC Camera Detection (YOLOv8 + FaceRec)', annotated_frame)
            
        # --- 7. Key Press Logic (MODIFIED) ---
        key = cv2.waitKey(1) & 0xFF

        # 'q' to quit
        if key == ord('q'):
            break
            
        # 's' to save the last 'Unknown Person'
        if key == ord('s'):
            if last_unknown_embedding is None:
                print("[Save] No 'Unknown Person' has been seen to save.")
                speak("No unknown person to save.")
            else:
                speak("Saving last unknown person. Please state their name.")
                print("\n[Save] --- ADDING NEW FACE ---")
                name = input("[Save] Enter the name for this person (or press Enter to cancel): ")
                
                if name:
                    if name in known_faces_db_video:
                        known_faces_db_video[name].append(last_unknown_embedding)
                        print(f"[Save] Added new embedding to '{name}'.")
                    else:
                        known_faces_db_video[name] = [last_unknown_embedding]
                        print(f"[Save] Created new entry for '{name}'.")
                    try:
                        with open(EMBEDDINGS_FILE, "wb") as f:
                            pickle.dump(known_faces_db_video, f)
                        print(f"[Save] Successfully saved '{name}' to {EMBEDDINGS_FILE}")
                        speak(f"Successfully saved {name}.")
                    except Exception as e:
                        print(f"[Save] Error saving file: {e}")
                else:
                    print("[Save] Save cancelled.")
                    speak("Save cancelled.")
            last_unknown_embedding = None # Clear after attempt
            
        # 'a' to add a new person via video enrollment
        if key == ord('a'):
            print("\n[Enroll] Pausing detection to enroll a new person...")
            speak("Pausing to add a new person.")
            
            # --- PAUSE MAIN LOOP ---
            cap.release()
            cv2.destroyAllWindows()
            
            # --- RUN ENROLLMENT SUB-ROUTINE ---
            run_face_enrollment_video(
                compiled_model_face_rec, 
                face_cascade, 
                known_faces_db_video
            )
            
            # --- RESUME MAIN LOOP ---
            print("[Enroll] Enrollment complete. Resuming detection.")
            speak("Enrollment complete. Resuming detection.")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not re-open camera. Exiting.")
                speak("Error: Could not re-open camera.")
                break
            # --- END OF KEY PRESS LOGIC ---

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")

# ---
# --- MAIN EXECUTION ---
# ---
if __name__ == "__main__":
    # --- Configuration ---
    # Path to your YOLOv8 model
    openvino_model_dir = "yolo8_16class/openvino_model"
    openvino_xml_path = os.path.join(openvino_model_dir, "best.xml")
    
    # Path to your Face Recognition model
    face_rec_model_dir = "../face_recognition" 
    face_rec_xml_path = os.path.join(face_rec_model_dir, "face_recognition.xml")
    
    # Path to your Haar Cascade
    haar_cascade_path = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\haarcascade_frontalface_default.xml"
    
    device_to_use = "AUTO"

    # --- Run on PC Camera ---
    print("\n\n--- Starting live detection on PC camera ---")
    run_camera_detection(
        openvino_xml_path, 
        face_rec_xml_path,
        haar_cascade_path,
        device=device_to_use
    )