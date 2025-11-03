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
from huggingface_hub import hf_hub_download
import urllib.request
from collections import Counter
import json # <--- Added for compatibility (though not used by str_to_image)

# --- NEW SERVER IMPORTS ---
import socket
import base64
# --- END SERVER IMPORTS ---


# --- Configuration ---
# File to store the video-cluster embeddings
EMBEDDINGS_FILE = "face_embeddings_video.pkl"
# Face recognition confidence threshold
FACE_REC_THRESHOLD = 0.5 

# --- MODEL PATHS (Centralized) ---
# Path to your YOLOv8 model
YOLO_MODEL_DIR = "yolo8_16class/openvino_model"
YOLO_XML_PATH = os.path.join(YOLO_MODEL_DIR, "best.xml")

# Path to your Face Recognition model
FACE_REC_MODEL_DIR = "../face_recognition" 
FACE_REC_XML_PATH = os.path.join(FACE_REC_MODEL_DIR, "face_recognition.xml")
FACE_REC_BIN_PATH = os.path.join(FACE_REC_MODEL_DIR, "face_recognition.bin")

# Path to your Haar Cascade
HAAR_CASCADE_PATH = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\haarcascade_frontalface_default.xml"
# --- End Configuration ---


# ---
# --- NEW MODEL DOWNLOAD FUNCTIONS ---
# ---
def download_face_rec_model_if_needed():
    """Checks if the face rec model exists, if not, downloads it from Hugging Face."""
    
    # Ensure the target directory exists
    os.makedirs(FACE_REC_MODEL_DIR, exist_ok=True)
    
    repo_id = "openvinotoolkit/face-recognition-resnet100-arcface-onnx"
    original_xml = "face-recognition-resnet100-arcface-onnx.xml"
    original_bin = "face-recognition-resnet100-arcface-onnx.bin"

    try:
        if not os.path.exists(FACE_REC_XML_PATH):
            print(f"Downloading {original_xml} from Hugging Face...")
            hf_hub_download(
                repo_id=repo_id,
                filename=original_xml,
                local_dir=FACE_REC_MODEL_DIR
            )
            # Rename the downloaded file to what the script expects
            os.rename(os.path.join(FACE_REC_MODEL_DIR, original_xml), FACE_REC_XML_PATH)
            print("XML file downloaded.")

        if not os.path.exists(FACE_REC_BIN_PATH):
            print(f"Downloading {original_bin} from Hugging Face...")
            hf_hub_download(
                repo_id=repo_id,
                filename=original_bin,
                local_dir=FACE_REC_MODEL_DIR
            )
            # Rename the downloaded file to what the script expects
            os.rename(os.path.join(FACE_REC_MODEL_DIR, original_bin), FACE_REC_BIN_PATH)
            print("BIN file downloaded.")
            
    except Exception as e:
        print(f"Error downloading face recognition model: {e}")
        print("Please check your internet connection or download the model manually.")
        sys.exit(1)

def download_haar_cascade_if_needed():
    """Checks if the Haar cascade exists, if not, downloads it from GitHub."""
    
    if os.path.exists(HAAR_CASCADE_PATH):
        return # File already exists

    url = "https.raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
    
    try:
        print(f"Downloading {os.path.basename(HAAR_CASCADE_PATH)}...")
        # Ensure the directory exists
        os.makedirs(os.path.dirname(HAAR_CASCADE_PATH), exist_ok=True)
        
        with urllib.request.urlopen(url) as response, open(HAAR_CASCADE_PATH, 'wb') as out_file:
            data = response.read() # Read the file
            out_file.write(data)   # Write it locally
        print("Haar Cascade downloaded.")
        
    except Exception as e:
        print(f"Error downloading Haar Cascade file: {e}")
        print("Please check your internet connection or download the file manually to:")
        print(HAAR_CASCADE_PATH)
        sys.exit(1)
# --- END NEW DOWNLOAD FUNCTIONS ---


# ---
# --- NEW SERVER HELPER FUNCTIONS ---
# ---
def str_to_image(im_str):
    """Converts a Base64 string to an OpenCV (numpy) image."""
    try:
        im_str = im_str.strip()
        # Add padding if required
        im_str += "=" * (4 - len(im_str) % 4) 
        image_data = base64.b64decode(im_str)
        np_arr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[SERVER] Error decoding Base64: {e}")
        return None

def send_msg(conn, msg_bytes):
    """Safely send a message with an 8-byte length prefix."""
    try:
        # Create an 8-byte header storing the length of the message
        data_len = f"{len(msg_bytes):<8}".encode()
        conn.sendall(data_len + msg_bytes)
    except Exception as e:
        print(f"[SERVER] Error sending message: {e}")

def recv_msg(conn):
    """Safely receive a message with an 8-byte length prefix."""
    try:
        # 1. Receive the 8-byte header first
        data_len_bytes = b""
        while len(data_len_bytes) < 8:
            packet = conn.recv(8 - len(data_len_bytes))
            if not packet:
                return None  # Connection closed
            data_len_bytes += packet
        
        # 2. Decode the header to get the message length
        data_len = int(data_len_bytes.decode().strip())
        
        # 3. Receive data until the full message is received
        data = b""
        while len(data) < data_len:
            packet = conn.recv(data_len - len(data))
            if not packet:
                return None  # Connection closed
            data += packet
            
        return data
    except Exception as e:
        print(f"[SERVER] Error receiving message: {e}")
        return None
# --- END SERVER HELPER FUNCTIONS ---


# --- Helper Function: Text-to-Speech ---
def speak(text):
    """Uses gTTS and pygame to speak the given text aloud."""
    # This check prevents trying to speak an empty string
    if not text:
        return
        
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
        # Model expects 112x112 BGR image
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
# --- SUB-ROUTINE: FACE ENROLLMENT (VIDEO RECORDING) ---
# ---
def run_face_enrollment_video(compiled_model_rec, face_cascade, known_faces_db_video):
    """
    Called to enroll a new user. Opens its own camera loop,
    records a short video, and extracts multiple embeddings.
    """
    
    # --- Configuration for this function ---
    RECORD_DURATION = 7.0 # Record for 7 seconds
    FRAMES_TO_SKIP_FOR_EMBEDDING = 5 # Grab an embedding every 5 frames
    # ---
    
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

    # 4. Give user a countdown
    print("[Enroll] Starting 7-second recording...")
    speak("Starting 7 second recording in 3...")
    print("[Enroll] ... 3"); time.sleep(1)
    speak("2")
    print("[Enroll] ... 2"); time.sleep(1)
    speak("1")
    print("[Enroll] ... 1"); time.sleep(1)
    
    speak("Recording! Please move your head slowly: up, down, left, and right.")
    print("[Enroll] --- RECORDING --- (Move your head slowly!)")

    # 5. Start the recording loop
    start_time = time.time()
    frame_count = 0
    
    while True:
        current_time = time.time()
        elapsed = current_time - start_time
        
        # Stop if the duration is over
        if elapsed > RECORD_DURATION:
            break
            
        ret, frame = cap_enroll.read()
        if not ret:
            print("[Enroll] Error: Could not read frame.")
            break
            
        # We only process every Nth frame to get variety
        if frame_count % FRAMES_TO_SKIP_FOR_EMBEDDING == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                x, y, w, h = faces[0]
                face_crop = frame[y:y+h, x:x+w]
                
                preprocessed_face = preprocess_for_face_rec(face_crop)
                if preprocessed_face is not None:
                    result = run_openvino_inference(compiled_model_rec, preprocessed_face)[0]
                    if result is not None:
                        embedding = result.flatten()
                        new_embeddings_list.append(embedding)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Show visual feedback
        time_left = RECORD_DURATION - elapsed
        cv2.putText(frame, f"Recording... Move your head.", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Time left: {time_left:.1f}s", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Embeddings captured: {len(new_embeddings_list)}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Enrolling Face...", frame)
        frame_count += 1
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[Enroll] User cancelled recording.")
            speak("Cancelled recording.")
            break
            
    cap_enroll.release()
    cv2.destroyAllWindows()
    print(f"[Enroll] Recording complete. Captured {len(new_embeddings_list)} embeddings.")
    speak("Recording complete.")

    # 6. Save the data
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
        
    # 7. Save back to the file
    try:
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_faces_db_video, f)
        print(f"[Enroll] Successfully saved {len(new_embeddings_list)} new embeddings for '{name}'.")
        speak(f"Successfully saved {name}.")
    except Exception as e:
        print(f"[Enroll] Error saving file: {e}")
        speak("Error saving file.")

# ---
# --- (MODE 1) MAIN FUNCTION: LIVE DETECTION LOOP (LOCAL CAMERA) ---
# ---
def run_camera_detection(
    compiled_model_yolo, 
    compiled_model_face_rec, 
    face_cascade, 
    device_to_use
):
    """
    Main function to run live detection on the PC camera feed.
    This mode is also used for enrolling new faces.
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

    # --- Load Known Face Embeddings ---
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Warning: Embeddings file not found at {EMBEDDINGS_FILE}.")
        print("Press 'a' to add a new person.")
        known_faces_db_video = {}
    else:
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                known_faces_db_video = pickle.load(f)
            print(f"Loaded {len(known_faces_db_video)} known face(s): {list(known_faces_db_video.keys())}")
        except Exception as e:
            print(f"Error loading {EMBEDDINGS_FILE}: {e}. Starting with an empty database.")
            known_faces_db_video = {}
    
    # --- Camera Setup ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera."); return
    
    print("\nStarting camera feed... Press 'q' to quit.")
    print("      *** Press 's' to save a new unknown face ***")
    print("      *** Press 'a' to add a new person (video) ***")
    
    # ---
    # --- (Step 1: Initialization) NEW DECOUPLED ANTI-SPAM MEMORY ---
    # ---
    last_spoken_objects_str = None # <-- Memory for objects
    last_spoken_persons_str = None # <-- Memory for people
    last_unknown_embedding = None
    last_asked_to_save_unknown = False 
    
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
            
            # Use a list to store all detections for counting
            current_detected_labels_list = []
            processed_detections = [] 
            
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
                                    if rec_result is None: continue
                                    
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
                                        
                        # Add the final label (either object or person) to the list
                        current_detected_labels_list.append(label_name)
                        processed_detections.append((label_name, score, box))
            
            # --- 
            # --- NEW: DECOUPLED AUDIO LOGIC ---
            # ---
            
            # 1. Count all detected labels
            label_counts = Counter(current_detected_labels_list)
            
            # ---
            # --- *** SMARTER BALANCING LOGIC *** ---
            # ---
            # First, count all the *specific* faces we found
            total_specific_faces = 0
            for label, count in label_counts.items():
                if label in known_faces_db_video or label == "Unknown Person":
                    total_specific_faces += count
            
            # Now, adjust the generic "Person" count
            if "Person" in label_counts and total_specific_faces > 0:
                label_counts["Person"] -= total_specific_faces
                
                # If the count is now zero or less, remove it completely
                if label_counts["Person"] <= 0:
                    del label_counts["Person"]
            # --- *** END NEW BALANCING LOGIC *** ---
            

            # 2. Build the *independent* object and person strings
            object_parts = []
            person_parts = []
            
            unknown_face_detected = "Unknown Person" in label_counts
            known_names_set = set(known_faces_db_video.keys())
            
            for label, count in label_counts.items():
                if label == "Unknown Person":
                    continue # Handled by the prompt logic
                
                # --- Pluralization ---
                s = ""
                if count > 1:
                    if label.endswith("s") or label.endswith("x") or label.endswith("Bus"): s = "es"
                    else: s = "s"
                
                formatted_str = f"{count} {label}{s}"
                
                # --- Sort into person/object lists ---
                if label in known_names_set or label == "Person":
                    person_parts.append(formatted_str)
                else:
                    object_parts.append(formatted_str)

            # 3. Build the current strings for objects and persons
            current_objects_str = ""
            if object_parts:
                current_objects_str = "I see " + ", ".join(sorted(object_parts))

            current_persons_str = ""
            if person_parts:
                current_persons_str = "I see " + ", ".join(sorted(person_parts))

            # 4. Check state *independently* to build the final speech
            final_speech_to_make = []

            # --- Check object state ---
            if current_objects_str != last_spoken_objects_str:
                if current_objects_str: # Only add if it's not empty
                    final_speech_to_make.append(current_objects_str)
                last_spoken_objects_str = current_objects_str

            # --- Check person state ---
            if current_persons_str != last_spoken_persons_str:
                if current_persons_str: # Only add if it's not empty
                    final_speech_to_make.append(current_persons_str)
                last_spoken_persons_str = current_persons_str

            # --- Check unknown person prompt state ---
            if unknown_face_detected and not last_asked_to_save_unknown:
                prompt_text = "I see an unknown person. Press 's' to save this person."
                final_speech_to_make.append(prompt_text)
                last_asked_to_save_unknown = True
            elif not unknown_face_detected and last_asked_to_save_unknown:
                last_asked_to_save_unknown = False

            # 5. Join all new speech parts and speak
            #    (filter out any empty strings)
            speak_string = ". ".join(part for part in final_speech_to_make if part)
            speak(speak_string)
            
            # --- END OF NEW AUDIO LOGIC ---
            
            # --- Drawing Logic ---
            annotated_frame = draw_detections_on_frame(frame, processed_detections, labels, colors)
            fps = 1.0 / time_taken if time_taken > 0 else 0
            inference_text = f"Inference: {time_taken*1000:.2f} ms ({fps:.1f} FPS)"
            cv2.putText(annotated_frame, inference_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('PC Camera Detection (YOLOv8 + FaceRec)', annotated_frame)
            
        # --- 7. Key Press Logic ---
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
            
            cap.release(); cv2.destroyAllWindows()
            
            run_face_enrollment_video(
                compiled_model_face_rec, 
                face_cascade, 
                known_faces_db_video
            )
            
            # --- After enrollment, clear ALL announcement memories ---
            last_spoken_objects_str = None
            last_spoken_persons_str = None
            last_asked_to_save_unknown = False # Reset unknown prompt flag
            
            print("[Enroll] Enrollment complete. Resuming detection.")
            speak("Enrollment complete. Resuming detection.")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not re-open camera. Exiting."); speak("Error: Could not re-open camera."); break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")


# ---
# --- (MODE 2) NEW MAIN FUNCTION: SERVER PROCESSING LOOP ---
# ---
def run_server_processing(
    compiled_model_yolo, 
    compiled_model_face_rec, 
    face_cascade, 
    device_to_use
):
    """
    Main function to run in server mode. Listens for images from a client,
    processes them, speaks the result, and sends a response.
    """
    
    # --- Labels (Same as local mode) ---
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
    
    # Note: Colors are not needed since we are not drawing

    # --- Load Known Face Embeddings ---
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"[SERVER] Warning: Embeddings file not found at {EMBEDDINGS_FILE}.")
        print("[SERVER] To add faces, run this script in 'local' mode first.")
        known_faces_db_video = {}
    else:
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                known_faces_db_video = pickle.load(f)
            print(f"[SERVER] Loaded {len(known_faces_db_video)} known face(s): {list(known_faces_db_video.keys())}")
        except Exception as e:
            print(f"[SERVER] Error loading {EMBEDDINGS_FILE}: {e}. Starting with an empty database.")
            known_faces_db_video = {}
    
    # --- Anti-Spam Memory ---
    last_spoken_objects_str = None
    last_spoken_persons_str = None
    last_asked_to_save_unknown = False 
    # last_unknown_embedding is not needed, as 's' key cannot be pressed

    # === NEW: SOCKET SERVER SETUP ===
    HOST = "0.0.0.0"  # Listen on all available interfaces
    PORT = 5050
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allows kernel to reuse socket
    server.bind((HOST, PORT))
    server.listen(1)
    print(f"[SERVER] Listening on {HOST}:{PORT}...")
    print("[SERVER] Running in server mode. Face enrollment (keys 'a', 's') is disabled.")
    print("[SERVER] To enroll new faces, restart in 'local' mode.")

    try:
        conn, addr = server.accept()
        print(f"[SERVER] Connected to {addr}")

        # --- Main Detection Loop (Server) ---
        while True:
            # --- 1. RECEIVE IMAGE FROM CLIENT ---
            data_bytes = recv_msg(conn)
            if data_bytes is None:
                print("[SERVER] Client disconnected.")
                break # Exit loop, close connection, and shut down server

            try:
                data_str = data_bytes.decode()
            except Exception as e:
                print(f"[SERVER] Error decoding data: {e}")
                continue

            frame = str_to_image(data_str)
            if frame is None:
                print("[SERVER] Error: Invalid image received.")
                send_msg(conn, "Error: Invalid image".encode())
                continue
            
            # --- 2. PROCESS THE IMAGE ---
            frame_height, frame_width = frame.shape[:2]
            
            img_np_for_inference = preprocess_for_inference(frame)
            
            if img_np_for_inference is None:
                send_msg(conn, "Error: Could not preprocess image".encode())
                continue
                
            output, time_taken = run_openvino_inference(compiled_model_yolo, img_np_for_inference)
            if output is None: 
                send_msg(conn, "Error: Inference failed".encode())
                continue

            boxes, scores, class_ids = postprocess_yolov8_output(output)
            
            current_detected_labels_list = []
            # processed_detections list is not needed (no drawing)
            
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
                                    if rec_result is None: continue
                                    
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
                                        found_match = True
                                    
                                    if not found_match:
                                        label_name = "Unknown Person"
                                        # We don't save last_unknown_embedding
                                        
                        current_detected_labels_list.append(label_name)
                        # processed_detections.append((label_name, score, box)) # Not needed
            
            # --- 3. GENERATE AUDIO/TEXT RESPONSE ---
            
            label_counts = Counter(current_detected_labels_list)
            
            # --- SMARTER BALANCING LOGIC ---
            total_specific_faces = 0
            for label, count in label_counts.items():
                if label in known_faces_db_video or label == "Unknown Person":
                    total_specific_faces += count
            if "Person" in label_counts and total_specific_faces > 0:
                label_counts["Person"] -= total_specific_faces
                if label_counts["Person"] <= 0:
                    del label_counts["Person"]
            
            # --- Build strings ---
            object_parts = []
            person_parts = []
            unknown_face_detected = "Unknown Person" in label_counts
            known_names_set = set(known_faces_db_video.keys())
            
            for label, count in label_counts.items():
                if label == "Unknown Person":
                    continue 
                s = ""
                if count > 1:
                    if label.endswith("s") or label.endswith("x") or label.endswith("Bus"): s = "es"
                    else: s = "s"
                formatted_str = f"{count} {label}{s}"
                if label in known_names_set or label == "Person":
                    person_parts.append(formatted_str)
                else:
                    object_parts.append(formatted_str)

            current_objects_str = ""
            if object_parts:
                current_objects_str = "I see " + ", ".join(sorted(object_parts))

            current_persons_str = ""
            if person_parts:
                current_persons_str = "I see " + ", ".join(sorted(person_parts))

            final_speech_to_make = []

            if current_objects_str != last_spoken_objects_str:
                if current_objects_str:
                    final_speech_to_make.append(current_objects_str)
                last_spoken_objects_str = current_objects_str

            if current_persons_str != last_spoken_persons_str:
                if current_persons_str:
                    final_speech_to_make.append(current_persons_str)
                last_spoken_persons_str = current_persons_str

            # --- MODIFIED UNKNOWN PROMPT ---
            if unknown_face_detected and not last_asked_to_save_unknown:
                # Changed prompt: No "Press 's'"
                prompt_text = "I see an unknown person." 
                final_speech_to_make.append(prompt_text)
                last_asked_to_save_unknown = True
            elif not unknown_face_detected and last_asked_to_save_unknown:
                last_asked_to_save_unknown = False

            speak_string = ". ".join(part for part in final_speech_to_make if part)
            
            # --- 4. SPEAK ON SERVER & SEND TO CLIENT ---
            
            # Speak on the server (PC)
            speak(speak_string) 
            
            # Send text response back to the client
            response_str = speak_string if speak_string else "No new objects detected."
            send_msg(conn, response_str.encode())
            print(f"[SERVER] Sent to client: {response_str}")

            # --- NO DRAWING OR KEYPRESS LOGIC ---
            # (cv2.imshow, cv2.waitKey, etc. are removed)

    except Exception as e:
        print(f"[SERVER] A server error occurred: {e}")
    
    finally:
        # This code runs on disconnect or error
        if 'conn' in locals():
            conn.close()
        server.close()
        print("[SERVER] Connection closed and server shut down.")


# ---
# --- MAIN EXECUTION ---
# ---
if __name__ == "__main__":
    
    print("--- Initializing Smart Glasses ---")
    
    # --- 1. Download models if they are missing ---
    download_face_rec_model_if_needed()
    download_haar_cascade_if_needed()
    
    # --- 2. Load OpenVINO Core ---
    core = ov.Core()
    
    # --- 3. Load YOLOv8 Model ---
    if not os.path.exists(YOLO_XML_PATH):
        print(f"Error: YOLOv8 model not found at {YOLO_XML_PATH}"); sys.exit(1)
    
    # Determine device
    device_to_use = "AUTO"
    available_devices = core.available_devices
    if "NPU" in available_devices: device_to_use = "NPU"
    elif "GPU" in available_devices: device_to_use = "GPU"
    else: device_to_use = "CPU"
    device_name = core.get_property(device_to_use, "FULL_DEVICE_NAME")
    print(f"Loading YOLOv8 model on device: {device_to_use} ({device_name})")
    
    model_yolo = core.read_model(model=YOLO_XML_PATH)
    compiled_model_yolo = core.compile_model(model_yolo, device_to_use)
    
    # --- 4. Load Face Recognition Model (on CPU) ---
    print("Loading Face Recognition model on CPU...")
    model_face_rec = core.read_model(model=FACE_REC_XML_PATH)
    compiled_model_face_rec = core.compile_model(model_face_rec, "CPU")
    
    # --- 5. Load Haar Cascade (for enrollment) ---
    print("Loading Haar Cascade classifier...")
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    
    print("--- Initialization Complete ---")

    # --- 6. NEW: Choose Mode ---
    mode = ""
    while mode not in ['1', '2']:
        print("\nSelect execution mode:")
        print("  [1] Local Camera Mode (For detection and face enrollment)")
        print("  [2] Server Mode (Listens for client images)")
        mode = input("Enter 1 or 2: ").strip()

    if mode == '1':
        print("Starting Local Camera Mode...")
        run_camera_detection(
            compiled_model_yolo, 
            compiled_model_face_rec,
            face_cascade,
            device_to_use
        )
    elif mode == '2':
        print("Starting Server Mode...")
        run_server_processing(
            compiled_model_yolo, 
            compiled_model_face_rec,
            face_cascade,
            device_to_use
        )