import cv2
import numpy as np
import openvino.runtime as ov
import os
import sys
import time

# --- NEW IMPORTS FOR AUDIO AND FACE REC ---
import pygame
from gtts import gTTS
import pickle
import numpy.linalg as LA
# --- END OF NEW IMPORTS ---


# --- TEXT-TO-SPEECH FUNCTION (Unchanged) ---
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
# --- END OF FUNCTION ---


# --- YOLO PREPROCESS FUNCTION (Unchanged) ---
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

# --- NEW: FACE RECOGNITION PREPROCESS FUNCTION ---
def preprocess_for_face_rec(face_crop):
    """Preprocesses a cropped face for the ArcFace recognition model."""
    try:
        # Model expects 112x112 BGR image
        resized_face = cv2.resize(face_crop, (112, 112))
        
        # Transpose to NCHW format (OpenVINO standard)
        # (H, W, C) -> (C, H, W)
        img_chw = resized_face.transpose(2, 0, 1)
        
        # Add batch dimension (N)
        img_batch = np.expand_dims(img_chw, axis=0)
        
        return img_batch
    except Exception as e:
        print(f"Error preprocessing face crop: {e}")
        return None
# --- END OF NEW FUNCTION ---


# --- OPENVINO INFERENCE FUNCTION (Unchanged) ---
def run_openvino_inference(compiled_model, image_input):
    """
    Runs inference on the preprocessed image using an already compiled OpenVINO model.
    """
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

# --- YOLO POST-PROCESS FUNCTION (Unchanged) ---
def postprocess_yolov8_output(output, conf_threshold=0.5, iou_threshold=0.5):
    """
    Post-processes the raw output of a YOLOv8 ONNX/OpenVINO model.
    """
    try:
        if output.shape[1] > output.shape[2]:
             pass
        else:
             output = output.transpose(0, 2, 1)
    except Exception as e:
        print(f"Error transposing output. Shape was {output.shape}. Error: {e}")
        return np.array([]), np.array([]), np.array([])
    output_batch = output[0]
    boxes_xywh = output_batch[:, :4]
    class_scores_all = output_batch[:, 4:]
    scores = np.max(class_scores_all, axis=1)
    class_ids = np.argmax(class_scores_all, axis=1)
    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    if len(boxes_xywh) == 0:
        return np.array([]), np.array([]), np.array([])
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    widths = boxes_xywh[:, 2]
    heights = boxes_xywh[:, 3]
    boxes_for_nms = np.stack([x1, y1, widths, heights], axis=1)
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
    indices = indices.flatten()
    boxes_xywh = boxes_xywh[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    final_boxes_xyxy = np.copy(boxes_xywh)
    final_boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    final_boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    final_boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    final_boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
    return final_boxes_xyxy, scores, class_ids

# --- DRAW DETECTIONS FUNCTION (Modified) ---
def draw_detections_on_frame(frame, detections, labels, colors):
    """Draws bounding boxes and labels on a single image frame."""
    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    scale_x = frame_width / 640.0
    scale_y = frame_height / 640.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    
    for label, score, box in detections:
        x1_scaled = int(box[0] * scale_x)
        y1_scaled = int(box[1] * scale_y)
        x2_scaled = int(box[2] * scale_x)
        y2_scaled = int(box[3] * scale_y)
        
        try:
            class_index = labels.index(label)
            color = colors[class_index]
        except ValueError:
            # Handle custom labels like 'Ameya' or 'Unknown Person'
            if label == "Unknown Person":
                color = (0, 165, 255) # Orange
            else:
                color = (0, 0, 255)   # Red for recognized faces
            
        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        
        text = f"{label}: {score:.2f}"
        if label == "Unknown Person":
            text = "Unknown Person" # Don't show score for unknown
            
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_bg_y1 = max(0, y1_scaled - text_height - 10)
        cv2.rectangle(frame, (x1_scaled, text_bg_y1), (x1_scaled + text_width, y1_scaled), color, -1)
        cv2.putText(frame, text, (x1_scaled, y1_scaled - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return frame

# --- NEW: COSINE SIMILARITY FUNCTION ---
def cosine_similarity(vec1, vec2):
    """Calculates the cosine similarity between two 1D vectors."""
    # Ensure vectors are 1D
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate norms (magnitudes)
    norm_vec1 = LA.norm(vec1)
    norm_vec2 = LA.norm(vec2)
    
    # Avoid division by zero
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
        
    # Calculate cosine similarity
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity
# --- END OF NEW FUNCTION ---


# --- MAIN CAMERA LOOP (HEAVILY MODIFIED) ---
def run_camera_detection(model_xml_path, face_rec_model_xml_path, device="AUTO"):
    """
    Main function to run live detection on the PC camera feed.
    """
    
    # --- LABELS from your data.yaml file ---
    labels = [
        "Person", "Car", "Bus", "Bicycle", "Motorcycle", "Traffic light",
        "Stop sign", "Chair", "Box", "Fire hydrant", "Door", "Window",
        "Laptop", "Mobile phone", "Book", "Human face"
    ]
    
    # --- NEW: Find the index for "Human face" ---
    try:
        HUMAN_FACE_CLASS_ID = labels.index("Human face")
        print(f"Class 'Human face' found at index {HUMAN_FACE_CLASS_ID}.")
    except ValueError:
        print("Error: 'Human face' is not in your labels list. Face recognition will be disabled.")
        HUMAN_FACE_CLASS_ID = -1 # Disable feature
        
    # --- NEW: Face Rec Threshold ---
    FACE_REC_THRESHOLD = 0.5 # Adjust this value (0.0 to 1.0)
                             # Higher = stricter, less false positives
                             # Lower = looser, more likely to match

    colors = []
    np.random.seed(42)
    for i in range(len(labels)):
        colors.append(tuple(np.random.randint(0, 256, 3).tolist()))

    
    core = ov.Core()
    available_devices = core.available_devices
    print(f"Available devices for OpenVINO: {available_devices}")
    
    if device == "AUTO":
        if "NPU" in available_devices:
            device_to_use = "NPU"
        elif "GPU" in available_devices:
            device_to_use = "GPU"
        else:
            print("Warning: Neither NPU nor GPU available. Defaulting to CPU.")
            device_to_use = "CPU"
    else:
        if device not in available_devices:
            print(f"Error: Specified device '{device}' not available. Available: {available_devices}")
            sys.exit(1)
        device_to_use = device

    device_name = core.get_property(device_to_use, "FULL_DEVICE_NAME")
    print(f"Running main inference on device: {device_to_use} ({device_name})")
    
    # --- 1. Load YOLOv8 Object Detection Model ---
    normalized_model_xml_path = os.path.normpath(model_xml_path)
    if not os.path.exists(normalized_model_xml_path):
        print(f"Error: YOLOv8 model XML file not found at {normalized_model_xml_path}")
        sys.exit(1)
    
    model_yolo = core.read_model(model=normalized_model_xml_path)
    compiled_model_yolo = core.compile_model(model_yolo, device_to_use)
    
    # --- 2. NEW: Load Face Recognition Model ---
    normalized_face_rec_path = os.path.normpath(face_rec_model_xml_path)
    if not os.path.exists(normalized_face_rec_path):
        print(f"Error: Face Recognition model XML file not found at {normalized_face_rec_path}")
        print("Check your path (should be '../face_recognition/face_recognition.xml')")
        sys.exit(1)
        
    model_face_rec = core.read_model(model=normalized_face_rec_path)
    compiled_model_face_rec = core.compile_model(model_face_rec, "CPU") # Use CPU
    face_rec_output_layer = compiled_model_face_rec.output(0)
    print("Face recognition model loaded onto CPU.")

    # --- 3. NEW: Load Known Face Embeddings ---
    EMBEDDINGS_FILE = "face_embeddings.pkl"
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Warning: Embeddings file not found at {EMBEDDINGS_FILE}.")
        print("Creating a new, empty one. Press 's' to add faces.")
        known_faces_db = {}
        # Save the empty file
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(known_faces_db, f)
    else:
        with open(EMBEDDINGS_FILE, "rb") as f:
            known_faces_db = pickle.load(f)
        print(f"Loaded {len(known_faces_db)} known face(s): {list(known_faces_db.keys())}")
        
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera.")
        return
    
    print("\nStarting camera feed... Press 'q' to quit.")
    print("      *** Press 's' to save a new unknown face ***")
    
    # --- AUDIO COOLDOWN VARIABLES ---
    last_spoken_time = 0
    last_spoken_labels = set()
    speak_cooldown = 3.0
    
    # --- NEW: Variable to hold the last unknown face embedding ---
    last_unknown_embedding = None
    # --- END NEW ---
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # Store frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # 1. Preprocess for YOLO
        img_np_for_inference = preprocess_for_inference(frame)
        
        if img_np_for_inference is not None:
            # 2. Run YOLO Inference
            output, time_taken = run_openvino_inference(compiled_model_yolo, img_np_for_inference)
            
            if output is None:
                continue

            # 3. Post-process YOLO output
            boxes, scores, class_ids = postprocess_yolov8_output(output)
            
            current_detected_labels = set() 
            processed_detections = [] # List to hold (label, score, box)
            
            if len(boxes) > 0:
                scale_x = frame_width / 640.0
                scale_y = frame_height / 640.0

                for i in range(len(class_ids)):
                    if class_ids[i] < len(labels):
                        label_name = labels[class_ids[i]]
                        box = boxes[i]
                        score = scores[i]
                        
                        # --- 4. NEW: FACE RECOGNITION LOGIC ---
                        if class_ids[i] == HUMAN_FACE_CLASS_ID:
                            
                            # A. Get scaled coordinates for cropping
                            x1_scaled = int(box[0] * scale_x)
                            y1_scaled = int(box[1] * scale_y)
                            x2_scaled = int(box[2] * scale_x)
                            y2_scaled = int(box[3] * scale_y)
                            
                            padding = 10
                            x1_crop = max(0, x1_scaled - padding)
                            y1_crop = max(0, y1_scaled - padding)
                            x2_crop = min(frame_width, x2_scaled + padding)
                            y2_crop = min(frame_height, y2_scaled + padding)

                            # B. Crop the face from the *original* frame
                            face_crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]

                            if face_crop.size > 0:
                                # C. Preprocess face for ArcFace model
                                preprocessed_face = preprocess_for_face_rec(face_crop)
                                
                                if preprocessed_face is not None:
                                    # D. Get embedding for the current face
                                    rec_result = compiled_model_face_rec([preprocessed_face])[face_rec_output_layer]
                                    current_embedding = rec_result.flatten()
                                    
                                    # E. Compare with known faces
                                    found_match = False
                                    for name, known_embedding in known_faces_db.items():
                                        similarity = cosine_similarity(current_embedding, known_embedding)
                                        
                                        if similarity > FACE_REC_THRESHOLD:
                                            label_name = name # Change "Human face" to "Ameya"
                                            score = similarity # Show similarity score
                                            found_match = True
                                            break 
                                    
                                    if not found_match:
                                        label_name = "Unknown Person"
                                        # NEW: Store this embedding in case we want to save it
                                        last_unknown_embedding = current_embedding
                                        
                        # --- END OF FACE RECOGNITION LOGIC ---
                        
                        current_detected_labels.add(label_name)
                        processed_detections.append((label_name, score, box))

            # --- 5. AUDIO LOGIC (Unchanged) ---
            current_time = time.time()
            if current_detected_labels != last_spoken_labels and (current_time - last_spoken_time) > speak_cooldown:
                if len(current_detected_labels) > 0:
                    # Filter out "Unknown Person" from spoken text
                    spoken_labels = [l for l in current_detected_labels if l != "Unknown Person"]
                    if spoken_labels:
                        spoken_text = "I see " + ", ".join(spoken_labels)
                        speak(spoken_text)
                        
                last_spoken_time = current_time
                last_spoken_labels = current_detected_labels
            elif len(current_detected_labels) == 0 and len(last_spoken_labels) > 0:
                last_spoken_labels = set()
            
            # --- 6. DRAWING LOGIC (Modified) ---
            annotated_frame = draw_detections_on_frame(frame, processed_detections, labels, colors)
            
            # Add FPS information
            fps = 1.0 / time_taken if time_taken > 0 else 0
            inference_text = f"Inference: {time_taken*1000:.2f} ms ({fps:.1f} FPS)"
            cv2.putText(annotated_frame, inference_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('PC Camera Detection (YOLOv8 + FaceRec)', annotated_frame)
            
        # --- 7. NEW: Check for key presses ('q' or 's') ---
        key = cv2.waitKey(1) & 0xFF

        # 'q' to quit
        if key == ord('q'):
            break
            
        # 's' to save the last 'Unknown Person'
        if key == ord('s'):
            if last_unknown_embedding is None:
                print("[Save] No 'Unknown Person' has been seen to save.")
            else:
                print("\n[Save] --- ADDING NEW FACE ---")
                # Temporarily pause to ask for name
                name = input("[Save] Enter the name for this person (or press Enter to cancel): ")
                
                if name:
                    # Add to the in-memory database
                    known_faces_db[name] = last_unknown_embedding
                    
                    # Save the updated database back to the file
                    try:
                        with open(EMBEDDINGS_FILE, "wb") as f:
                            pickle.dump(known_faces_db, f)
                        print(f"[Save] Successfully saved '{name}' to {EMBEDDINGS_FILE}")
                        print(f"[Save] Total faces in database: {len(known_faces_db)} ({list(known_faces_db.keys())})")
                    except Exception as e:
                        print(f"[Save] Error saving file: {e}")
                        
                    # Clear the last embedding
                    last_unknown_embedding = None
                else:
                    print("[Save] Save cancelled.")
        # --- END OF KEY PRESS LOGIC ---

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")


if __name__ == "__main__":
    # --- Configuration ---
    
    # Path to your YOLOv8 model
    openvino_model_dir = "yolo8_16class/openvino_model"
    openvino_xml_path = os.path.join(openvino_model_dir, "best.xml")
    
    # --- NEW: Path to your Face Recognition model ---
    # This path goes UP one level from 'yolo' to 'SemiX'
    face_rec_model_dir = "../face_recognition" 
    face_rec_xml_path = os.path.join(face_rec_model_dir, "face_recognition.xml")
    
    device_to_use = "AUTO"  # "CPU", "GPU", "NPU" or "AUTO"

    # --- Run on PC Camera ---
    print("\n\n--- Starting live detection on PC camera ---")
    run_camera_detection(
        openvino_xml_path, 
        face_rec_xml_path, # This is the corrected variable name
        device=device_to_use
    )