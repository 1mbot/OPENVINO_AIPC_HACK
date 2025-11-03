import cv2
import numpy as np
import openvino.runtime as ov
import pickle
import os
import sys
import time

# --- Configuration ---
# Number of samples to take per person
NUM_SAMPLES = 5
# Delay between samples (in seconds)
SAMPLE_DELAY = 2.0 

# Path to the Face Recognition (ArcFace) model
FACE_REC_MODEL_XML = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\face_recognition\face_recognition.xml"
# Path to a simple OpenCV face detector
HAAR_CASCADE_PATH = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\haarcascade_frontalface_default.xml"
# Output file
EMBEDDINGS_FILE = "face_embeddings_video.pkl" # <--- MODIFIED
# --- End Configuration ---

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

def main():
    # 1. Get user's name
    name = input("Enter the name of the person you are adding: ")
    if not name:
        print("No name entered. Exiting.")
        sys.exit(1)

    print(f"Adding '{name}' to the *video* face database...")

    # 2. Load OpenVINO models
    core = ov.Core()
    
    if not os.path.exists(FACE_REC_MODEL_XML):
        print(f"Error: Face recognition model not found at {FACE_REC_MODEL_XML}")
        sys.exit(1)
    print(f"Loading face recognition model...")
    model_rec = core.read_model(model=FACE_REC_MODEL_XML)
    compiled_model_rec = core.compile_model(model_rec, "CPU")
    rec_output_layer = compiled_model_rec.output(0)
    
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
        sys.exit(1)
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
    print("Models loaded.")

    # 3. Load existing database or create new one
    # This new DB will store a LIST of embeddings per name
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Loading existing embeddings from {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, "rb") as f:
            known_faces_db_video = pickle.load(f)
    else:
        print("Creating new embeddings database.")
        known_faces_db_video = {}
        
    # 4. Initialize embedding list for the new user
    new_embeddings_list = []
    
    # 5. Start camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    print(f"\n--- Starting Capture for '{name}' ---")
    print(f"We will take {NUM_SAMPLES} snapshots. Please move your head slowly (up, down, left, right).")
    
    for i in range(NUM_SAMPLES):
        sample_num = i + 1
        print(f"\nGetting sample {sample_num}/{NUM_SAMPLES}...")
        
        # Countdown
        for t in range(3, 0, -1):
            print(f"  ... {t}")
            time.sleep(1)
        print("  ... CAPTURING!")
        
        ret, frame = cap.read()
        if not ret:
            print("  Error: Could not read frame. Skipping sample.")
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            print("  Error: No face detected. Please make sure your face is visible.")
            continue
            
        # Find the largest face
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        face_crop = frame[y:y+h, x:x+w]
        
        # Preprocess and get embedding
        preprocessed_face = preprocess_for_face_rec(face_crop)
        if preprocessed_face is None:
            print("  Error preprocessing face. Skipping sample.")
            continue
            
        result = compiled_model_rec([preprocessed_face])[rec_output_layer]
        embedding = result.flatten()
        
        # Add embedding to our list
        new_embeddings_list.append(embedding)
        print(f"  Success! Sample {sample_num} captured.")
        
        # Show the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Sample {sample_num}/{NUM_SAMPLES} captured!", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Capturing...", frame)
        cv2.waitKey(500) # Show frame for 0.5 sec
        
    cap.release()
    cv2.destroyAllWindows()
    print("\nCapture complete.")

    # 6. Save the data
    if not new_embeddings_list:
        print("No new embeddings were captured. Exiting.")
        sys.exit(1)
        
    if name in known_faces_db_video:
        print(f"Updating existing entry for '{name}'.")
        # Add new samples to existing list
        known_faces_db_video[name].extend(new_embeddings_list)
    else:
        print(f"Creating new entry for '{name}'.")
        # Create a new list for this name
        known_faces_db_video[name] = new_embeddings_list
        
    # 7. Save back to the file
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known_faces_db_video, f)
        
    print(f"\nSuccessfully saved {len(new_embeddings_list)} new embeddings for '{name}'.")
    print(f"Total faces in database: {len(known_faces_db_video)} ({list(known_faces_db_video.keys())})")
    print(f"Total embeddings for '{name}': {len(known_faces_db_video[name])}")

if __name__ == "__main__":
    main()