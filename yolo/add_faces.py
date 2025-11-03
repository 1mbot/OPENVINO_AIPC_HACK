import cv2
import numpy as np
import openvino.runtime as ov
import pickle
import os
import sys

# --- Configuration ---
# Add all people and their corresponding image paths here
PEOPLE_TO_ADD = {
    "Ameya": r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\images\ameya.jpg",
    "Aditya": r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\images\aditya.jpg"
}

# Path to the Face Recognition (ArcFace) model
FACE_REC_MODEL_XML = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\face_recognition\face_recognition.xml"
# Path to a simple OpenCV face detector
HAAR_CASCADE_PATH = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\haarcascade_frontalface_default.xml"
# Output file
EMBEDDINGS_FILE = "face_embeddings.pkl"
# --- End Configuration ---

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

def get_embedding_for_image(image_path, face_cascade, compiled_model_rec, rec_output_layer):
    """
    Loads an image, finds the largest face, and returns its embedding.
    Returns None if an error occurs.
    """
    
    # 1. Load and process the image
    if not os.path.exists(image_path):
        print(f"  Error: Image of person not found at {image_path}")
        return None
        
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: Could not read image from {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("  Error: No face detected in the image. Please use a clearer, front-on photo.")
        return None
    elif len(faces) > 1:
        print(f"  Warning: Multiple faces ({len(faces)}) detected. Using the largest one.")
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        
    x, y, w, h = faces[0]
    
    # 3. Crop the face
    face_crop = img[y:y+h, x:x+w]
    print("  Face detected and cropped.")

    # 4. Preprocess the face for the recognition model
    preprocessed_face = preprocess_for_face_rec(face_crop)
    
    if preprocessed_face is None:
        print("  Error during face preprocessing.")
        return None

    # 5. Run inference to get the embedding
    result = compiled_model_rec([preprocessed_face])[rec_output_layer]
    
    # The result is the embedding, shape (1, 512). We flatten it.
    embedding = result.flatten()
    print(f"  Embedding generated with shape {embedding.shape}")
    return embedding


def main():
    print("Starting face embedding generation...")

    # 1. Initialize OpenVINO Core
    core = ov.Core()
    
    # 2. Load Face Recognition (Embedder) Model
    if not os.path.exists(FACE_REC_MODEL_XML):
        print(f"Error: Face recognition model not found at {FACE_REC_MODEL_XML}")
        sys.exit(1)
        
    print(f"Loading face recognition model from {FACE_REC_MODEL_XML}")
    model_rec = core.read_model(model=FACE_REC_MODEL_XML)
    compiled_model_rec = core.compile_model(model_rec, "CPU") # CPU is fine for this
    rec_output_layer = compiled_model_rec.output(0)
    
    # 3. Load Haar Cascade Face Detector
    if not os.path.exists(HAAR_CASCADE_PATH):
        print(f"Error: Haar Cascade file not found at {HAAR_CASCADE_PATH}")
        sys.exit(1)
        
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)

    # 4. Load existing database or create new one
    if os.path.exists(EMBEDDINGS_FILE):
        print(f"Loading existing embeddings from {EMBEDDINGS_FILE}")
        with open(EMBEDDINGS_FILE, "rb") as f:
            known_faces_db = pickle.load(f)
    else:
        print("No existing embeddings file found. Creating a new one.")
        known_faces_db = {}
        
    # 5. Loop through all people to add
    for name, image_path in PEOPLE_TO_ADD.items():
        print(f"\nProcessing '{name}'...")
        
        # Get the embedding
        embedding = get_embedding_for_image(
            image_path, 
            face_cascade, 
            compiled_model_rec, 
            rec_output_layer
        )
        
        if embedding is not None:
            # Add or update the new embedding
            known_faces_db[name] = embedding
            print(f"Successfully added/updated embedding for '{name}'.")
        else:
            print(f"Failed to generate embedding for '{name}'.")

    # 6. Save the (potentially) updated database back to the file
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known_faces_db, f)
        
    print(f"\nSuccessfully saved all embeddings to {EMBEDDINGS_FILE}")
    print(f"Total faces in database: {len(known_faces_db)} ({list(known_faces_db.keys())})")

if __name__ == "__main__":
    main()