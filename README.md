# YOLO-based Smart Glasses for the Visually Impaired

**Implementation Pipeline Setup Guide**

---

## 🧩 Overview

This setup guide outlines the complete pipeline for implementing the **YOLO-based Smart Glasses for the Visually Impaired** project.

It covers:

* Dataset curation using OIDv4 Toolkit
* Conversion to YOLO format
* Model training (PyTorch YOLOv8)
* Optimization (OpenVINO)
* Deployment on Raspberry Pi Zero (Client) and AIPC (Server)

---

## 🚀 Project Information

**Project:** YOLO-based Smart Glasses for the Visually Impaired
**Status:** Implementation Pipeline Steps

---

## 🧱 PART A: Data Preparation & Model Training (Steps 1–4)

---

### 1. Setup & Environment Preparation

#### 1.1 Clone the Repository

```bash
# Clone your main project repository (assuming it contains OIDv4_ToolKit)
git clone [YOUR_REPOSITORY_URL]
cd [YOUR_PROJECT_NAME]
```

#### 1.2 Create and Activate Virtual Environment

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

#### 1.3 Install OID Toolkit Dependencies

```bash
cd OIDv4_ToolKit
pip install -r requirements.txt
```

---

### 2. Dataset Curation (OID Download)

Download 5000 images for 16 specific classes from the OID training set.

> **Note:** The command below creates a separate folder for each class (default behavior).
> **Warning:** It’s missing `--multiclasses 1` and `--image_IsGroupOf 0`, which must be handled during conversion.

#### 2.1 Execute OID Download

```bash
python3 main.py downloader \
--Dataset ./OID/Dataset \
--classes "Person" "Car" "Bus" "Bicycle" "Motorcycle" "Traffic light" "Stop sign" "Chair" "Box" "Fire hydrant" "Door" "Window" "Laptop" "Mobile phone" "Book" "Human face" \
--type_csv train \
--limit 5000 \
--n_threads 30 \
-y
```

#### 2.2 Download Location

Data is saved at:

```
./OID/Dataset/train/[CLASS_NAME]/
./OID/Dataset/train/[CLASS_NAME]/Labels/
```

---

### 3. Dataset Conversion (OID → YOLO Format)

The OID output (denormalized pixel coordinates, class names) must be converted to YOLO format (normalized coordinates, class IDs).

#### 3.1 Select Conversion Script

Use `OID_to_YOLO_single_dir.py` for multi-folder datasets.

#### 3.2 Run Conversion

```bash
python3 OID_to_YOLO_single_dir.py
```

#### 3.3 Output Formats

* **OID_to_YOLO_single_dir.py (Selected):** Unified structure (`images/train`, `labels/train`)
* **OID_to_YOLO.py (Alternative):** Class-wise folders (requires merging)

---

### 4. YOLO Model Training

#### 4.1 Upload Final Dataset

Upload the converted dataset (with `/images`, `/labels`, `data.yaml`) to **Kaggle Datasets**.

#### 4.2 Upload Training Notebook

Upload `og-aipc-ok.ipynb` to Kaggle.

#### 4.3 Configure Notebook

Enable **Accelerator → 2x T4 GPU** for optimal performance.

#### 4.4 Start Training

Run the notebook directly, ensuring the dataset is in your input directory.

#### 4.5 Resume Training

To resume from a checkpoint:

* Place `last.pt` (67 epochs) in the input directory.
* Training resumes automatically until **80 epochs**.

---

## 🖥️ PART B: Server Deployment (AIPC / Windows) (Steps 5–11)

---

### 5. Windows (Server) Environment Setup

#### 5.1 Open PowerShell (Admin)

Right-click the Windows icon → *Terminal (Admin)* → Confirm with “Yes”.

#### 5.2 Install Git

```bash
winget install --id Git.Git -e --source winget
```

#### 5.3 Verify Git Installation

```bash
git --version
# Example output: git version 2.51.0.windows.1
```

#### 5.4 Install Python 3.12

```bash
winget install Python.Python.3.12.4
```

#### 5.5 Verify Python Installation

```bash
python --version
# Example output: Python 3.12.4
```

> ⚠️ **Check PATH Variable:**
> Ensure `C:\Program Files\Python312` and its `Scripts` folder are prioritized in PATH.

---

### 6. Project & Virtual Environment (AIPC)

#### 6.1 Clone Deployment Repository

```bash
git clone https://github.com/1mbot/OPENVINO_AIPC_HACK.git
```

#### 6.2 Navigate to Directory

```bash
cd OPENVINO_AIPC_HACK
```

#### 6.3 Create Virtual Environment

```bash
python -m venv venv
```

#### 6.4 Activate Virtual Environment (PowerShell)

```bash
.\venv\Scripts\Activate
```

#### 6.5 Troubleshooting

If activation fails:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate
```

---

### 7. Install Core Inference Libraries

```bash
pip install opencv-python openvino openvino-dev onnx ultralytics
```

---

### 8. Running YOLOv8 Inference

Ensure your model (`best.pt` or IR format) is correctly placed.

#### 8.1 Static Image Inference

```bash
python openvino_yolo8.py
```

#### 8.2 Static Image (Alternative)

```bash
python openvino_yolo8_10.py
```

#### 8.3 Live Video (Visual Only)

```bash
python openvino_videolive.py
```

#### 8.4 Live Video (with Audio)

```bash
python openvino_audiovideolive.py
```

---

### 9. Face Recognition Setup (ArcFace Model)

#### 9.1 Install Dependencies

```bash
pip install deepface tf-keras pyyaml
```

#### 9.2 Create Model Directory

```bash
mkdir face_recognition
```

#### 9.3 Download ArcFace Model

```bash
omz_downloader --name face-recognition-resnet100-arcface-onnx
```

#### 9.4 Convert Model to OpenVINO IR (OMZ)

```bash
omz_converter --name face-recognition-resnet100-arcface-onnx --output_dir face_recognition --model_name face_recognition
```

#### 9.5 Manual Conversion (Alternative)

```bash
mo --input_model public/face-recognition-resnet100-arcface-onnx/arcfaceresnet100-8.onnx \
--output_dir face_recognition \
--model_name face_recognition
```

---

### 10. Running Face Recognition & Embeddings

These scripts manage the `.pkl` face embedding database.

#### 10.1 Add Faces from Static Images

```bash
python add_faces.py
```

#### 10.2 Run Audio/Video (Face Rec Only)

```bash
python openvino_audiovideo.py
```

#### 10.3 Real-time Embeddings (YOLOv8)

```bash
python openvino_realtimeembeddings_yolo8.py
```

#### 10.4 Create Embeddings from Video

```bash
python all_faces_video.py
```

#### 10.5 Live Audio/Video (YOLO + Face Rec)

```bash
python openvino_yolo8_audiovideolive_video.py
```

#### 10.6 Main Embedding Collector (Interactive)

```bash
python openvino_yolo8_main_embeddings.py
# Press 's' → store single image embedding
# Press 'a' → store embeddings from video
```

#### 10.7 Main Embedding Collector (Video)

```bash
python openvino_yolo8_main_embeddings_video.py
```

---

### 11. Alternative: Hugging Face Model (Face Recognition)

#### 11.1 Install Dependency

```bash
pip install huggingface_hub
```

#### 11.2 Run Inference Script

```bash
python openvino_main_huggingface.py
```

---

✅ **End of Setup Guide**
