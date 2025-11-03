# YOLO-based Smart Glasses for the Visually Impaired

This repository outlines the complete pipeline for implementing a YOLO-based Smart Glasses project. The system provides real-time object detection (16 classes) and face recognition, optimized with OpenVINO for deployment on an AIPC server.

The pipeline covers:
* **Data Curation:** Downloading a 16-class dataset using the OIDv4 Toolkit.
* **Format Conversion:** Converting OID annotations to YOLO format.
* **Model Training:** Training a PyTorch YOLOv8 model on Kaggle.
* **Optimization:** Converting the model to OpenVINO IR format.
* **Deployment:** Running the inference server on an AIPC (Windows) machine.

---

## Part 1: Data Preparation & Model Training (Steps 1-4)

This section covers the setup, dataset download, format conversion, and model training.

### 1. Setup & Environment Preparation

1.  **Clone the Repository:**
    ```bash
    # Clone your main project repository (assuming it contains the OIDv4_ToolKit)
    git clone [YOUR_REPOSITORY_URL]
    cd [YOUR_PROJECT_NAME]
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    # Create a virtual environment
    python3 -m venv venv
    # Activate the environment
    source venv/bin/activate
    ```

3.  **Install OID ToolKit Dependencies:**
    ```bash
    # Navigate to the toolkit directory and install required libraries
    cd OIDv4_ToolKit
    pip install -r requirements.txt
    ```

### 2. Dataset Curation (OID Download)

This step downloads 5000 images for 16 specific classes from the OID training set.

> **Warning:** The command below is missing `--multiclasses 1` and `--image_IsGroupOf 0`. The conversion script in Step 3 must be able to account for this.

```bash
python3 main.py downloader \
    --Dataset ./OID/Dataset \
    --classes "Person" "Car" "Bus" "Bicycle" "Motorcycle" "Traffic light" "Stop sign" "Chair" "Box" "Fire hydrant" "Door" "Window" "Laptop" "Mobile phone" "Book" "Human face" \
    --type_csv train \
    --limit 5000 \
    --n_threads 30 \
    -y
