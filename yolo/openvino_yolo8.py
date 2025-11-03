import cv2
import numpy as np
import openvino.runtime as ov
import os
import sys
import time

# Import the OpenVINO conversion API
# In OpenVINO 2023.0 and newer, 'openvino.convert_model' is the recommended API.
# For older versions or specific needs, 'openvino.tools.mo.convert' might be used.
try:
    from openvino import convert_model # This is the preferred way for modern OpenVINO versions
except ImportError:
    print("Warning: 'openvino.convert_model' not found directly. Trying 'openvino.tools.mo.convert'.")
    try:
        from openvino.tools.mo import convert_model
    except ImportError:
        print("Error: OpenVINO Model Optimizer API ('openvino.convert_model' or 'openvino.tools.mo.convert') not found.")
        print("Please ensure OpenVINO Development tools are installed (`pip install openvino-dev`).")
        sys.exit(1)


# Load and preprocess image for inference
def load_image_for_inference(image_path):
    try:
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found")
            sys.exit(1)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Failed to load image {image_path}")
            sys.exit(1)
        
        # Convert to RGB, resize, transpose, normalize for model input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_chw = img_resized.transpose(2, 0, 1) # HWC to CHW
        img_normalized = img_chw.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0) # Add batch dimension
        return img_batch
    except Exception as e:
        print(f"Error preprocessing image for inference: {e}")
        sys.exit(1)

# Convert YOLOv8 model to OpenVINO format using the Python API (ovc)
def convert_to_openvino_api(onnx_path, output_dir="yolov8_openvino"):
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Define output XML and BIN paths (names derived from ONNX file)
        onnx_filename = os.path.basename(onnx_path)
        base_model_name = os.path.splitext(onnx_filename)[0]
        
        output_xml_path = os.path.join(output_dir, f"{base_model_name}.xml")
        # output_bin_path is implicitly created by save_model next to the XML

        print(f"Converting ONNX model '{onnx_path}' to OpenVINO IR (FP16) using OpenVINO API...")
        
        # 1. Convert the ONNX model to an OpenVINO Model object in memory
        converted_model = convert_model(
            onnx_path,
            example_input=np.zeros([1, 3, 640, 640], dtype=np.float32), # Use example_input for shape if input is dynamic
            # Alternatively, if you know the input name:
            # input=[(input_name, ov.PartialShape([1, 3, 640, 640]))]
            # input=[ov.PartialShape([1, 3, 640, 640])] # This also works if input name is not needed
        )
        
        # 2. Save the converted model to disk, specifying the precision and output path
        # The 'compress_to_fp16=True' argument will save weights as FP16.
        ov.save_model(converted_model, output_xml_path, compress_to_fp16=True)

        print(f"OpenVINO model conversion successful. Files saved to: {output_dir}")
        print(f"Generated XML: {output_xml_path}")
        print(f"Generated BIN: {output_xml_path.replace('.xml', '.bin')}") # Explicitly show bin path

        # Return the output directory where XML/BIN files are located
        return output_dir
    except Exception as e:
        print(f"Error converting ONNX to OpenVINO IR via API: {e}")
        sys.exit(1)

# Run inference with OpenVINO
def run_openvino_inference(image_input, model_xml_path, device="AUTO"):
    try:
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
        print(f"Running inference on device: {device_to_use} ({device_name})")
        
        normalized_model_xml_path = os.path.normpath(model_xml_path)
        
        if not os.path.exists(normalized_model_xml_path):
            print(f"Error: OpenVINO model XML file not found at {normalized_model_xml_path}")
            sys.exit(1)
        
        model = core.read_model(model=normalized_model_xml_path)
        compiled_model = core.compile_model(model, device_to_use)
        
        output_layer = compiled_model.output(0)
        # Note: If your YOLOv5 ONNX has multiple outputs, you would need to iterate
        # and collect them, then concatenate for post-processing.
        # However, recent Ultralytics YOLOv5 exports often have a single output.
        result = compiled_model([image_input])[output_layer]
        return result
    except Exception as e:
        print(f"Error during OpenVINO inference: {e}")
        sys.exit(1)

# Post-process YOLOv8 output
def postprocess_yolo_output(output, conf_threshold=0.5, iou_threshold=0.5):
    """
    Post-processes the raw output of a YOLOv8 ONNX/OpenVINO model.
    Assumes output shape is [batch, 4 + num_classes, num_proposals],
    e.g., [1, 20, 8400] for 16 classes.
    """
    
    print(f"Original output shape: {output.shape}")

    # Transpose the output from [1, 20, 8400] to [1, 8400, 20]
    # This makes it easier to process each of the 8400 proposals
    try:
        # Check if transposition is needed
        if output.shape[1] > output.shape[2]:
             print("Output shape seems to be [B, N, C]. Skipping transpose.")
             # Shape is already [1, 8400, 20]
        else:
             print(f"Transposing output from {output.shape} to [B, N, C] format.")
             output = output.transpose(0, 2, 1) # [1, 20, 8400] -> [1, 8400, 20]
    except Exception as e:
        print(f"Error transposing output. Shape was {output.shape}. Error: {e}")
        print("Assuming output is already in [B, N, C] format.")
        pass

    # output is now [1, num_proposals, 4 + num_classes], e.g. [1, 8400, 20]
    
    # Extract the first (and only) batch item
    output_batch = output[0] # Shape [8400, 20]
    
    # Parse boxes and class scores
    boxes_xywh = output_batch[:, :4]    # [8400, 4]
    class_scores_all = output_batch[:, 4:] # [8400, 16] (for your 16 classes)
    
    # Find the max class score (confidence) and class ID for each proposal
    scores = np.max(class_scores_all, axis=1) # [8400]
    class_ids = np.argmax(class_scores_all, axis=1) # [8400]

    # Filter by confidence threshold
    mask = scores > conf_threshold
    boxes_xywh = boxes_xywh[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes_xywh) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert boxes from [x_center, y_center, width, height] to [x1, y1, width, height] for NMSBoxes
    x_centers = boxes_xywh[:, 0]
    y_centers = boxes_xywh[:, 1]
    widths = boxes_xywh[:, 2]
    heights = boxes_xywh[:, 3]

    x1 = x_centers - widths / 2
    y1 = y_centers - heights / 2
    
    boxes_for_nms = np.stack([x1, y1, widths, heights], axis=1)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    indices = indices.flatten()
    boxes_xywh = boxes_xywh[indices]      # Filter boxes
    scores = scores[indices]              # Filter scores
    class_ids = class_ids[indices]        # Filter class IDs
    
    # Convert final boxes to (x1, y1, x2, y2) for easier display/interpretation
    final_boxes_xyxy = np.copy(boxes_xywh)
    final_boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1 = x_center - width/2
    final_boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1 = y_center - height/2
    final_boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2 = x_center + width/2
    final_boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2 = y_center + height/2

    return final_boxes_xyxy, scores, class_ids
    
# Main function for helmet detection
def simple_helmet_detection(image_path, onnx_model_path, openvino_model_dir, device="AUTO", output_dir="output_images"):
    # Define paths for the expected OpenVINO XML and BIN files
    # The names will be derived from the ONNX model name
    onnx_filename = os.path.basename(onnx_model_path)
    base_model_name = os.path.splitext(onnx_filename)[0] # e.g., 'best_model'
    openvino_xml_path = os.path.join(openvino_model_dir, f"{base_model_name}.xml")
    openvino_bin_path = os.path.join(openvino_model_dir, f"{base_model_name}.bin")

    # Check if the OpenVINO model (XML and BIN) already exists
    if os.path.exists(openvino_xml_path) and os.path.exists(openvino_bin_path):
        print(f"Found existing OpenVINO model at {openvino_model_dir}. Skipping ONNX conversion.")
    else:
        print(f"OpenVINO model not found at {openvino_model_dir}. Attempting conversion from ONNX...")
        if not os.path.exists(onnx_model_path):
            print(f"Error: ONNX model not found at {onnx_model_path}. Please ensure it exists before conversion.")
            sys.exit(1)
        
        # Perform conversion using the OpenVINO API
        convert_to_openvino_api(onnx_model_path, output_dir=openvino_model_dir)
        
    print(f"Loading image from: {image_path}")
    img_np_for_inference = load_image_for_inference(image_path)
    
    start_time = time.time()
    print(f"Running inference using model: {openvino_xml_path}")
    output = run_openvino_inference(img_np_for_inference, openvino_xml_path, device=device)
    end_time = time.time()
    inference_duration_ms = (end_time - start_time) * 1000
    print(f"Inference completed in {inference_duration_ms:.2f} ms")
    
    print("Post-processing inference output...")
    boxes, scores, class_ids = postprocess_yolo_output(output)
    
    
    
    # ⚠️ Make sure these labels exactly match the training classes of your YOLO model
    labels = [
        "Person",
        "Car",
        "Bus",
        "Bicycle",
        "Motorcycle",
        "Traffic light",
        "Stop sign",
        "Chair",
        "Box",
        "Fire hydrant",
        "Door",
        "Window",
        "Laptop",
        "Mobile phone",
        "Book",
        "Human face"
    ]

    detections = []
    if len(boxes) > 0:
        for i in range(len(class_ids)):
            if class_ids[i] < len(labels):
                label_name = labels[class_ids[i]]
            else:
                label_name = f"Unknown Class {class_ids[i]}"
            box_int = [int(coord) for coord in boxes[i]]
            detections.append((label_name, scores[i], box_int))

    print("\n--- Detections ---")
    if detections:
        for label, score, box in detections:
            print(f"Class: {label}, Confidence: {score:.4f}, Box: {box}")
    else:
        print("No objects detected above confidence threshold.")


       # --- Code to visualize and save the image ---
    print("\nDrawing detections on image and saving...")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Could not load original image for drawing: {image_path}")
        return

    # Resize original image to 640x640 for drawing, as detections are relative to this size
    image_display = cv2.resize(original_image, (640, 640))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2

    # Define color map for each class (customize as needed)
    class_colors = {
        "Person": (0, 255, 0),          # Green
        "Car": (255, 0, 0),             # Blue
        "Bus": (0, 255, 255),           # Yellow
        "Bicycle": (255, 165, 0),       # Orange
        "Motorcycle": (255, 20, 147),   # Pink
        "Traffic light": (0, 0, 255),   # Red
        "Stop sign": (128, 0, 128),     # Purple
        "Chair": (139, 69, 19),         # Brown
        "Box": (0, 128, 128),           # Teal
        "Fire hydrant": (255, 69, 0),   # Bright red-orange
        "Door": (0, 100, 0),            # Dark green
        "Window": (100, 149, 237),      # Cornflower blue
        "Laptop": (105, 105, 105),      # Gray
        "Mobile phone": (255, 105, 180),# Hot pink
        "Book": (0, 191, 255),          # Deep sky blue
        "Human face": (255, 215, 0)     # Gold
    }

    for label, score, box in detections:
        x1, y1, x2, y2 = box
        color = class_colors.get(label, (0, 255, 0))  # Default green

        # Draw rectangle
        cv2.rectangle(image_display, (x1, y1), (x2, y2), color, 2)

        text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

        text_bg_y1 = max(0, y1 - text_height - 10)
        cv2.rectangle(image_display, (x1, text_bg_y1), (x1 + text_width, y1), color, -1)
        cv2.putText(image_display, text, (x1, y1 - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.basename(image_path)
    name, ext = os.path.splitext(base_filename)
    output_filename = f"{name}_detected{ext}"
    output_filepath = os.path.join(output_dir, output_filename)

    cv2.imwrite(output_filepath, image_display)
    print(f"✅ Output image saved to: {output_filepath}")


# Test with a specific image and ONNX model
if __name__ == "__main__":
    # --- IMPORTANT: Adjust these paths to your actual file locations ---
    image_to_process = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\images\test1.jpg"
    
    # Path to your ONNX model file.
    # Make sure this ONNX model is exported with the correct input shape (1,3,640,640)
    # and preferably with `export.py --include onnx --simplify` for best OpenVINO compatibility.
    onnx_model_path = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\best.onnx"
    
    # Directory where you want to save the OpenVINO XML and BIN files
    openvino_output_directory = "yolov8/openvino_model"
    
    device_to_use = "AUTO"  # Or "CPU", "GPU", "NPU"
    output_directory_for_images = "detected_images" 

    simple_helmet_detection(
        image_to_process, 
        onnx_model_path, 
        openvino_output_directory, 
        device=device_to_use, 
        output_dir=output_directory_for_images
    )


