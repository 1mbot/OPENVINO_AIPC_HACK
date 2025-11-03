import cv2
import numpy as np
import openvino.runtime as ov
import os
import sys
import time

# Import the OpenVINO conversion API
try:
    from openvino import convert_model
except ImportError:
    print("Warning: 'openvino.convert_model' not found directly. Trying 'openvino.tools.mo.convert'.")
    try:
        from openvino.tools.mo import convert_model
    except ImportError:
        print("Error: OpenVINO Model Optimizer API ('openvino.convert_model' or 'openvino.tools.mo.convert') not found.")
        print("Please ensure OpenVINO Development tools are installed (`pip install openvino-dev`).")
        sys.exit(1)

# --- Configuration ---
NUM_INFERENCE_RUNS = 10
DEVICES_TO_BENCHMARK = ["CPU", "GPU", "NPU"] # List of devices to benchmark

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

# Convert YOLO model to OpenVINO format using the Python API (ovc)
def convert_to_openvino_api(onnx_path, output_dir="yolo_openvino"):
    try:
        os.makedirs(output_dir, exist_ok=True)

        onnx_filename = os.path.basename(onnx_path)
        base_model_name = os.path.splitext(onnx_filename)[0]
        
        output_xml_path = os.path.join(output_dir, f"{base_model_name}.xml")

        print(f"Converting ONNX model '{onnx_path}' to OpenVINO IR (FP16) using OpenVINO API...")
        
        converted_model = convert_model(
            onnx_path,
            example_input=np.zeros([1, 3, 640, 640], dtype=np.float32), # Use example_input for shape
        )
        
        ov.save_model(converted_model, output_xml_path, compress_to_fp16=True)

        print(f"OpenVINO model conversion successful. Files saved to: {output_dir}")
        print(f"Generated XML: {output_xml_path}")
        print(f"Generated BIN: {output_xml_path.replace('.xml', '.bin')}")

        return output_dir
    except Exception as e:
        print(f"Error converting ONNX to OpenVINO IR via API: {e}")
        sys.exit(1)

# Run a single inference with OpenVINO
def run_single_openvino_inference(compiled_model, image_input):
    """
    Performs a single inference run on a pre-compiled model.
    """
    output_layer = compiled_model.output(0)
    result = compiled_model([image_input])[output_layer]
    return result

# 
# ==============================================================================
# ‼️ START OF FIXED SECTION ‼️
# This function is the correct one for your YOLOv8 model.
# ==============================================================================
#

# Post-process YOLOv8 output
def postprocess_yolov8_output(output, conf_threshold=0.5, iou_threshold=0.5):
    """
    Post-processes the raw output of a YOLOv8 ONNX/OpenVINO model.
    Assumes output shape is [batch, 4 + num_classes, num_proposals],
    e.g., [1, 20, 8400] for 16 classes.
    """
    
    print(f"Original output shape: {output.shape}")
    num_classes = output.shape[1] - 4
    print(f"Detected {num_classes} classes in the model output.")

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

#
# ==============================================================================
# ‼️ END OF FIXED SECTION ‼️
# ==============================================================================
#

# Main function for object detection and benchmarking
def simple_object_detection_and_benchmark(image_path, onnx_model_path, openvino_model_dir, devices_to_benchmark, num_runs, output_dir="output_images"):
    # Define paths for the expected OpenVINO XML and BIN files
    onnx_filename = os.path.basename(onnx_model_path)
    base_model_name = os.path.splitext(onnx_filename)[0]
    openvino_xml_path = os.path.join(openvino_model_dir, f"{base_model_name}.xml")
    openvino_bin_path = os.path.join(openvino_model_dir, f"{base_model_name}.bin")

    # Perform ONNX to OpenVINO conversion if IR not found
    if os.path.exists(openvino_xml_path) and os.path.exists(openvino_bin_path):
        print(f"Found existing OpenVINO model at {openvino_model_dir}. Skipping ONNX conversion.")
    else:
        print(f"OpenVINO model not found at {openvino_model_dir}. Attempting conversion from ONNX...")
        if not os.path.exists(onnx_model_path):
            print(f"Error: ONNX model not found at {onnx_model_path}. Please ensure it exists before conversion.")
            sys.exit(1)
        
        convert_to_openvino_api(onnx_model_path, output_dir=openvino_model_dir)
        
    print(f"Loading image for inference from: {image_path}")
    img_np_for_inference = load_image_for_inference(image_path)
    
    core = ov.Core()
    available_devices = core.available_devices
    print(f"\nOpenVINO available devices: {available_devices}")
    
    normalized_model_xml_path = os.path.normpath(openvino_xml_path)
    if not os.path.exists(normalized_model_xml_path):
        print(f"Error: OpenVINO model XML file not found at {normalized_model_xml_path}")
        sys.exit(1)
    
    model = core.read_model(model=normalized_model_xml_path)

    overall_benchmark_results = {}
    
    for device in devices_to_benchmark:
        inference_times = []
        try:
            if device not in available_devices:
                print(f"\nSkipping benchmarking for '{device}' as it's not available.")
                continue

            device_name = core.get_property(device, "FULL_DEVICE_NAME")
            print(f"\n--- Benchmarking on device: '{device}' ({device_name}) for {num_runs} runs ---")
            
            compiled_model = core.compile_model(model, device)
            
            # Warm-up run
            print("Performing warm-up run...")
            _ = run_single_openvino_inference(compiled_model, img_np_for_inference)
            
            print(f"Running {num_runs} inference iterations...")
            for i in range(num_runs):
                start_time = time.perf_counter() # Use perf_counter for more accurate timing
                result = run_single_openvino_inference(compiled_model, img_np_for_inference)
                end_time = time.perf_counter()
                inference_times.append(end_time - start_time)
                # print(f"  Run {i+1}: {inference_times[-1]*1000:.2f} ms") # Uncomment for verbose output

            avg_time_ms = np.mean(inference_times) * 1000
            min_time_ms = np.min(inference_times) * 1000
            max_time_ms = np.max(inference_times) * 1000
            
            overall_benchmark_results[device] = {
                "average_inference_time_ms": avg_time_ms,
                "min_inference_time_ms": min_time_ms,
                "max_inference_time_ms": max_time_ms,
                "num_runs": num_runs,
                "last_result_for_postprocessing": result # Keep the last result for post-processing
            }
            
            print(f"\n--- Benchmark Summary for {device} ---")
            print(f"  Average Inference Time: {avg_time_ms:.2f} ms")
            print(f"  Min Inference Time: {min_time_ms:.2f} ms")
            print(f"  Max Inference Time: {max_time_ms:.2f} ms")
            print("-" * 40)

        except Exception as e:
            print(f"Error during OpenVINO inference benchmark on {device}: {e}")
            overall_benchmark_results[device] = {"status": "Failed", "error": str(e)}

    print("\n--- All Benchmarking Complete ---")
    if overall_benchmark_results:
        print("\nOverall Benchmark Results:")
        for device, data in overall_benchmark_results.items():
            if data.get("status") == "Failed":
                print(f"Device: {device} - FAILED ({data['error']})")
            else:
                print(f"Device: {device}")
                print(f"  Avg Time: {data['average_inference_time_ms']:.2f} ms")
                print(f"  Min Time: {data['min_inference_time_ms']:.2f} ms")
                print(f"  Max Time: {data['max_inference_time_ms']:.2f} ms")
                print(f"  Number of Runs: {data['num_runs']}")
            print()
    else:
        print("No successful benchmarks were recorded.")

    # --- Post-processing and visualization (using the last result from the first successfully benchmarked device) ---
    final_detection_result = None
    for device in devices_to_benchmark:
        if device in overall_benchmark_results and "last_result_for_postprocessing" in overall_benchmark_results[device]:
            final_detection_result = overall_benchmark_results[device]["last_result_for_postprocessing"]
            print(f"\nPerforming post-processing and visualization using results from device: {device}")
            break # Use the first available successful result

    if final_detection_result is not None:
        print("Post-processing inference output...")
        
        # 
        # --- Calling the FIXED YOLOv8 post-processing function ---
        #
        boxes, scores, class_ids = postprocess_yolov8_output(final_detection_result)
        
        # 
        # --- Using the 16 labels from your data.yaml file ---
        #
        labels = [
            "Person", "Car", "Bus", "Bicycle", "Motorcycle", "Traffic light",
            "Stop sign", "Chair", "Box", "Fire hydrant", "Door", "Window",
            "Laptop", "Mobile phone", "Book", "Human face"
        ]
        
        # Generate some colors for the classes
        colors = []
        np.random.seed(42) # for reproducible colors
        for i in range(len(labels)):
            colors.append(tuple(np.random.randint(0, 256, 3).tolist()))

        
        detections = []
        if len(boxes) > 0:
            for i in range(len(class_ids)):
                if class_ids[i] < len(labels):
                    label_name = labels[class_ids[i]]
                else:
                    label_name = f"Unknown Class {class_ids[i]}"
                
                box_int = [int(coord) for coord in boxes[i]]
                detections.append((label_name, scores[i], box_int, class_ids[i]))
        
        print("\n--- Detections ---")
        if detections:
            for label, score, box, class_id in detections:
                print(f"Class: {label} (ID: {class_id}), Confidence: {score:.4f}, Box: {box}")
        else:
            print("No objects detected above confidence threshold.")


        print("\nDrawing detections on image and saving...")
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Error: Could not load original image for drawing: {image_path}")
            return

        image_display = cv2.resize(original_image, (640, 640)) 

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2

        for label, score, box, class_id in detections:
            x1, y1, x2, y2 = box
            
            color = colors[class_id] # Get color based on class ID

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
    else:
        print("\nNo successful inference result available for post-processing and visualization.")


# Test with a specific image and ONNX model
if __name__ == "__main__":
    # --- IMPORTANT: Adjust these paths to your actual file locations ---
    image_to_process = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\images\test1.jpg"
    
    onnx_model_path = r"C:\Users\ce_jo\AIPC_Hackathon\Hack_Work\devcloud\SemiX\yolo\best.onnx"
    
    openvino_output_directory = "yolo8_16class/openvino_model" # Changed this to be more descriptive
    
    output_directory_for_images = "detected_images" 

    simple_object_detection_and_benchmark(
        image_to_process, 
        onnx_model_path, 
        openvino_output_directory, 
        DEVICES_TO_BENCHMARK, # Use the global list of devices
        NUM_INFERENCE_RUNS,   # Use the global number of runs
        output_dir=output_directory_for_images
    )