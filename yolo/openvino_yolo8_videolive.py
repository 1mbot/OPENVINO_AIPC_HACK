import cv2
import numpy as np
import openvino.runtime as ov
import os
import sys
import time

# Load and preprocess image/frame for inference
def preprocess_for_inference(frame):
    """Preprocesses a single image frame for OpenVINO model inference."""
    try:
        # Convert to RGB, resize, transpose, normalize for model input
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 640))
        img_chw = img_resized.transpose(2, 0, 1)  # HWC to CHW
        img_normalized = img_chw.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)  # Add batch dimension
        return img_batch
    except Exception as e:
        print(f"Error preprocessing frame for inference: {e}")
        return None

# Run inference with OpenVINO
def run_openvino_inference(compiled_model, image_input):
    """
    Runs inference on the preprocessed image using an already compiled OpenVINO model.
    """
    try:
        output_layer = compiled_model.output(0)
        start_time = time.perf_counter() # Use perf_counter for more accuracy
        result = compiled_model([image_input])[output_layer]
        end_time = time.perf_counter()
        time_taken = end_time - start_time
        return result, time_taken
    except Exception as e:
        print(f"Error during OpenVINO inference: {e}")
        return None, 0.0

#
# ==============================================================================
# ‼️ START OF FIXED SECTION ‼️
# This is the correct post-processing function for your 16-class YOLOv8 model.
# ==============================================================================
#
def postprocess_yolov8_output(output, conf_threshold=0.5, iou_threshold=0.5):
    """
    Post-processes the raw output of a YOLOv8 ONNX/OpenVINO model.
    Assumes output shape is [batch, 4 + num_classes, num_proposals],
    e.g., [1, 20, 8400] for 16 classes.
    """
    
    # print(f"Original output shape: {output.shape}") # Uncomment for debugging
    num_classes = output.shape[1] - 4
    # print(f"Detected {num_classes} classes in the model output.") # Uncomment for debugging

    # Transpose the output from [1, 20, 8400] to [1, 8400, 20]
    try:
        if output.shape[1] > output.shape[2]:
             # Shape is already [1, 8400, 20]
             pass
        else:
             output = output.transpose(0, 2, 1) # [1, 20, 8400] -> [1, 8400, 20]
    except Exception as e:
        print(f"Error transposing output. Shape was {output.shape}. Error: {e}")
        return np.array([]), np.array([]), np.array([])

    # output is now [1, num_proposals, 4 + num_classes], e.g. [1, 8400, 20]
    output_batch = output[0] # Shape [8400, 20]
    
    # Parse boxes and class scores
    boxes_xywh = output_batch[:, :4]    # [8400, 4]
    class_scores_all = output_batch[:, 4:] # [8400, 16]
    
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
    x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    widths = boxes_xywh[:, 2]
    heights = boxes_xywh[:, 3]
    boxes_for_nms = np.stack([x1, y1, widths, heights], axis=1)

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    
    if len(indices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    indices = indices.flatten()
    boxes_xywh = boxes_xywh[indices]
    scores = scores[indices]
    class_ids = class_ids[indices]
    
    # Convert final boxes to (x1, y1, x2, y2)
    final_boxes_xyxy = np.copy(boxes_xywh)
    final_boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    final_boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    final_boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    final_boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

    return final_boxes_xyxy, scores, class_ids
#
# ==============================================================================
# ‼️ END OF FIXED SECTION ‼️
# ==============================================================================
#

def draw_detections_on_frame(frame, detections, labels, colors):
    """Draws bounding boxes and labels on a single image frame."""
    frame_width = int(frame.shape[1])
    frame_height = int(frame.shape[0])
    
    # Scale boxes to original frame size (since we resized for inference)
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
        
        #
        # --- FIXED: Generic color logic for 16 classes ---
        #
        try:
            # Find the index of the label to get its color
            class_index = labels.index(label)
            color = colors[class_index]
        except ValueError:
            # Default color if label isn't in the list
            color = (0, 255, 0) # Green
            
        cv2.rectangle(frame, (x1_scaled, y1_scaled), (x2_scaled, y2_scaled), color, 2)
        
        text = f"{label}: {score:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
        
        text_bg_y1 = max(0, y1_scaled - text_height - 10)
        cv2.rectangle(frame, (x1_scaled, text_bg_y1), (x1_scaled + text_width, y1_scaled), color, -1)
        
        cv2.putText(frame, text, (x1_scaled, y1_scaled - 5), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return frame

def run_camera_detection(model_xml_path, device="AUTO"):
    """
    Main function to run live detection on the PC camera feed.
    """
    
    #
    # --- FIXED: Using the 16 labels from your data.yaml file ---
    #
    labels = [
        "Person", "Car", "Bus", "Bicycle", "Motorcycle", "Traffic light",
        "Stop sign", "Chair", "Box", "Fire hydrant", "Door", "Window",
        "Laptop", "Mobile phone", "Book", "Human face"
    ]
    
    # --- FIXED: Generate a list of colors for all 16 classes ---
    colors = []
    np.random.seed(42) # for reproducible colors
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
    print(f"Running inference on device: {device_to_use} ({device_name})")
    
    normalized_model_xml_path = os.path.normpath(model_xml_path)
    if not os.path.exists(normalized_model_xml_path):
        print(f"Error: OpenVINO model XML file not found at {normalized_model_xml_path}")
        print(f"Please ensure 'best.onnx' was converted and 'best.xml' exists in that path.")
        sys.exit(1)
    
    model = core.read_model(model=normalized_model_xml_path)
    compiled_model = core.compile_model(model, device_to_use)
    
    # Start video capture from PC camera (index 0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from camera. Please check your camera connection.")
        return
    
    print("\nStarting camera feed... Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera.")
            break
        
        # Preprocess frame and run inference
        img_np_for_inference = preprocess_for_inference(frame)
        if img_np_for_inference is not None:
            output, time_taken = run_openvino_inference(compiled_model, img_np_for_inference)
            
            if output is None:
                continue

            #
            # --- FIXED: Calling the correct post-processing function ---
            #
            boxes, scores, class_ids = postprocess_yolov8_output(output)
            
            detections = []
            if len(boxes) > 0:
                for i in range(len(class_ids)):
                    if class_ids[i] < len(labels):
                        label_name = labels[class_ids[i]]
                    else:
                        label_name = f"Unknown: {class_ids[i]}"
                    detections.append((label_name, scores[i], boxes[i]))
            
            #
            # --- FIXED: Passing 'colors' to the draw function ---
            #
            annotated_frame = draw_detections_on_frame(frame, detections, labels, colors)
            
            # Add timing information to the frame
            fps = 1.0 / time_taken if time_taken > 0 else 0
            inference_text = f"Inference: {time_taken*1000:.2f} ms ({fps:.1f} FPS)"
            cv2.putText(annotated_frame, inference_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('PC Camera Detection (YOLOv8)', annotated_frame)
            
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Camera feed stopped.")


# Test with a specific image and pre-converted model
if __name__ == "__main__":
    # --- Configuration ---
    #
    # --- FIXED: Updated path to match previous examples ---
    #
    openvino_model_dir = "yolo8_16class/openvino_model"
    
    #
    # --- FIXED: 'best.onnx' converts to 'best.xml' ---
    #
    openvino_xml_path = os.path.join(openvino_model_dir, "best.xml")
    
    device_to_use = "AUTO"  # Change to "CPU", "GPU", "NPU" or "AUTO"

    # --- Run on PC Camera ---
    print("\n\n--- Starting live detection on PC camera ---")
    run_camera_detection(openvino_xml_path, device=device_to_use)