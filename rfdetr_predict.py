
import os
import sys
import argparse
import cv2
import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

def predict_image(model_path, image_path, output_path=None, threshold=0.2, model_type="base"):
    """
    Run RF-DETR prediction on an image
    
    Args:
        model_path: Path to the RF-DETR model weights
        image_path: Path to the input image
        output_path: Path to save the output image with detections (optional)
        threshold: Detection confidence threshold
        model_type: Either "base" (ResNet-50) or "large" (ResNet-101)
    
    Returns:
        List of detection dictionaries with 'box', 'class', and 'score' keys
    """
    # Load model
    if model_type.lower() == "large":
        model = RFDETRLarge(pretrain_weights=model_path)
        print(f"Loaded RF-DETR Large model from {model_path}")
    else:
        model = RFDETRBase(pretrain_weights=model_path)
        print(f"Loaded RF-DETR Base model from {model_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Convert to RGB (RF-DETR expects RGB input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run prediction
    detections = model.predict(image_rgb, threshold=threshold)
    print(f"Detected {len(detections)} objects with confidence >= {threshold}")
    
    # Draw results on image
    if output_path:
        image_with_boxes = image.copy()
        for det in detections:
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            label = det['class']
            score = det['score']
            
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, f"{label}: {score:.2f}", 
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image_with_boxes)
        print(f"Saved detection results to {output_path}")
    
    return detections

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF-DETR object detection on an image")
    parser.add_argument("--model", required=True, help="Path to RF-DETR model weights")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output image with detections")
    parser.add_argument("--threshold", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--model-type", choices=["base", "large"], default="base", 
                        help="Model type: base (ResNet-50) or large (ResNet-101)")
    
    args = parser.parse_args()
    predict_image(
        args.model, 
        args.image, 
        args.output, 
        args.threshold, 
        args.model_type
    )
