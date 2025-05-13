
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
    from PIL import Image
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Running RF-DETR prediction on {image_path} with model {model_path}")
    
    # Convert OpenCV image to PIL image if needed
    if isinstance(image_rgb, np.ndarray):
        pil_image = Image.fromarray(image_rgb)
    else:
        pil_image = image_rgb
    
    try:
        # Run prediction with error handling
        detections = model.predict(pil_image, threshold=threshold)
        
        # Log detection details
        if hasattr(detections, 'class_id'):
            detection_count = len(detections.class_id) if hasattr(detections.class_id, '__len__') else 1
            logger.info(f"Detected {detection_count} objects with confidence >= {threshold} (structured format)")
        elif isinstance(detections, list):
            logger.info(f"Detected {len(detections)} objects with confidence >= {threshold} (list format)")
        else:
            logger.info(f"Detection result: {type(detections)}")
    except Exception as e:
        logger.error(f"Error during model prediction: {str(e)}")
        # Return empty detections instead of raising to avoid breaking the UI
        return []
    
    # Draw results on image
    if output_path:
        image_with_boxes = image.copy()
        
        # Process detections based on their format
        if hasattr(detections, 'class_id') and hasattr(detections, 'confidence') and hasattr(detections, 'xyxy'):
            # New format with structured attributes
            labels = [
                f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                for class_id, confidence
                in zip(detections.class_id, detections.confidence)
            ]
            
            for i, (class_id, bbox) in enumerate(zip(detections.class_id, detections.xyxy)):
                class_name = COCO_CLASSES.get(class_id, f"Class {class_id}")
                x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for cv2
                
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, labels[i], 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Original dictionary format
            for det in detections:
                box = det['box']
                # Handle different box formats safely
                if hasattr(box, 'tolist') and callable(getattr(box, 'tolist')):
                    # Handle numpy array
                    box_list = box.tolist()
                    x1, y1, x2, y2 = map(int, box_list)
                elif isinstance(box, (list, tuple, np.ndarray)):
                    # Handle list/tuple/array where elements might be numpy arrays
                    try:
                        # Check if individual elements are numpy arrays that need item() extraction
                        if hasattr(box[0], 'item') and callable(getattr(box[0], 'item')):
                            x1 = int(box[0].item())
                            y1 = int(box[1].item())
                            x2 = int(box[2].item())
                            y2 = int(box[3].item())
                        else:
                            # Standard conversion
                            x1 = int(float(box[0]))
                            y1 = int(float(box[1])) 
                            x2 = int(float(box[2]))
                            y2 = int(float(box[3]))
                    except (TypeError, ValueError) as e:
                        print(f"Error converting box coordinates: {e}, box: {box}")
                        raise  # Re-raise to handle at higher level
                else:
                    # Unknown format
                    raise ValueError(f"Unexpected box format: {type(box)}")
                
                # Get class and score
                if 'class_id' in det:
                    class_id = det['class_id']
                    label = COCO_CLASSES.get(class_id, f"Class {class_id}")
                else:
                    label = det.get('class', 'Object')
                
                score = det.get('score', 1.0)
                
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, f"{label}: {score:.2f}", 
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image_with_boxes)
        print(f"Saved detection results to {output_path}")
    
    return detections

def sync_rfdetr_metrics_to_mlflow(job_id, mlflow_run_id, mlflow_tracking_uri):
    """
    Sincronizza le metriche di un modello RF-DETR con MLFlow
    
    Args:
        job_id: ID del job di training
        mlflow_run_id: ID della run MLFlow
        mlflow_tracking_uri: URI del server MLFlow
    """
    import logging
    import mlflow
    import os
    
    logger = logging.getLogger(__name__)
    
    try:
        # Configura MLFlow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Verifica se la run esiste
        try:
            run = mlflow.get_run(mlflow_run_id)
            if not run:
                logger.warning(f"Run MLFlow {mlflow_run_id} non trovata")
                return False
        except Exception as e:
            logger.warning(f"Errore nel recupero della run MLFlow: {str(e)}")
            return False
        
        # Trova il file di metriche generato durante il training
        training_dir = os.path.join(os.getcwd(), f"training_jobs/job_{mlflow_run_id[:8]}")
        metrics_path = os.path.join(training_dir, "metrics.json")
        
        if os.path.exists(metrics_path):
            import json
            with open(metrics_path, 'r') as f:
                try:
                    metrics = json.load(f)
                    
                    # Ottieni le metriche finali
                    final_metrics = {}
                    for metric_name, values in metrics.items():
                        if isinstance(values, list) and values:
                            final_metrics[metric_name] = values[-1]  # Prendi l'ultimo valore
                    
                    # Registra le metriche su MLFlow
                    with mlflow.start_run(run_id=mlflow_run_id):
                        for metric_name, value in final_metrics.items():
                            try:
                                mlflow.log_metric(metric_name, value)
                                logger.info(f"Metrica sincronizzata: {metric_name}={value}")
                            except Exception as e:
                                logger.warning(f"Errore sincronizzazione metrica {metric_name}: {str(e)}")
                        
                        # Registra anche il modello
                        model_path = os.path.join(training_dir, "weights", "best_model.pth")
                        if os.path.exists(model_path):
                            mlflow.log_artifact(model_path, "model")
                            logger.info(f"Artefatto modello sincronizzato: {model_path}")
                        
                        # Registra eventuali immagini di test
                        for root, _, files in os.walk(training_dir):
                            for file in files:
                                if file.endswith(('.png', '.jpg')) and not file.startswith('.'):
                                    img_path = os.path.join(root, file)
                                    mlflow.log_artifact(img_path, "plots")
                                    logger.info(f"Immagine sincronizzata: {img_path}")
                    
                    return True
                except json.JSONDecodeError:
                    logger.warning(f"File di metriche non valido: {metrics_path}")
                    return False
        else:
            logger.warning(f"File di metriche non trovato: {metrics_path}")
            return False
    
    except Exception as e:
        logger.exception(f"Errore durante la sincronizzazione delle metriche: {str(e)}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF-DETR object detection on an image")
    parser.add_argument("--model", required=True, help="Path to RF-DETR model weights")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--output", help="Path to save output image with detections")
    parser.add_argument("--threshold", type=float, default=0.2, help="Detection confidence threshold")
    parser.add_argument("--model-type", choices=["base", "large"], default="base", 
                        help="Model type: base (ResNet-50) or large (ResNet-101)")
    parser.add_argument("--sync-mlflow", action="store_true", help="Sync metrics to MLFlow")
    parser.add_argument("--job-id", type=int, help="Training job ID for syncing metrics")
    parser.add_argument("--mlflow-run-id", help="MLFlow run ID for syncing metrics")
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5001", help="MLFlow tracking URI")
    
    args = parser.parse_args()
    
    if args.sync_mlflow and args.job_id and args.mlflow_run_id:
        success = sync_rfdetr_metrics_to_mlflow(args.job_id, args.mlflow_run_id, args.mlflow_tracking_uri)
        print(f"Sincronizzazione metriche MLFlow: {'Successo' if success else 'Fallita'}")
    else:
        predict_image(
            args.model, 
            args.image, 
            args.output, 
            args.threshold, 
            args.model_type
        )
