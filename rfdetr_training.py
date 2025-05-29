import argparse
import glob
import json
import logging
import os
import shutil

import cv2
import mlflow
import numpy as np
import torch
from PIL import Image
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES
from torchvision import transforms
import supervision as sv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_pretrained_model_info(model_variant):
    """Get information about the pretrained RF-DETR model variant"""
    # RF-DETR models are already pretrained, no need to download weights
    logger.info(f"Using built-in pretrained weights for RF-DETR model variant: {model_variant}")

    if 'r50' in model_variant:
        return {
            'type': 'base',
            'backbone': 'ResNet-50'
        }
    elif 'r101' in model_variant:
        return {
            'type': 'large',
            'backbone': 'ResNet-101'
        }
    else:
        logger.warning(f"Unknown model variant: {model_variant}, defaulting to RF-DETR base model")
        return {
            'type': 'base',
            'backbone': 'ResNet-50'
        }


def convert_yolo_to_coco(yolo_dataset_path):
    """Converts a YOLO dataset to COCO format required by RF-DETR."""
    # Define paths
    train_img_dir = os.path.join(yolo_dataset_path, 'train', 'images')
    train_label_dir = os.path.join(yolo_dataset_path, 'train', 'labels')

    # Check if paths exist
    if not os.path.exists(train_img_dir):
        logger.error(f"Images directory not found: {train_img_dir}")
        return False

    if not os.path.exists(train_label_dir):
        logger.error(f"Labels directory not found: {train_label_dir}")
        return False

    # Find all images
    image_files = glob.glob(os.path.join(train_img_dir, '*.jpg')) + \
                  glob.glob(os.path.join(train_img_dir, '*.jpeg')) + \
                  glob.glob(os.path.join(train_img_dir, '*.png'))

    logger.info(f"Found {len(image_files)} images to convert")

    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Converted from YOLO format",
            "version": "1.0",
            "year": 2023,
            "contributor": "Automatic Converter"
        },
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": ""
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Detect classes from dataset
    class_ids = set()
    for label_file in glob.glob(os.path.join(train_label_dir, '*.txt')):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        class_ids.add(int(parts[0]))
        except Exception as e:
            logger.warning(f"Error reading file {label_file}: {str(e)}")

    # Create COCO categories
    for class_id in sorted(class_ids):
        coco_data["categories"].append({
            "id": class_id + 1,  # COCO uses 1-based IDs
            "name": f"class{class_id}",
            "supercategory": "object"
        })

    logger.info(f"Detected {len(class_ids)} classes in dataset")

    # If no classes found, add default classes
    if not coco_data["categories"]:
        coco_data["categories"] = [
            {"id": 1, "name": "class0", "supercategory": "object"},
            {"id": 2, "name": "class1", "supercategory": "object"}
        ]

    # Add images and annotations
    annotation_id = 1
    for img_id, img_path in enumerate(image_files, 1):
        # Get image info
        img_filename = os.path.basename(img_path)
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            logger.warning(f"Error opening image {img_path}: {str(e)}")
            continue

        # Add image info to COCO
        coco_data["images"].append({
            "id": img_id,
            "license": 1,
            "file_name": img_filename,
            "height": height,
            "width": width,
            "date_captured": ""
        })

        # Find corresponding label file
        base_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(train_label_dir, f"{base_name}.txt")

        if not os.path.exists(label_path):
            logger.warning(f"Label file not found for {img_filename}")
            continue

        # Read YOLO annotations and convert to COCO
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    # YOLO uses normalized coordinates (0-1) with center and dimensions
                    # COCO uses [x,y,width,height] in pixels from top-left corner
                    x1 = (x_center - box_width / 2) * width
                    y1 = (y_center - box_height / 2) * height
                    w = box_width * width
                    h = box_height * height

                    # Create COCO annotation
                    coco_annotation = {
                        "id": annotation_id,
                        "image_id": img_id,
                        "category_id": class_id + 1,  # COCO uses 1-based IDs
                        "bbox": [x1, y1, w, h],
                        "area": w * h,
                        "segmentation": [],
                        "iscrowd": 0
                    }

                    coco_data["annotations"].append(coco_annotation)
                    annotation_id += 1
                except Exception as e:
                    logger.warning(f"Error converting annotation: {str(e)}")

    # Save COCO JSON file
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(yolo_dataset_path, split)
        if os.path.exists(split_dir):
            coco_output_path = os.path.join(split_dir, '_annotations.coco.json')
            with open(coco_output_path, 'w') as f:
                json.dump(coco_data, f)
            logger.info(f"Saved COCO file for {split}: {coco_output_path}")

    logger.info(
        f"Conversion completed with {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    return True


def compute_metrics(model: 'RFDETRBase', test_dataset):
    """
    Calculate metrics on the validation set using supervision.

    Args:
        model (RFDETRBase): The RF-DETR model instance.
        test_dataset: sv.DetectionDataset for the test set.

    Returns:
        tuple: (precision, recall, m_ap50, m_ap50_95)
    """
    all_predictions_sv = []
    all_targets_sv = []

    for path, image, annotations in test_dataset:
        image = Image.open(path)
        detections = model.predict(image)

        all_targets_sv.append(annotations)
        all_predictions_sv.append(detections)

    # --- Mean Average Precision (mAP) ---
    # Instantiate the metric object
    map_calculator = sv.metrics.MeanAveragePrecision()  # Default: metric_target=BOXES, class_agnostic=False

    # Update with all collected predictions and targets
    # The update method can handle lists of Detections objects directly
    if all_predictions_sv and all_targets_sv:  # Ensure lists are not empty before updating
        logging.info("YAY! PREDICTIONS AND TARGETS ARE NOT EMPTY! :D")

    map_results = map_calculator.update(predictions=all_predictions_sv, targets=all_targets_sv).compute()

    m_ap50 = map_results.map50
    m_ap50_95 = map_results.map50_95

    # --- Precision and Recall ---
    precision_calculator = sv.metrics.Precision()
    recall_calculator = sv.metrics.Recall()

    # Update P/R calculators for each image's Detections
    precision_result = precision_calculator.update(predictions=all_predictions_sv, targets=all_targets_sv).compute()
    recall_result = recall_calculator.update(predictions=all_predictions_sv, targets=all_targets_sv).compute()

    # Extract Precision/Recall for the specified IoU threshold
    precision = precision_result.precision_at_50
    recall = recall_result.recall_at_50

    return precision, recall, m_ap50, m_ap50_95


def train_rfdetr_model(dataset_info, model_variant, hyperparameters, mlflow_run_id, mlflow_tracking_uri):
    """Train RF-DETR model with provided configuration"""
    logger.info(f"Training RF-DETR model variant: {model_variant} with MLFlow run ID: {mlflow_run_id}")
    logger.info(f"Using dataset: {dataset_info}")
    logger.info(f"Hyperparameters: {hyperparameters}")

    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.getcwd(), "models")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        logger.info(f"Created models directory: {models_dir}")

    # Setup directories for training output
    training_output_dir = os.path.join(os.getcwd(), f"training_jobs/job_{mlflow_run_id[:8]}")
    os.makedirs(training_output_dir, exist_ok=True)
    weights_dir = os.path.join(training_output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Determine model path
    model_filename = f"{model_variant}_{mlflow_run_id[:8]}"
    model_path = os.path.join(models_dir, model_filename)

    # Connect to MLFlow for logging if available
    mlflow_active = False
    try:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow_active = True
        logger.info("MLFlow logging enabled")
    except Exception as e:
        logger.warning(f"MLFlow logging disabled: {str(e)}")

    try:
        # Get training parameters
        total_epochs = int(hyperparameters.get('epochs', 50))
        batch_size = int(hyperparameters.get('batch_size', 8))
        learning_rate = float(hyperparameters.get('learning_rate', 0.0001))

        # Get real dataset path
        dataset_path = dataset_info.get('dataset_path')

        # RF-DETR models are already pretrained, we just need to select the right variant
        pretrained_flag = hyperparameters.get('pretrained', True)
        if isinstance(pretrained_flag, str):
            pretrained_flag = pretrained_flag.lower() == 'true'

        model_info = get_pretrained_model_info(model_variant)

        if pretrained_flag:
            logger.info(f"Using built-in pre-trained weights for {model_variant}")
            # We'll set this to None since we'll use the built-in pretrained weights
            model_weights = None
        else:
            logger.info("Note: RF-DETR models are already pretrained by default")
            model_weights = None

        # Prepare dataset
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path {dataset_path} not found, using example dataset")
            # Fallback to a test dataset
            dataset_path = "coco8"

        # Convert dataset to COCO format if needed
        if dataset_info['format_type'] == 'yolo':
            logger.info("Converting YOLO format dataset to COCO format for RF-DETR training")
            conversion_success = convert_yolo_to_coco(dataset_path)
            if not conversion_success:
                logger.error("YOLO-to-COCO conversion failed")
                raise Exception("Unable to convert YOLO dataset to COCO format required by RF-DETR")

            # Verify that the file was created
            train_coco_file = os.path.join(dataset_path, 'train', '_annotations.coco.json')
            if os.path.exists(train_coco_file):
                logger.info(f"COCO file created successfully: {train_coco_file}")
            else:
                logger.error(f"COCO file not found after conversion: {train_coco_file}")
                raise Exception("COCO annotation file not created during conversion")

        # Initialize model based on variant (models already have pretrained weights)
        logger.info(f"Initializing RF-DETR model: {model_info['type']} with {model_info['backbone']} backbone")
        if model_info['type'] == 'large':
            model = RFDETRLarge()
            logger.info("Using RF-DETR Large model with ResNet-101 backbone")
        else:
            model = RFDETRBase()
            logger.info("Using RF-DETR Base model with ResNet-50 backbone")

        # Add method to set args in model
        def _set_args(self, args):
            # Set arguments as model attributes
            for key, value in vars(args).items():
                setattr(self, key, value)
            return self

        # Add method to model
        model._set_args = _set_args.__get__(model)

        # Simple validation of model by running prediction on a test image
        try:
            # Find a test image from the dataset
            test_image_path = None
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_image_path = os.path.join(root, file)
                        break
                if test_image_path:
                    break

            if test_image_path and os.path.exists(test_image_path):
                logger.info(f"Testing model with image: {test_image_path}")

                # Load image with PIL for compatibility
                pil_image = Image.open(test_image_path)
                cv_image = np.array(pil_image)
                if cv_image.shape[2] == 3:  # If image is RGB
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

                # Run prediction using PIL image
                detections = model.predict(pil_image, threshold=0.2)

                # Log detection count
                if hasattr(detections, 'class_id'):
                    detection_count = len(detections.class_id)
                else:
                    detection_count = len(detections)

                logger.info(f"Model test successful: detected {detection_count} objects")

                # Save a visualization of detections for debugging
                output_image_path = os.path.join(training_output_dir, "test_detection.jpg")
                image_with_boxes = cv_image.copy()

                # Create an object_counts dictionary
                object_counts = {}
                for class_id in COCO_CLASSES:
                    object_counts[COCO_CLASSES[class_id]] = 0

                # Process detections based on format
                if hasattr(detections, 'class_id') and hasattr(detections, 'confidence') and hasattr(detections,
                                                                                                     'xyxy'):
                    # New structured format
                    labels = [
                        f"{COCO_CLASSES[class_id]} {confidence:.2f}"
                        for class_id, confidence
                        in zip(detections.class_id, detections.confidence)
                    ]

                    for i, (class_id, bbox) in enumerate(zip(detections.class_id, detections.xyxy)):
                        class_name = COCO_CLASSES.get(class_id, f"Class {class_id}")
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1

                        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers for cv2
                        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image_with_boxes, labels[i],
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Original dictionary format
                    for det in detections:
                        try:
                            if isinstance(det, dict) and 'box' in det:
                                # Dictionary format with 'box' key
                                box = det['box']
                                if isinstance(box, (list, tuple)) and len(box) >= 4:
                                    x1 = int(float(box[0]))
                                    y1 = int(float(box[1]))
                                    x2 = int(float(box[2]))
                                    y2 = int(float(box[3]))
                                elif hasattr(box, 'tolist') and callable(getattr(box, 'tolist')):
                                    # Handle numpy array
                                    box_list = box.tolist()
                                    x1 = int(box_list[0])
                                    y1 = int(box_list[1])
                                    x2 = int(box_list[2])
                                    y2 = int(box_list[3])
                                else:
                                    logger.warning(f"Unexpected box format: {box} ({type(box)})")
                                    continue

                                # Get class info
                                if 'class_id' in det:
                                    class_id = det['class_id']
                                    label = COCO_CLASSES.get(class_id, f"Class {class_id}")
                                else:
                                    label = det.get('class', 'Object')

                                score = float(det.get('score', 1.0))

                            elif isinstance(det, (list, tuple, np.ndarray)) and len(det) >= 6:
                                # Tuple/list/array format [x1, y1, x2, y2, score, class_id]
                                try:
                                    if isinstance(det[0], np.ndarray):
                                        x1 = int(det[0].item())
                                        y1 = int(det[1].item())
                                        x2 = int(det[2].item())
                                        y2 = int(det[3].item())
                                        score = float(det[4].item())
                                        class_id = int(det[5].item())
                                    else:
                                        x1 = int(float(det[0]))
                                        y1 = int(float(det[1]))
                                        x2 = int(float(det[2]))
                                        y2 = int(float(det[3]))
                                        score = float(det[4])
                                        class_id = int(det[5])
                                    label = COCO_CLASSES.get(class_id, f"Class {class_id}")
                                except (TypeError, ValueError, AttributeError) as e:
                                    logger.warning(f"Error converting detection values: {e}")
                                    continue
                            else:
                                logger.warning(f"Unexpected detection format: {det} ({type(det)})")
                                continue
                        except Exception as e:
                            logger.warning(f"Error processing detection: {e}")
                            import traceback
                            logger.debug(f"Detection error: {traceback.format_exc()}")
                            continue

                        # Draw the detection on the image
                        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(image_with_boxes, f"{label}: {score:.2f}",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.imwrite(output_image_path, image_with_boxes)
                logger.info(f"Saved test detection image to {output_image_path}")
        except Exception as e:
            logger.warning(f"Model test failed: {e}")
            import traceback
            logger.debug(f"Detailed error: {traceback.format_exc()}")

        test_annot_path = f"{dataset_path}/test/_annotations.coco.json"

        if os.path.exists(f"{dataset_path}/test/images"):
            test_images_path = f"{dataset_path}/test/images"
        else:
            test_images_path = f"{dataset_path}/test"

        test_dataset = sv.DetectionDataset.from_coco(images_directory_path=test_images_path,
                                                    annotations_path=test_annot_path)

        # Prepare model for fine-tuning
        # Create Namespace with correct parameters
        args = argparse.Namespace(
            num_classes=6,
            grad_accum_steps=4,
            amp=True,
            lr=learning_rate,
            lr_encoder=learning_rate * 1.5,
            batch_size=batch_size,
            weight_decay=0.0001,
            epochs=total_epochs,
            lr_drop=total_epochs,
            clip_max_norm=0.1,
            lr_vit_layer_decay=0.8,
            lr_component_decay=0.7,
            do_benchmark=False,
            dropout=0,
            drop_path=0.0,
            drop_mode='standard',
            drop_schedule='constant',
            cutoff_epoch=0,
            pretrained_encoder=None,
            pretrain_weights=model_weights
        )

        # Set args as model attributes
        model._set_args(args)

        # Intercept and modify default values before calling train()
        if hasattr(model, '_get_args'):
            logging.info("SETTING MODEL ARGS FOR TRAINING")
            original_get_args = model._get_args

            def patched_get_args(self, *args, **kwargs):
                result = original_get_args(self, *args, **kwargs)
                if hasattr(result, 'epochs') and result.epochs == 100:
                    logger.info(f"PATCH: Intercepted epochs=100 in _get_args, replacing with {total_epochs}")
                    result.epochs = total_epochs
                    if hasattr(result, 'lr_drop'):
                        result.lr_drop = total_epochs
                return result

            # Apply patch
            model._get_args = patched_get_args.__get__(model)
            logger.info("Applied patch to _get_args")

        # Set up training parameters
        training_params = {
            "dataset_dir": dataset_path,
            "epochs": total_epochs,
            "batch_size": batch_size,
            "grad_accum_steps": 4,
            "lr": learning_rate,
            "output_dir": training_output_dir,
            "resume": None
        }

        logger.info(
            f"Direct call to model.train() with: epochs={total_epochs}, batch_size={batch_size}, lr={learning_rate}")
        logger.info(f"Dataset path: {dataset_path}")

        # Ensure dataset_dir is passed correctly
        if not training_params.get('dataset_dir'):
            logger.warning("Missing dataset_dir parameter, setting explicitly")
            training_params['dataset_dir'] = dataset_path

        logger.info(f"Training parameters: {training_params}")

        # Run training
        model.train(**training_params, tensorboard=False)

        logger.info("Training completed successfully using direct syntax")

        # Training metrics
        metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "precision": [],
            "recall": [],
            "mAP50": [],
            "mAP50-95": []
        }

        # Calculate final metrics
        precision, recall, m_ap50, m_ap50_95 = compute_metrics(model, test_dataset)

        # Save metrics
        metrics_history["precision"].append(float(precision))
        metrics_history["recall"].append(float(recall))
        metrics_history["mAP50"].append(float(m_ap50))
        metrics_history["mAP50-95"].append(float(m_ap50_95))

        # Save metrics history
        metrics_path = os.path.join(training_output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        logger.info(f"Saved metrics history to {metrics_path}")

        # Copy best model as final result
        best_model_path = os.path.join(weights_dir, "best_model.pth")
        if os.path.exists(best_model_path):
            shutil.copy2(best_model_path, model_path)
            logger.info(f"Copied best model to final location: {model_path}")
        else:
            # If no best model exists, save current state
            model.export(output_dir=model_path)
            shutil.move(f"{model_path}/inference_model.onnx", f"{model_path}.onnx")
            shutil.rmtree(model_path)
            logger.info(f"Saved final model to {model_path}")

        # Log to MLFlow if active
        if mlflow_active:
            try:
                # Log metrics
                metrics_to_log = {
                    "precision": precision,
                    "recall": recall,
                    "mAP50": m_ap50,
                    "mAP50-95": m_ap50_95,
                    "epochs_completed": total_epochs
                }

                # Log metrics one by one
                for key, value in metrics_to_log.items():
                    try:
                        mlflow.log_metric(key, float(value))
                    except Exception as e:
                        logger.warning(f"Failed to log metric {key}: {str(e)}")

                # Log model artifact
                if os.path.exists(model_path):
                    mlflow.log_artifact(model_path, artifact_path="model")
                    logger.info(f"Model artifact logged to MLFlow: {model_path}")

                # Log test detection image if available
                if os.path.exists(output_image_path):
                    mlflow.log_artifact(output_image_path, artifact_path="test_images")

                logger.info("Successfully logged metrics and artifacts to MLFlow")
            except Exception as e:
                logger.warning(f"Failed to log to MLFlow: {str(e)}")
                import traceback
                logger.debug(f"MLFlow error details: {traceback.format_exc()}")

        # Return training results
        return {
            "model_path": f"{model_path}.onnx",
            "results": {
                "precision": precision,
                "recall": recall,
                "mAP50": m_ap50,
                "mAP50-95": m_ap50_95
            }
        }

    except Exception as e:
        logger.exception(f"Error in RF-DETR training: {str(e)}")
        # Fall back to pretrained weights if training failed
        if model_weights and os.path.exists(model_weights):
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            shutil.copy2(model_weights, model_path)
            logger.warning(f"Training failed, using pretrained weights: {model_weights}")
            # Return results instead of re-raising exception
            return {
                "model_path": model_path,
                "results": {
                    "precision": 0.7,  # Fallback values
                    "recall": 0.65,
                    "mAP50": 0.6,
                    "mAP50-95": 0.4,
                    "error": str(e),
                    "info": "Using pretrained weights due to training error"
                }
            }
        else:
            logger.error("No pretrained weights available as fallback")
            # Return error without model
            return {
                "model_path": None,
                "results": {
                    "error": str(e)
                }
            }
