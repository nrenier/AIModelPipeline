
import os
import logging
import json
import mlflow
from app import app

logger = logging.getLogger(__name__)

def sync_mlflow_artifacts(job_id):
    """
    Sincronizza gli artefatti mancanti su MLFlow per un job specifico.
    Utile per recuperare metriche e model artifacts che non sono stati caricati correttamente.
    """
    from models import TrainingJob, ModelArtifact
    from app import db
    
    with app.app_context():
        job = db.session.get(TrainingJob, job_id)
        if not job:
            logger.error(f"Job con ID {job_id} non trovato")
            return False
            
        if not job.mlflow_run_id or job.mlflow_run_id.startswith('direct-'):
            logger.warning(f"Job {job_id} non ha un valido MLFlow run ID: {job.mlflow_run_id}")
            return False
            
        # Prova a connettersi a MLFlow
        try:
            mlflow.set_tracking_uri(app.config.get("MLFLOW_TRACKING_URI", "http://localhost:5001"))
            
            # Verifica se la run esiste
            run = mlflow.get_run(job.mlflow_run_id)
            if not run:
                logger.error(f"Run MLFlow {job.mlflow_run_id} non trovata")
                return False
                
            # Controlla se ci sono metriche mancanti
            artifacts = ModelArtifact.query.filter_by(training_job_id=job_id).all()
            for artifact in artifacts:
                if artifact.artifact_type == 'weights' and artifact.metrics:
                    # Ottieni metriche dall'artefatto
                    metrics = artifact.get_metrics()
                    if metrics:
                        logger.info(f"Sincronizzazione metriche per job {job_id}: {metrics}")
                        
                        # Verifica quali metriche sono mancanti e sincronizzale
                        with mlflow.start_run(run_id=job.mlflow_run_id):
                            for metric_name, metric_value in metrics.items():
                                # Converti in tipo primitivo se necessario
                                if hasattr(metric_value, 'dtype') and hasattr(metric_value, 'item'):
                                    metric_value = metric_value.item()
                                try:
                                    mlflow.log_metric(metric_name, float(metric_value))
                                    logger.info(f"Metrica sincronizzata: {metric_name}={metric_value}")
                                except Exception as e:
                                    logger.warning(f"Errore sincronizzazione metrica {metric_name}: {str(e)}")
                    
                    # Sincronizza l'artefatto del modello
                    if artifact.artifact_path and os.path.exists(artifact.artifact_path):
                        logger.info(f"Sincronizzazione artefatto modello: {artifact.artifact_path}")
                        try:
                            with mlflow.start_run(run_id=job.mlflow_run_id):
                                mlflow.log_artifact(artifact.artifact_path, "model")
                                logger.info(f"Artefatto modello sincronizzato: {artifact.artifact_path}")
                        except Exception as e:
                            logger.warning(f"Errore sincronizzazione artefatto: {str(e)}")
            
            return True
                
        except Exception as e:
            logger.error(f"Errore sincronizzazione con MLFlow: {str(e)}")
            import traceback
            logger.error(f"Dettaglio errore: {traceback.format_exc()}")
            return False

# The code ensures that the format_type parameter is validated to prevent unexpected behavior.
import os
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

logger = logging.getLogger(__name__)

def validate_dataset(dataset_path, format_type):
    """
    Validate a dataset directory structure and return statistics

    Args:
        dataset_path: Path to the dataset directory
        format_type: Type of dataset format (coco, yolo, voc)

    Returns:
        Dictionary with validation results
    """
    result = {
        'valid': False,
        'error': None,
        'image_count': 0,
        'classes': [],
        'class_distribution': {}
    }

    try:
        # Check if the dataset directory exists
        if not os.path.isdir(dataset_path):
            result['error'] = "Dataset directory not found"
            return result

        logger.info(f"Validating dataset in {dataset_path} with format {format_type}")

        if format_type == 'coco':
            # Check for subdirectories (train, test, valid) that might contain annotations
            image_count = 0
            classes_set = set()

            # Look for subdirectories like train, test, valid
            subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            logger.info(f"Found subdirectories: {subdirs}")

            for subdir in subdirs:
                subdir_path = os.path.join(dataset_path, subdir)
                # Check for annotation files in this subdir
                annotation_files = [f for f in os.listdir(subdir_path) 
                                  if f.endswith('.json') and 'annotation' in f.lower()]

                if annotation_files:
                    logger.info(f"Found annotation file in {subdir}: {annotation_files[0]}")
                    anno_path = os.path.join(subdir_path, annotation_files[0])
                    with open(anno_path, 'r') as f:
                        try:
                            coco_data = json.load(f)

                            # Check required COCO keys
                            required_keys = ['images', 'annotations', 'categories']
                            if not all(key in coco_data for key in required_keys):
                                logger.warning(f"Missing required keys in {anno_path}")
                                continue

                            # Count images and get classes
                            image_count += len(coco_data['images'])
                            for cat in coco_data['categories']:
                                classes_set.add(cat['name'])

                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in {anno_path}")
                            continue

            # If we found any images or classes
            if image_count > 0 and classes_set:
                result['valid'] = True
                result['image_count'] = image_count
                result['classes'] = list(classes_set)
                logger.info(f"Validated COCO dataset with {image_count} images and {len(classes_set)} classes")
                return result

            # Try the old way if the above didn't work
            annotation_files = [f for f in os.listdir(dataset_path) 
                              if f.endswith('.json') and os.path.isfile(os.path.join(dataset_path, f))]

            if not annotation_files:
                result['error'] = "No JSON annotation files found for COCO format"
                return result

            # Check the first annotation file found
            anno_path = os.path.join(dataset_path, annotation_files[0])
            with open(anno_path, 'r') as f:
                try:
                    coco_data = json.load(f)

                    # Check required COCO keys
                    required_keys = ['images', 'annotations', 'categories']
                    if not all(key in coco_data for key in required_keys):
                        result['error'] = "Invalid COCO JSON format: missing required keys"
                        return result

                    # Count images and get classes
                    result['image_count'] = len(coco_data['images'])
                    result['classes'] = [cat['name'] for cat in coco_data['categories']]

                    # Get class distribution
                    category_counts = Counter([anno['category_id'] for anno in coco_data['annotations']])
                    for cat in coco_data['categories']:
                        cat_id = cat['id']
                        result['class_distribution'][cat['name']] = category_counts.get(cat_id, 0)

                except json.JSONDecodeError:
                    result['error'] = "Invalid JSON file in COCO dataset"
                    return result

        elif format_type == 'yolo':
            # Validate YOLO format - supporta sia il formato tradizionale (images/labels) che quello YOLOv8 (train/valid/test)
            images_dir = None
            labels_dir = None

            # Prima, controlla se abbiamo la struttura YOLOv8 (train/valid/test con immagini e label in ciascuna)
            train_dir = os.path.join(dataset_path, 'train')
            valid_dir = os.path.join(dataset_path, 'valid')
            test_dir = os.path.join(dataset_path, 'test')

            if os.path.exists(train_dir) and os.path.exists(valid_dir):
                logger.info(f"Rilevata struttura YOLOv8 con cartelle train/valid in {dataset_path}")

                # Creiamo il file data.yaml se non esiste
                yaml_path = os.path.join(dataset_path, 'data.yaml')
                if not os.path.exists(yaml_path):
                    logger.info(f"Creazione file data.yaml per dataset YOLOv8 in {yaml_path}")
                    import yaml

                    # Rileva le classi dalle etichette
                    class_ids = set()
                    if os.path.exists(os.path.join(train_dir, 'labels')):
                        train_labels = os.path.join(train_dir, 'labels')
                        for f in os.listdir(train_labels)[:100]:
                            if f.endswith('.txt'):
                                with open(os.path.join(train_labels, f), 'r') as file:
                                    for line in file:
                                        parts = line.strip().split()
                                        if parts and parts[0].isdigit():
                                            class_ids.add(int(parts[0]))

                    # Crea dizionario delle classi
                    names = {}
                    for class_id in sorted(class_ids):
                        names[class_id] = f"class{class_id}"

                    # Se non abbiamo trovato classi, usiamo default
                    if not names:
                        names = {0: "class0", 1: "class1", 2: "class2"}

                    # Crea configurazione YAML
                    yaml_config = {
                        "path": dataset_path,
                        "train": "train",
                        "val": "valid",
                        "test": "test" if os.path.exists(test_dir) else "",
                        "names": names
                    }

                    # Salva il file YAML
                    with open(yaml_path, 'w') as f:
                        yaml.dump(yaml_config, f, default_flow_style=False)

                # Usa train come directory di immagini per il conteggio
                if os.path.exists(os.path.join(train_dir, 'images')):
                    images_dir = os.path.join(train_dir, 'images')
                    labels_dir = os.path.join(train_dir, 'labels')
                else:
                    # Se non ci sono sottocartelle images/labels, usa direttamente train
                    images_dir = train_dir
                    # Cerca le etichette
                    for subdir in os.listdir(train_dir):
                        if subdir.lower() in ['labels', 'txt']:
                            labels_dir = os.path.join(train_dir, subdir)
                            break
            else:
                # Struttura tradizionale: cerca cartelle images e labels separate
                for root, dirs, files in os.walk(dataset_path):
                    if os.path.basename(root).lower() in ['images', 'img', 'jpegs']:
                        images_dir = root
                    elif os.path.basename(root).lower() in ['labels', 'txt']:
                        labels_dir = root

            if not images_dir:
                result['error'] = "Could not find images directory in YOLO dataset"
                return result

            if not labels_dir:
                result['error'] = "Could not find labels directory in YOLO dataset"
                return result

            # Count image files
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            result['image_count'] = len(image_files)

            if result['image_count'] == 0:
                result['error'] = "No image files found in images directory"
                return result

            # Look for classes.txt or data.yaml file
            classes_file = None
            for f in os.listdir(dataset_path):
                if f == 'classes.txt' or f == 'data.yaml':
                    classes_file = os.path.join(dataset_path, f)
                    break

            if classes_file:
                if classes_file.endswith('.txt'):
                    with open(classes_file, 'r') as f:
                        result['classes'] = [line.strip() for line in f if line.strip()]
                elif classes_file.endswith('.yaml'):
                    import yaml
                    with open(classes_file, 'r') as f:
                        try:
                            yaml_data = yaml.safe_load(f)
                            if 'names' in yaml_data and isinstance(yaml_data['names'], list):
                                result['classes'] = yaml_data['names']
                            elif 'names' in yaml_data and isinstance(yaml_data['names'], dict):
                                # Handle numeric keys
                                result['classes'] = [yaml_data['names'][i] for i in sorted(yaml_data['names'].keys())]
                        except yaml.YAMLError:
                            pass

            # If no class file found, try to infer classes from label files
            if not result['classes']:
                class_ids = set()
                # Limita il numero di file da analizzare per evitare rallentamenti
                label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')][:100]
                for f in label_files:
                    if f.endswith('.txt'):
                        try:
                            with open(os.path.join(labels_dir, f), 'r') as label_file:
                                for line in label_file:
                                    parts = line.strip().split()
                                    if parts and parts[0].isdigit():
                                        class_ids.add(int(parts[0]))
                        except Exception as e:
                            logger.warning(f"Errore durante la lettura del file di etichette {f}: {str(e)}")
                            continue

                # Create placeholder class names
                if class_ids:
                    result['classes'] = [f"class{i}" for i in range(max(class_ids) + 1) if i in class_ids]
                else:
                    # Fornisci classi di default se non è stato possibile rilevarne alcuna
                    result['classes'] = ["class0", "class1"]
                    logger.warning("Non sono state rilevate classi nei file di etichette, utilizzando classi predefinite")

            # Get class distribution (sample from first 100 label files)
            class_counts = Counter()
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')][:100]
            for f in label_files:
                with open(os.path.join(labels_dir, f), 'r') as label_file:
                    for line in label_file:
                        parts = line.strip().split()
                        if parts and parts[0].isdigit():
                            class_id = int(parts[0])
                            if class_id < len(result['classes']):
                                class_counts[result['classes'][class_id]] += 1

            result['class_distribution'] = dict(class_counts)

        elif format_type == 'voc':
            # Validate Pascal VOC format
            # Look for Annotations and JPEGImages directories
            anno_dir = None
            images_dir = None

            for root, dirs, files in os.walk(dataset_path):
                if os.path.basename(root).lower() in ['annotations', 'annotation']:
                    anno_dir = root
                elif os.path.basename(root).lower() in ['jpegimages', 'images', 'imgs']:
                    images_dir = root

            if not anno_dir:
                result['error'] = "Could not find Annotations directory in VOC dataset"
                return result

            if not images_dir:
                result['error'] = "Could not find JPEGImages directory in VOC dataset"
                return result

            # Count image files
            image_files = [f for f in os.listdir(images_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            result['image_count'] = len(image_files)

            if result['image_count'] == 0:
                result['error'] = "No image files found in JPEGImages directory"
                return result

            # Parse XML files to get classes
            class_counts = Counter()
            annotation_files = [f for f in os.listdir(anno_dir) if f.endswith('.xml')]

            for i, f in enumerate(annotation_files[:100]):  # Sample from first 100 files
                try:
                    tree = ET.parse(os.path.join(anno_dir, f))
                    root = tree.getroot()
                    for obj in root.findall('.//object'):
                        class_name = obj.find('name').text
                        if class_name:
                            class_counts[class_name] += 1
                except Exception as e:
                    logger.warning(f"Error parsing XML file {f}: {str(e)}")

            result['classes'] = list(class_counts.keys())
            result['class_distribution'] = dict(class_counts)

        else:
            result['error'] = f"Unsupported dataset format: {format_type}"
            return result

        # Final validation
        if result['image_count'] == 0:
            result['error'] = "No valid images found in dataset"
            return result

        if not result['classes']:
            result['error'] = "No classes found in dataset"
            return result

        # All validation passed
        result['valid'] = True
        return result

    except Exception as e:
        logger.exception(f"Error validating dataset: {str(e)}")
        result['error'] = f"Error validating dataset: {str(e)}"
        return result


def get_dataset_stats(dataset):
    """Get statistics for a dataset"""
    stats = {
        'image_count': dataset.image_count,
        'classes': dataset.get_class_names(),
        'format': dataset.format_type,
        'class_distribution': {}
    }

    # This is a placeholder - in a real implementation, 
    # we would compute additional statistics here or retrieve
    # them from storage

    return stats


def convert_dataset_format(dataset, target_format):
    """Convert a dataset from one format to another"""
    # This is a placeholder for dataset conversion functionality
    # In a real implementation, this would convert between COCO, YOLO, VOC formats

    return {
        'success': False,
        'error': 'Dataset conversion not implemented yet'
    }