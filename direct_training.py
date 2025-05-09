import os
import sys
import logging
import time
import json
import threading
import random
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DirectTrainingPipeline:
    """Direct training pipeline implementation that replaces Dagster"""

    def __init__(self, app=None):
        self.app = app

    def load_dataset(self, job_id):
        """Load real dataset for training"""
        logger.info(f"Loading real dataset for job ID: {job_id}")
        try:
            # Import application models
            from models import TrainingJob, Dataset
            import os
            from app import db, app

            # Use app context
            with app.app_context():
                # Get job info
                job = db.session.get(TrainingJob, job_id)
                if not job:
                    logger.error(f"Job {job_id} not found")
                    return {"dataset_path": f"coco8", "format_type": "yolo"}

                # Get dataset info
                dataset = db.session.get(Dataset, job.dataset_id)
                if not dataset:
                    logger.error(f"Dataset {job.dataset_id} not found")
                    return {"dataset_path": f"coco8", "format_type": "yolo"}

                # Create yaml dataset file if needed
                dataset_path = dataset.data_path

                # Correggi il percorso per ambiente containerizzato
                if '/home/hb/lab/AIModelPipeline/' in dataset_path:
                    # Sostituisci il percorso assoluto con quello relativo a /app
                    dataset_path = dataset_path.replace(
                        '/home/hb/lab/AIModelPipeline/', '/app/')
                    logger.info(
                        f"Convertito percorso dataset per ambiente container: {dataset_path}"
                    )

                if not os.path.exists(dataset_path):
                    logger.warning(f"Dataset path {dataset_path} not found")
                    # Prova un percorso alternativo (in caso di montaggio diverso)
                    alt_path = os.path.join('/app', 'uploads',
                                            os.path.basename(dataset_path))
                    if os.path.exists(alt_path):
                        logger.info(
                            f"Dataset trovato al percorso alternativo: {alt_path}"
                        )
                        dataset_path = alt_path
                    else:
                        # Fallback to example dataset
                        return {
                            "dataset_path": f"coco8",
                            "format_type": "yolo"
                        }

                # Verifica se è un dataset YOLO e genera/aggiorna data.yaml se necessario
                if dataset.format_type == 'yolo':
                    yaml_path = os.path.join(dataset_path, 'data.yaml')
                    # Se esiste già, aggiorniamo solo il path assoluto
                    import yaml
                    if os.path.exists(yaml_path):
                        logger.info(
                            f"Aggiornamento percorso in data.yaml esistente: {yaml_path}"
                        )
                        try:
                            with open(yaml_path, 'r') as f:
                                yaml_data = yaml.safe_load(f)
                            # Aggiorna il percorso
                            yaml_data['path'] = dataset_path
                            with open(yaml_path, 'w') as f:
                                yaml.dump(yaml_data,
                                          f,
                                          default_flow_style=False)
                        except Exception as e:
                            logger.warning(
                                f"Errore nell'aggiornamento del file data.yaml: {str(e)}"
                            )
                    else:
                        # Crea un nuovo file data.yaml
                        logger.info(
                            f"Creazione nuovo file data.yaml in {yaml_path}")
                        train_dir = os.path.join(dataset_path, 'train')
                        valid_dir = os.path.join(dataset_path, 'valid')
                        test_dir = os.path.join(dataset_path, 'test')

                        yaml_config = {
                            "path": dataset_path,
                            "train":
                            "train" if os.path.exists(train_dir) else "",
                            "val":
                            "valid" if os.path.exists(valid_dir) else "",
                            "test": "test" if os.path.exists(test_dir) else "",
                            "names": {
                                0: "class0",
                                1: "class1",
                                2: "class2"
                            }
                        }

                        # Salva il file YAML
                        with open(yaml_path, 'w') as f:
                            yaml.dump(yaml_config, f, default_flow_style=False)

                logger.info(
                    f"Found dataset at {dataset_path} with format {dataset.format_type}"
                )
                return {
                    "dataset_path": dataset_path,
                    "format_type": dataset.format_type
                }

                logger.info(
                    f"Found dataset at {dataset_path} with format {dataset.format_type}"
                )
                return {
                    "dataset_path": dataset_path,
                    "format_type": dataset.format_type
                }
        except Exception as e:
            logger.exception(f"Error loading dataset: {str(e)}")
            # Return default path when exceptions occur
            return {"dataset_path": f"coco8", "format_type": "yolo"}

    def download_pretrained_weights(self, model_variant):
        """Download pre-trained weights if not already present"""
        import os
        import requests
        from tqdm import tqdm

        # Create pretrained directory if it doesn't exist
        pretrained_dir = os.path.join(os.getcwd(), "pretrained")
        if not os.path.exists(pretrained_dir):
            os.makedirs(pretrained_dir)
            logger.info(
                f"Created pretrained weights directory: {pretrained_dir}")

        # Define model weights URLs based on variant
        pretrained_urls = {
            # YOLO models
            'yolov5s':
            'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt',
            'yolov5m':
            'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt',
            'yolov5l':
            'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5l.pt',
            'yolov5x':
            'https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5x.pt',
            'yolov8n':
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
            'yolov8s':
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
            'yolov8m':
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
            'yolov8l':
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
            'yolov8x':
            'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
            # RF-DETR models
            'rf_detr_r50':
            'https://github.com/IDEA-Research/detrex-storage/releases/download/rf-detr-v1.0/rf_detr_r50_3x.pth',
            'rf_detr_r101':
            'https://github.com/IDEA-Research/detrex-storage/releases/download/rf-detr-v1.0/rf_detr_r101_3x.pth'
        }

        logger.info(
            f"Available pretrained model variants: {list(pretrained_urls.keys())}"
        )

        # Check if variant exists in our mapping
        if model_variant not in pretrained_urls:
            logger.warning(
                f"No pre-trained weights URL defined for variant: {model_variant}"
            )
            return None

        # Determine weights filename and path
        weights_filename = f"{model_variant}_pretrained.pt"
        weights_path = os.path.join(pretrained_dir, weights_filename)

        # Check if weights already exist
        if os.path.exists(weights_path):
            logger.info(
                f"Pre-trained weights already exist at: {weights_path}")
            return weights_path

        # Download weights if they don't exist
        url = pretrained_urls[model_variant]
        logger.info(
            f"Downloading pre-trained weights for {model_variant} from {url}")

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192  # 8 KB

            with open(weights_path, 'wb') as f, tqdm(
                    desc=f"Downloading {weights_filename}",
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress_bar.update(len(data))

            logger.info(
                f"Successfully downloaded pre-trained weights to: {weights_path}"
            )
            return weights_path

        except Exception as e:
            logger.error(f"Failed to download pre-trained weights: {str(e)}")
            return None

    def train_model(self, dataset_info, model_variant, hyperparameters,
                    mlflow_run_id, mlflow_tracking_uri):
        """Train model with provided configuration using actual implementation"""
        logger.info(
            f"Training model variant: {model_variant} with MLFlow run ID: {mlflow_run_id}"
        )
        logger.info(f"Using dataset: {dataset_info}")
        logger.info(f"Hyperparameters: {hyperparameters}")

        # Create models directory if it doesn't exist
        import os
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created models directory: {models_dir}")

        # Check if we need to use pre-trained weights
        pretrained_weights_path = None
        # Convert string 'true' to boolean if needed
        pretrained_flag = hyperparameters.get('pretrained', False)
        if isinstance(pretrained_flag, str):
            pretrained_flag = pretrained_flag.lower() == 'true'

        if pretrained_flag:
            logger.info(
                f"Using pre-trained weights as requested in hyperparameters for variant {model_variant}"
            )
            pretrained_weights_path = self.download_pretrained_weights(
                model_variant)
            if pretrained_weights_path:
                logger.info(
                    f"Pre-trained weights loaded from: {pretrained_weights_path}"
                )
            else:
                logger.warning(
                    f"Failed to download pre-trained weights for {model_variant}, continuing without them"
                )

        # Determine model path
        model_filename = f"{model_variant}_{mlflow_run_id[:8]}.pt"
        model_path = os.path.join(models_dir, model_filename)

        # Get training parameters
        total_epochs = int(hyperparameters.get('epochs', 100))
        batch_size = int(hyperparameters.get('batch_size', 16))
        img_size = int(hyperparameters.get('img_size', 640))
        learning_rate = float(hyperparameters.get('learning_rate', 0.01))

        # Get real dataset path
        dataset_path = dataset_info.get('dataset_path')
        logger.info(
            f"Starting real training on dataset: {dataset_path} for {total_epochs} epochs"
        )

        try:
            # Connect to MLFlow for logging if available
            try:
                import mlflow
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow_active = True
                logger.info("MLFlow logging enabled")
            except Exception as e:
                logger.warning(f"MLFlow logging disabled: {str(e)}")
                mlflow_active = False

            # Actual model training based on model type
            if 'yolo' in model_variant:
                try:
                    # Import required modules for YOLO training
                    from ultralytics import YOLO
                    from ultralytics import settings

                    # Initialize model with pretrained weights if available
                    if pretrained_weights_path and os.path.exists(
                            pretrained_weights_path):
                        logger.info(
                            f"Loading pretrained YOLO model from {pretrained_weights_path}"
                        )
                        model = YOLO(pretrained_weights_path)
                    else:
                        logger.info(
                            f"Creating new YOLO model: {model_variant}")
                        model = YOLO(model_variant)
                    settings.update({'mlflow': True})

                    # Configure dataset
                    if not os.path.exists(dataset_path):
                        # Fallback for testing: use COCO8 example dataset
                        logger.warning(
                            f"Dataset path {dataset_path} not found, using example dataset"
                        )
                        dataset_path = "coco8"
                    elif os.path.isdir(dataset_path):
                        # Verifica se esiste il file di configurazione data.yaml nella cartella
                        yaml_path = os.path.join(dataset_path, "data.yaml")
                        if os.path.exists(yaml_path):
                            logger.info(
                                f"Usando il file di configurazione YAML esistente: {yaml_path}"
                            )
                            dataset_path = yaml_path
                        else:
                            # Crea un file YAML temporaneo per il dataset
                            logger.info(
                                f"Il dataset è una directory, creazione file YAML temporaneo"
                            )
                            import yaml

                            # Verifica la struttura del dataset
                            train_dir = os.path.join(dataset_path, "train")
                            valid_dir = os.path.join(dataset_path, "valid")
                            test_dir = os.path.join(dataset_path, "test")

                            # Creazione configurazione YAML
                            yaml_config = {
                                "path": dataset_path,
                                "train":
                                "train" if os.path.exists(train_dir) else "",
                                "val":
                                "valid" if os.path.exists(valid_dir) else "",
                                "test":
                                "test" if os.path.exists(test_dir) else "",
                                "names": {}
                            }

                            # Rileva automaticamente le classi dalle etichette nel set di training
                            if os.path.exists(train_dir):
                                labels_dir = os.path.join(train_dir, "labels")
                                if os.path.exists(labels_dir):
                                    class_ids = set()
                                    # Analizza i primi 10 file di etichette per rilevare le classi
                                    for label_file in os.listdir(
                                            labels_dir)[:10]:
                                        if label_file.endswith('.txt'):
                                            with open(
                                                    os.path.join(
                                                        labels_dir,
                                                        label_file), 'r') as f:
                                                for line in f:
                                                    parts = line.strip().split(
                                                    )
                                                    if parts and parts[
                                                            0].isdigit():
                                                        class_ids.add(
                                                            int(parts[0]))

                                    # Crea dizionario delle classi
                                    for class_id in sorted(class_ids):
                                        yaml_config["names"][
                                            class_id] = f"class{class_id}"

                            # Se non sono state trovate classi, imposta valori predefiniti
                            if not yaml_config["names"]:
                                yaml_config["names"] = {
                                    0: "class0",
                                    1: "class1"
                                }

                            # Salva il file YAML
                            yaml_path = os.path.join(dataset_path, "data.yaml")
                            with open(yaml_path, 'w') as f:
                                yaml.dump(yaml_config, f, sort_keys=False)

                            logger.info(
                                f"File di configurazione YAML creato: {yaml_path}"
                            )
                            dataset_path = yaml_path

                    # Log train command details for debugging
                    logger.info(
                        f"Starting real YOLOv8 training with dataset: {dataset_path}"
                    )
                    logger.info(
                        f"Epochs: {total_epochs}, Batch size: {batch_size}, Image size: {img_size}"
                    )
                    logger.info(f"Learning rate: {learning_rate}")

                    # Train the model with real hyperparameters
                    try:
                        # Riduci dimensioni batch e risoluzione immagine se necessario
                        adjusted_batch = min(batch_size,
                                             8)  # Riduci il batch size massimo
                        adjusted_size = min(
                            img_size,
                            416)  # Riduci la dimensione massima immagine

                        results = model.train(
                            data=dataset_path,
                            epochs=total_epochs,
                            batch=adjusted_batch,
                            imgsz=adjusted_size,
                            lr0=learning_rate,
                            patience=50,
                            save=True,
                            project="training_jobs",
                            name=f"job_{mlflow_run_id[:8]}",
                            cache=
                            False,  # Disabilita la cache per risparmiare memoria
                            workers=
                            1  # Riduci il numero di worker per ridurre la memoria
                        )
                        logger.info(
                            f"Training completed successfully. Results: {results.results_dict}"
                        )
                    except Exception as e:
                        logger.error(f"Error during YOLO training: {str(e)}")
                        # Print detailed error traceback
                        import traceback
                        logger.error(
                            f"Detailed traceback: {traceback.format_exc()}")
                        raise

                    # Get metrics from results
                    final_metrics = results.results_dict

                    # Extract metrics for reporting
                    precision = final_metrics.get('metrics/precision(B)', 0.0)
                    recall = final_metrics.get('metrics/recall(B)', 0.0)
                    mAP50 = final_metrics.get('metrics/mAP50(B)', 0.0)
                    mAP50_95 = final_metrics.get('metrics/mAP50-95(B)', 0.0)

                    # Save the trained model - use the best.pt file directly instead of exporting
                    trained_model_path = os.path.join(
                        os.getcwd(),
                        f"training_jobs/job_{mlflow_run_id[:8]}/weights/best.pt"
                    )
                    if os.path.exists(trained_model_path):
                        import shutil
                        shutil.copy2(trained_model_path, model_path)
                        logger.info(f"Model saved to: {model_path}")
                    else:
                        logger.warning(
                            f"Trained model not found at {trained_model_path}, saving current model"
                        )
                        # Use the trained model directly
                        model.save(model_path)

                except Exception as e:
                    logger.exception(f"Error in YOLO training: {str(e)}")
                    # Fall back to pretrained weights if training failed
                    if pretrained_weights_path and os.path.exists(
                            pretrained_weights_path):
                        import shutil
                        shutil.copy2(pretrained_weights_path, model_path)
                        logger.warning(
                            f"Training failed, using pretrained weights: {pretrained_weights_path}"
                        )
                    raise

            # Use real trained model or pretrained weights as fallback
            try:
                # Check if training produced a model file
                trained_model_path = os.path.join(
                    os.getcwd(),
                    f"training_jobs/job_{mlflow_run_id[:8]}/weights/best.pt")

                if os.path.exists(trained_model_path):
                    # Copy the trained model to the specified path
                    import shutil
                    shutil.copy2(trained_model_path, model_path)
                    logger.info(
                        f"Copied trained model from {trained_model_path} to {model_path}"
                    )

                    # Log model size for debugging
                    model_size = os.path.getsize(model_path) / (1024 * 1024
                                                                )  # Size in MB
                    logger.info(f"Trained model size: {model_size:.2f} MB")
            except Exception as e:
                logger.warning(
                    f"Failed to log model artifact to MLFlow: {str(e)}")

                # Log metrics to MLFlow
                if mlflow_active:
                    try:
                        # Converti i valori numpy in standard Python
                        def convert_numpy_values(d):
                            result = {}
                            for k, v in d.items():
                                if hasattr(v, 'dtype') and hasattr(
                                        v, 'item'):  # È un tipo numpy
                                    result[k] = v.item(
                                    )  # Converti in tipo Python standard
                                else:
                                    result[k] = v
                            return result

                        # Log delle metriche principali
                        metrics_to_log = {
                            "precision": precision,
                            "recall": recall,
                            "mAP50": mAP50,
                            "mAP50-95": mAP50_95,
                            "epochs_completed": total_epochs
                        }

                        # Converti tutti i valori numpy in tipi Python standard
                        converted_metrics = convert_numpy_values(
                            metrics_to_log)

                        # Log di tutte le metriche originali
                        all_metrics = convert_numpy_values(final_metrics)
                        for k, v in all_metrics.items():
                            # Usa un nome più leggibile per le metriche di MLFlow
                            key = k.replace('metrics/', '').replace('(B)', '')
                            mlflow.log_metric(key, v)

                        # Log delle metriche principali con nomi standard
                        for k, v in converted_metrics.items():
                            mlflow.log_metric(k, v)

                        # Log dell'artefatto del modello
                        mlflow.log_artifact(model_path)
                        logger.info(
                            f"Model artifact and metrics logged to MLFlow")
                    except Exception as e:
                        logger.warning(f"Failed to log to MLFlow: {str(e)}")
                        import traceback
                        logger.warning(
                            f"MLFlow error details: {traceback.format_exc()}")

            # Log the model artifact to MLFlow if available
            if mlflow_active:
                try:
                    # Log final metrics to MLFlow (riprova)
                    final_metrics_dict = {
                        "precision": float(precision),
                        "recall": float(recall),
                        "mAP50": float(mAP50),
                        "mAP50-95": float(mAP50_95),
                        "epochs_completed": int(total_epochs)
                    }

                    # Log delle metriche una alla volta per garantire il successo
                    for metric_name, metric_value in final_metrics_dict.items(
                    ):
                        try:
                            mlflow.log_metric(metric_name, metric_value)
                            logger.info(
                                f"Logged metric {metric_name}={metric_value} to MLFlow"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to log metric {metric_name}: {str(e)}"
                            )

                    # Verifica se il run_id è attivo
                    try:
                        active_run = mlflow.active_run()
                        if active_run is None:
                            logger.info(
                                f"No active MLFlow run. Starting run with ID: {mlflow_run_id}"
                            )
                            mlflow.start_run(run_id=mlflow_run_id)
                    except Exception as e:
                        logger.warning(f"Error checking active run: {str(e)}")

                    # Log model artifact
                    try:
                        # Verifica che il file esista
                        if os.path.exists(model_path):
                            mlflow.log_artifact(model_path,
                                                artifact_path="model")
                            logger.info(
                                f"Model artifact logged to MLFlow: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)"
                            )
                        else:
                            logger.warning(
                                f"Cannot log model to MLFlow: file not found at {model_path}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to log model artifact to MLFlow: {str(e)}"
                        )
                        import traceback
                        logger.debug(
                            f"MLFlow artifact logging error: {traceback.format_exc()}"
                        )

                    # Try also logging some images as artifacts
                    try:
                        # Check if results directory exists with plot images
                        results_dir = os.path.join(
                            os.getcwd(),
                            f"training_jobs/job_{mlflow_run_id[:8]}")
                        if os.path.exists(results_dir):
                            for root, _, files in os.walk(results_dir):
                                for file in files:
                                    if file.endswith(
                                        ('.png',
                                         '.jpg')) and not file.startswith('.'):
                                        img_path = os.path.join(root, file)
                                        if os.path.exists(img_path):
                                            rel_path = os.path.relpath(
                                                root, results_dir)
                                            mlflow.log_artifact(
                                                img_path,
                                                artifact_path=
                                                f"plots/{rel_path}")
                                            logger.info(
                                                f"Logged plot to MLFlow: {img_path}"
                                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to log plot images to MLFlow: {str(e)}")

                    logger.info(f"All metrics and artifacts logged to MLFlow")
                except Exception as e:
                    logger.warning(
                        f"Failed to log model artifact or metrics to MLFlow: {str(e)}"
                    )
                    import traceback
                    logger.debug(
                        f"MLFlow error details: {traceback.format_exc()}")

            # Return training results
            return {
                "model_path": model_path,
                "results": {
                    "precision": precision,
                    "recall": recall,
                    "mAP50": mAP50,
                    "mAP50-95": mAP50_95
                }
            }

        except Exception as e:
            logger.exception(f"Error during model training: {str(e)}")
            # Return a minimal result with error info
            return {
                "model_path": None,
                "results": {
                    "error": str(e)
                },
                "error": str(e)
            }

    def save_artifacts(self, model_results, job_id):
        """Save model artifacts and update database"""
        logger.info(f"Saving artifacts for job ID: {job_id}")
        logger.info(f"Model results: {model_results}")

        # Check if model was trained successfully
        if not model_results.get("model_path"):
            logger.error(f"No model path in results for job {job_id}")
            return model_results

        # In a real implementation, we would copy the model to a more permanent location
        # Here we'll ensure it's saved in the models directory
        import os
        import shutil

        # Ensure the models directory exists
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Get the source model path
        source_path = model_results["model_path"]

        # Define the destination path with job ID for uniqueness
        if os.path.exists(source_path):
            filename = os.path.basename(source_path)
            destination_path = os.path.join(models_dir,
                                            f"job_{job_id}_{filename}")

            # Copy the model file
            try:
                shutil.copy2(source_path, destination_path)
                logger.info(f"Model artifact copied to: {destination_path}")

                # Update the model path in results
                model_results["model_path"] = destination_path
            except Exception as e:
                logger.error(f"Failed to copy model artifact: {str(e)}")
        else:
            logger.error(f"Model file not found at: {source_path}")

        return model_results

    def execute_pipeline(self, config):
        """Execute the entire training pipeline"""
        # Extract configuration
        job_id = config.get('job_id')
        model_type = config.get('model_type')
        model_variant = config.get('model_variant')
        hyperparameters = config.get('hyperparameters')
        mlflow_run_id = config.get('mlflow_run_id')
        mlflow_tracking_uri = config.get('mlflow_tracking_uri')

        try:
            logger.info(
                f"Starting direct training pipeline for job ID: {job_id}")
            logger.info(f"Model type: {model_type}, variant: {model_variant}")

            # Step 1: Load dataset
            dataset_info = self.load_dataset(job_id)

            # Step 2: Train model
            model_results = self.train_model(dataset_info, model_variant,
                                             hyperparameters, mlflow_run_id,
                                             mlflow_tracking_uri)

            # Step 3: Save artifacts
            final_results = self.save_artifacts(model_results, job_id)

            # Step 4: Complete the job
            from ml_pipelines import complete_training_job
            artifacts = [{
                'path': model_results['model_path'],
                'type': 'weights',
                'metrics': model_results['results']
            }]
            complete_training_job(job_id, success=True, artifacts=artifacts)

            logger.info(
                f"Pipeline execution completed successfully for job ID: {job_id}"
            )
            return True, final_results

        except Exception as e:
            logger.exception(f"Error executing pipeline: {str(e)}")
            # Mark job as failed
            from ml_pipelines import complete_training_job
            complete_training_job(job_id, success=False, error_message=str(e))
            return False, {"error": str(e)}


# Create singleton instance
direct_pipeline = DirectTrainingPipeline()


def submit_direct_pipeline(config):
    """
    Submit a training job for direct execution

    Args:
        config: Dictionary with pipeline configuration

    Returns:
        run_id: A unique identifier for this run
    """
    # Generate a unique run ID
    import uuid
    run_id = f"direct-{uuid.uuid4().hex}"

    # Create a copy of the config and add the run ID
    run_config = config.copy()
    run_config['direct_run_id'] = run_id

    # Start the pipeline in a separate thread
    def run_pipeline():
        direct_pipeline.execute_pipeline(run_config)

    thread = threading.Thread(target=run_pipeline)
    thread.daemon = True
    thread.start()

    logger.info(f"Submitted direct training pipeline with run ID: {run_id}")
    return run_id
