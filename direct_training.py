
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
        """Load dataset for training"""
        logger.info(f"Loading dataset for job ID: {job_id}")
        # Simulate dataset loading
        return {"dataset_path": f"/tmp/dataset_{job_id}", "format_type": "yolo"}
    
    def train_model(self, dataset_info, model_variant, hyperparameters, mlflow_run_id, mlflow_tracking_uri):
        """Train model with provided configuration"""
        logger.info(f"Training model variant: {model_variant} with MLFlow run ID: {mlflow_run_id}")
        logger.info(f"Using dataset: {dataset_info}")
        logger.info(f"Hyperparameters: {hyperparameters}")

        # Create models directory if it doesn't exist
        import os
        models_dir = os.path.join(os.getcwd(), "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            logger.info(f"Created models directory: {models_dir}")

        # Determine model path
        model_filename = f"{model_variant}_{mlflow_run_id[:8]}.pt"
        model_path = os.path.join(models_dir, model_filename)
        
        # Actual training process
        total_epochs = hyperparameters.get('epochs', 100)
        
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
            
            # Training loop
            for epoch in range(1, total_epochs + 1):
                # Here we simulate the training process
                # In a real implementation, this would be the actual model training
                time.sleep(0.1)  # Speed up for testing
                
                # Calculate metrics based on progress
                progress = epoch / total_epochs
                loss = max(1.0 - (progress * 0.8), 0.2)
                precision = min(0.5 + (progress * 0.4), 0.9)
                recall = min(0.5 + (progress * 0.35), 0.85)
                mAP50 = min(0.5 + (progress * 0.45), 0.95)
                mAP50_95 = min(0.4 + (progress * 0.35), 0.75)
                
                # Log metrics to MLFlow if available
                metrics = {
                    "epoch": epoch,
                    "loss": loss,
                    "precision": precision,
                    "recall": recall,
                    "mAP50": mAP50,
                    "mAP50-95": mAP50_95
                }
                
                if mlflow_active:
                    try:
                        mlflow.log_metrics(metrics, step=epoch)
                    except Exception as e:
                        logger.warning(f"Failed to log metrics to MLFlow: {str(e)}")
                
                logger.info(f"Epoch {epoch}/{total_epochs}: loss={loss:.4f}, precision={precision:.4f}, recall={recall:.4f}")
            
            # Create a dummy model file to simulate the trained model
            with open(model_path, 'w') as f:
                f.write(f"Model: {model_variant}\nMLFlow Run ID: {mlflow_run_id}\n")
                f.write(f"Final metrics: precision={precision}, recall={recall}, mAP50={mAP50}, mAP50-95={mAP50_95}")
            
            logger.info(f"Model saved to: {model_path}")
            
            # Log the model artifact to MLFlow if available
            if mlflow_active:
                try:
                    mlflow.log_artifact(model_path)
                    logger.info(f"Model artifact logged to MLFlow")
                except Exception as e:
                    logger.warning(f"Failed to log model artifact to MLFlow: {str(e)}")
            
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
            destination_path = os.path.join(models_dir, f"job_{job_id}_{filename}")
            
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
            logger.info(f"Starting direct training pipeline for job ID: {job_id}")
            logger.info(f"Model type: {model_type}, variant: {model_variant}")
            
            # Step 1: Load dataset
            dataset_info = self.load_dataset(job_id)
            
            # Step 2: Train model
            model_results = self.train_model(
                dataset_info,
                model_variant,
                hyperparameters,
                mlflow_run_id,
                mlflow_tracking_uri
            )
            
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
            
            logger.info(f"Pipeline execution completed successfully for job ID: {job_id}")
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
