import os
import sys
import logging
import mlflow
import tempfile
import json
from datetime import datetime
from app import app, db
from models import TrainingJob, ModelArtifact
from dagster_pipelines import submit_dagster_pipeline

logger = logging.getLogger(__name__)

def start_training_job(job_id):
    """Start a training job by creating MLFlow run and submitting to Dagster"""
    with app.app_context():
        job = db.session.get(TrainingJob, job_id)
        if not job:
            logger.error(f"Training job with ID {job_id} not found")
            return False

        # Modalità simulazione per ambiente di sviluppo
        SIMULATION_MODE = True
        
        # Verifichiamo se dobbiamo simulare MLFlow e Dagster
        if SIMULATION_MODE:
            logger.info(f"SIMULAZIONE: Avvio del job {job_id} in modalità simulazione (senza MLFlow e Dagster)")
            
            import uuid
            # Simula un run MLFlow
            job.mlflow_run_id = f"simulated-mlflow-{uuid.uuid4().hex}"
            # Simula un run Dagster
            job.dagster_run_id = f"simulated-dagster-{uuid.uuid4().hex}"
            
            # Aggiorna lo stato del job
            job.status = 'running'
            job.started_at = datetime.utcnow()
            db.session.commit()
            
            # Simuliamo il completamento del job dopo un po'
            # In una app reale, questo sarebbe fatto da un worker asincrono
            import threading
            def simulate_completion():
                import time, random
                # Attendi tra 10 e 30 secondi
                time.sleep(random.randint(10, 30))
                with app.app_context():
                    try:
                        job = db.session.get(TrainingJob, job_id)
                        if job and job.status == 'running':
                            # Simula completamento con successo
                            job.status = 'completed'
                            job.completed_at = datetime.utcnow()
                            
                            # Aggiungi alcuni artifact fittizi
                            artifact = ModelArtifact()
                            artifact.training_job_id = job.id
                            artifact.artifact_path = "/models/simulated_model.pt"
                            artifact.artifact_type = "weights"
                            metrics = {
                                "precision": random.uniform(0.7, 0.95),
                                "recall": random.uniform(0.7, 0.95),
                                "mAP50": random.uniform(0.7, 0.95),
                                "mAP50-95": random.uniform(0.6, 0.85)
                            }
                            artifact.set_metrics(metrics)
                            
                            db.session.add(artifact)
                            db.session.commit()
                            logger.info(f"SIMULAZIONE: Job {job_id} completato con successo")
                    except Exception as e:
                        logger.error(f"Errore nella simulazione: {str(e)}")
            
            # Avvia il thread di simulazione
            threading.Thread(target=simulate_completion).start()
            return job.dagster_run_id
                    
        # Se non siamo in modalità simulazione, procedi normalmente
        # Set up MLFlow tracking
        mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])
        experiment_name = f"{job.model_type}-training"
        
        # Create MLFlow experiment if it doesn't exist
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.error(f"Error setting up MLFlow experiment: {str(e)}")
            job.status = 'failed'
            job.error_message = f"MLFlow setup error: {str(e)}"
            db.session.commit()
            return False
        
        # Start MLFlow run
        try:
            with mlflow.start_run(experiment_id=experiment_id, run_name=job.job_name) as run:
                # Log parameters
                hyperparams = job.get_hyperparameters()
                for param_name, param_value in hyperparams.items():
                    mlflow.log_param(param_name, param_value)
                
                mlflow.log_param("model_type", job.model_type)
                mlflow.log_param("model_variant", job.model_variant)
                
                # Store MLFlow run ID
                job.mlflow_run_id = run.info.run_id
                
                # Create pipeline configuration file
                config = {
                    "job_id": job.id,
                    "model_type": job.model_type,
                    "model_variant": job.model_variant,
                    "hyperparameters": hyperparams,
                    "mlflow_run_id": run.info.run_id,
                    "mlflow_tracking_uri": app.config["MLFLOW_TRACKING_URI"]
                }
                
                # Submit pipeline to Dagster
                dagster_run_id = submit_dagster_pipeline(config)
                job.dagster_run_id = dagster_run_id
                
                # Update job status
                job.status = 'running'
                job.started_at = datetime.utcnow()
                db.session.commit()
                
                logger.info(f"Training job {job_id} started with MLFlow run ID {run.info.run_id} and Dagster run ID {dagster_run_id}")
                return True
                
        except Exception as e:
            logger.exception(f"Error starting training job: {str(e)}")
            job.status = 'failed'
            job.error_message = str(e)
            db.session.commit()
            return False


def get_job_status(job):
    """Get the latest status of a training job from MLFlow and Dagster"""
    status_data = {
        'status': job.status,
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'completed_at': job.completed_at.isoformat() if job.completed_at else None,
        'error_message': job.error_message,
        'metrics': {},
        'progress': 0.0,
    }
    
    # If job is completed or failed, return saved status
    if job.status in ['completed', 'failed', 'cancelled']:
        # Get metrics from artifacts if available
        artifacts = ModelArtifact.query.filter_by(training_job_id=job.id).all()
        for artifact in artifacts:
            if artifact.metrics:
                metrics = artifact.get_metrics()
                if metrics:
                    status_data['metrics'] = metrics
                    break
        return status_data
    
    # For running jobs, check MLFlow for latest metrics
    if job.mlflow_run_id:
        try:
            mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])
            run = mlflow.get_run(job.mlflow_run_id)
            
            # Get metrics from MLFlow
            if run and run.data.metrics:
                status_data['metrics'] = run.data.metrics
                
                # Calculate progress if 'epoch' metric is available
                hyperparams = job.get_hyperparameters()
                if 'epoch' in run.data.metrics and 'epochs' in hyperparams:
                    current_epoch = run.data.metrics['epoch']
                    total_epochs = hyperparams['epochs']
                    status_data['progress'] = (current_epoch / total_epochs) * 100
        except Exception as e:
            logger.error(f"Error fetching MLFlow metrics: {str(e)}")
    
    # Check Dagster status (simplified - would normally use Dagster API)
    # For demo purposes, we're just returning the current status
    
    return status_data


def cancel_training_job(job):
    """Cancel a running training job"""
    try:
        # In a real implementation, we would call Dagster API to cancel the pipeline
        # For demo purposes, we'll just update the job status
        
        # Update job status
        job.status = 'cancelled'
        job.error_message = "Job cancelled by user"
        db.session.commit()
        
        # Cancel MLFlow run (if exists)
        if job.mlflow_run_id:
            mlflow.set_tracking_uri(app.config["MLFLOW_TRACKING_URI"])
            mlflow.end_run(status="KILLED")
        
        logger.info(f"Training job {job.id} cancelled")
        return True
    except Exception as e:
        logger.exception(f"Error cancelling training job: {str(e)}")
        return False


def complete_training_job(job_id, success=True, error_message=None, artifacts=None):
    """Mark a training job as completed or failed"""
    with app.app_context():
        job = db.session.get(TrainingJob, job_id)
        if not job:
            logger.error(f"Training job with ID {job_id} not found")
            return False
        
        job.completed_at = datetime.utcnow()
        
        if success:
            job.status = 'completed'
        else:
            job.status = 'failed'
            job.error_message = error_message
        
        # Save artifacts if provided
        if artifacts and success:
            for artifact_info in artifacts:
                artifact = ModelArtifact(
                    training_job_id=job_id,
                    artifact_path=artifact_info['path'],
                    artifact_type=artifact_info['type']
                )
                
                if 'metrics' in artifact_info:
                    artifact.set_metrics(artifact_info['metrics'])
                
                db.session.add(artifact)
        
        db.session.commit()
        logger.info(f"Training job {job_id} marked as {'completed' if success else 'failed'}")
        return True
