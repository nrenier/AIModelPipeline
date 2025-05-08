
import os
import json
import logging
import time
import threading
import mlflow
from datetime import datetime
from models import ModelArtifact

# Configura il logger
logger = logging.getLogger(__name__)

def run_training_job(job, app_config):
    """
    Esegue un job di training direttamente senza Dagster
    
    Args:
        job: oggetto TrainingJob dal database
        app_config: configurazione dell'applicazione
    """
    from app import db
    
    try:
        logger.info(f"Avvio training diretto per job {job.id} - {job.job_name}")
        
        # Connetti a MLFlow
        mlflow.set_tracking_uri(app_config["MLFLOW_TRACKING_URI"])
        
        # Ottieni il dataset e altri parametri
        dataset = job.dataset
        if not dataset:
            raise ValueError(f"Dataset non trovato per il job {job.id}")
        
        # Simula le fasi di training
        logger.info(f"Caricamento dataset: {dataset.data_path}")
        # Fase 1: Caricamento dataset - simulato
        time.sleep(3)
        
        # Fase 2: Training del modello
        hyperparams = job.get_hyperparameters()
        logger.info(f"Avvio training modello {job.model_type} variante {job.model_variant}")
        logger.info(f"Parametri: {hyperparams}")
        
        # Simula il progresso del training
        total_epochs = hyperparams.get('epochs', 100)
        
        # Log in MLFlow (se disponibile)
        try:
            mlflow_run = mlflow.get_run(job.mlflow_run_id)
            for epoch in range(1, total_epochs + 1):
                # Simula il training per ogni epoca
                time.sleep(0.2)  # Più veloce per test
                
                # Calcola metriche simulate
                loss = max(1.0 - (epoch / total_epochs * 0.8), 0.2)
                precision = min(0.5 + (epoch / total_epochs * 0.4), 0.9)
                recall = min(0.5 + (epoch / total_epochs * 0.4), 0.9)
                
                # Log nella console
                if epoch % 10 == 0 or epoch == total_epochs:
                    logger.info(f"Epoca {epoch}/{total_epochs}: loss={loss:.4f}, precision={precision:.4f}, recall={recall:.4f}")
                
                # Log in MLFlow
                mlflow.log_metrics({
                    'epoch': epoch,
                    'loss': loss,
                    'precision': precision,
                    'recall': recall
                }, step=epoch)
        except Exception as e:
            logger.warning(f"Errore nel logging MLFlow: {str(e)}")
            # Continua comunque con il training simulato
            for epoch in range(1, total_epochs + 1):
                time.sleep(0.2)
                if epoch % 10 == 0 or epoch == total_epochs:
                    logger.info(f"Epoca {epoch}/{total_epochs} (logging locale)")
        
        # Fase 3: Salvataggio modello e metriche
        logger.info("Training completato, salvataggio modello...")
        time.sleep(2)
        
        # Simula la creazione di un modello
        model_path = f"/tmp/model_{job.model_type}_{job.model_variant}_{job.id}.pt"
        
        # Crea un artifact per il modello
        with app_config['FLASK_APP'].app_context():
            artifact = ModelArtifact()
            artifact.training_job_id = job.id
            artifact.artifact_path = model_path
            artifact.artifact_type = "weights"
            
            # Metriche finali
            metrics = {
                'precision': 0.85,
                'recall': 0.83,
                'mAP50': 0.87,
                'mAP50-95': 0.62,
                'epochs_completed': total_epochs
            }
            artifact.set_metrics(metrics)
            
            # Aggiorna stato del job
            job.status = 'completed'
            job.completed_at = datetime.utcnow()
            
            # Salva nel database
            db.session.add(artifact)
            db.session.commit()
            
            logger.info(f"Job {job.id} completato con successo, artifact salvato")
        
        return True
    
    except Exception as e:
        logger.exception(f"Errore nell'esecuzione del job {job.id}: {str(e)}")
        
        # Aggiorna stato del job in caso di errore
        with app_config['FLASK_APP'].app_context():
            job.status = 'failed'
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            db.session.commit()
        
        return False


def execute_training_job_async(job_id, app_config):
    """
    Esegue un job di training in modo asincrono in un thread separato
    
    Args:
        job_id: ID del job di training
        app_config: configurazione dell'applicazione
    """
    from app import db
    
    def _run_job_thread():
        with app_config['FLASK_APP'].app_context():
            # Ottieni il job dal database
            job = db.session.query(app_config['MODELS']['TrainingJob']).get(job_id)
            if not job:
                logger.error(f"Job {job_id} non trovato nel database")
                return
            
            # Esegui il job
            run_training_job(job, app_config)
    
    # Avvia in un thread separato
    thread = threading.Thread(target=_run_job_thread)
    thread.daemon = True  # Assicura che il thread termini quando l'app principale termina
    thread.start()
    
    logger.info(f"Job {job_id} avviato in thread separato")
    return thread
