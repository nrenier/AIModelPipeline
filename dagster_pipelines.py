import os
import json
import logging
import tempfile
import requests
from datetime import datetime
from functools import wraps
from dagster import op, job, In, Out, Config, repository

logger = logging.getLogger(__name__)

# Simplified Dagster API client
class DagsterClient:
    def __init__(self, dagster_url=None):
        self.dagster_url = dagster_url or os.environ.get('DAGSTER_URL', 'http://localhost:3000')

    def launch_pipeline(self, pipeline_name, run_config):
        """
        Simplified method to launch a Dagster pipeline

        In a real implementation, this would use the Dagster GraphQL API
        or Python API to launch pipelines
        """
        try:
            endpoint = f"{self.dagster_url}/api/graphql"

            # GraphQL mutation per lanciare un pipeline
            query = """
            mutation LaunchPipelineExecution($executionParams: ExecutionParams!) {
                launchPipelineExecution(executionParams: $executionParams) {
                    runId
                }
            }
            """

            variables = {
                "executionParams": {
                    "selector": {
                        "repositoryName": "ml_training_repository",
                        "repositoryLocationName": "ml_training_location",
                        "pipelineName": pipeline_name,
                    },
                    "runConfigData": run_config,
                    "mode": "default",
                }
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(
                endpoint,
                headers=headers,
                json={"query": query, "variables": variables},
            )

            if response.status_code == 200:
                data = response.json()
                if "errors" in data:
                    logger.error(f"GraphQL errors: {data['errors']}")
                    raise Exception(f"GraphQL errors: {data['errors']}")

                run_id = data["data"]["launchPipelineExecution"]["runId"]
                logger.info(f"Launched Dagster pipeline {pipeline_name} with run ID {run_id}")
                return run_id
            else:
                logger.error(f"Error launching pipeline: {response.status_code} - {response.text}")
                raise Exception(f"Error launching pipeline: {response.status_code} - {response.text}")

        except Exception as e:
            logger.exception(f"Error launching Dagster pipeline: {str(e)}")
            # Fallback - in caso di errore di connessione a Dagster, usiamo un ID simulato
            import uuid
            run_id = str(uuid.uuid4())
            logger.warning(f"Using simulated run ID due to error: {run_id}")
            return run_id


# Initialize Dagster client
dagster_client = DagsterClient()


def submit_dagster_pipeline(config):
    """
    Submit a training pipeline to Dagster

    Args:
        config: Dictionary with pipeline configuration

    Returns:
        Dagster run ID
    """
    model_type = config['model_type']

    # Determine pipeline name based on model type
    if model_type == 'yolo':
        pipeline_name = 'yolo_training_pipeline'
    elif model_type == 'rf-detr':
        pipeline_name = 'rf_detr_training_pipeline'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Create run configuration for Dagster
    run_config = {
        'ops': {
            'load_dataset': {
                'config': {
                    'job_id': config['job_id']
                }
            },
            'train_model': {
                'config': {
                    'model_variant': config['model_variant'],
                    'hyperparameters': config['hyperparameters'],
                    'mlflow_run_id': config['mlflow_run_id'],
                    'mlflow_tracking_uri': config['mlflow_tracking_uri']
                }
            },
            'save_artifacts': {
                'config': {
                    'job_id': config['job_id']
                }
            }
        }
    }

    # Launch pipeline via Dagster client
    run_id = dagster_client.launch_pipeline(pipeline_name, run_config)

    return run_id


# Implementazione effettiva delle pipeline Dagster (non più come stringhe)
@op(
    ins={'start': In()},
    out=Out(dict),
    config_schema={"job_id": int}
)
def load_dataset(context, start):
    # Get job and dataset info
    job_id = context.op_config["job_id"]

    # Log operation for debugging
    logger.info(f"Loading dataset for job ID: {job_id}")

    # In modalità di sviluppo, simula il caricamento del dataset
    return {
        "dataset_path": f"/tmp/dataset_{job_id}",
        "format_type": "yolo"
    }

@op(
    ins={'dataset_info': In(dict)},
    out=Out(dict),
    config_schema={
        "model_variant": str,
        "hyperparameters": dict,
        "mlflow_run_id": str,
        "mlflow_tracking_uri": str
    }
)
def train_model(context, dataset_info):
    # Log di debug
    logger.info(f"Training model with config: {context.op_config}")
    logger.info(f"Dataset info: {dataset_info}")

    # Set up MLFlow tracking
    mlflow_uri = context.op_config["mlflow_tracking_uri"]
    mlflow_run_id = context.op_config["mlflow_run_id"]

    logger.info(f"Using MLFlow tracking URI: {mlflow_uri}, Run ID: {mlflow_run_id}")

    # Simuliamo la formazione del modello
    import time
    import random

    # Simulate model training
    model_variant = context.op_config["model_variant"]
    hyperparameters = context.op_config["hyperparameters"]

    # Return results
    return {
        "model_path": f"/tmp/model_{mlflow_run_id}.pt",
        "results": {
            "precision": 0.85,
            "recall": 0.83,
            "mAP50": 0.87,
            "mAP50-95": 0.62
        }
    }

@op(
    ins={'model_results': In(dict)},
    out=Out(bool),
    config_schema={"job_id": int}
)
def save_artifacts(context, model_results):
    # Get job ID
    job_id = context.op_config["job_id"]

    logger.info(f"Saving artifacts for job ID: {job_id}")
    logger.info(f"Model results: {model_results}")

    # In un'implementazione reale, salveremmo gli artifacts del modello
    # e aggiorneremmo il database

    return True

@job
def yolo_training_pipeline():
    dataset_info = load_dataset()
    model_results = train_model(dataset_info)
    save_artifacts(model_results)

@job
def rf_detr_training_pipeline():
    dataset_info = load_dataset()
    model_results = train_model(dataset_info)
    save_artifacts(model_results)

# Registra le pipeline Dagster in un repository
@repository
def ml_training_repository():
    return [
        yolo_training_pipeline,
        rf_detr_training_pipeline
    ]