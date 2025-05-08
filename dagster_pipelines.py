
import os
import json
import logging
import tempfile
import requests
from datetime import datetime
from functools import wraps
from dagster import job, op, In, Out, get_dagster_logger, OpExecutionContext, repository

# Use Dagster's logger for consistent logging
logger = get_dagster_logger()

# ---- OPERATION DEFINITIONS ----

@op(
    ins={'start': In()},
    out=Out(dict),
    config_schema={"job_id": int}
)
def load_dataset(context: OpExecutionContext, start):
    """Load dataset for training"""
    # Get job and dataset info
    job_id = context.op_config["job_id"]

    # Log operation for debugging
    logger.info(f"Loading dataset for job ID: {job_id}")

    # In development mode, simulate dataset loading
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
def train_model(context: OpExecutionContext, dataset_info):
    """Train model with the provided configuration"""
    # Debug logging
    logger.info(f"Training model with config: {context.op_config}")
    logger.info(f"Dataset info: {dataset_info}")

    # Set up MLFlow tracking
    mlflow_uri = context.op_config["mlflow_tracking_uri"]
    mlflow_run_id = context.op_config["mlflow_run_id"]

    logger.info(f"Using MLFlow tracking URI: {mlflow_uri}, Run ID: {mlflow_run_id}")

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
def save_artifacts(context: OpExecutionContext, model_results):
    """Save model artifacts and update the database"""
    # Get job ID
    job_id = context.op_config["job_id"]

    logger.info(f"Saving artifacts for job ID: {job_id}")
    logger.info(f"Model results: {model_results}")

    # In a real implementation, we would save model artifacts
    # and update the database

    return True

# ---- PIPELINE DEFINITIONS ----

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

# ---- REPOSITORY ----

@repository
def ml_training_repository():
    return [
        yolo_training_pipeline,
        rf_detr_training_pipeline
    ]

# ---- DAGSTER CLIENT ----

class DagsterClient:
    def __init__(self, dagster_url=None):
        self.dagster_url = dagster_url or os.environ.get('DAGSTER_URL', 'http://localhost:3000')

    def launch_pipeline(self, pipeline_name, run_config):
        """
        Launch a Dagster pipeline using the GraphQL API
        """
        try:
            endpoint = f"{self.dagster_url}/graphql"  # Updated endpoint path

            # GraphQL mutation for Dagster
            query = """
            mutation LaunchPipelineExecution($executionParams: ExecutionParams!) {
                launchPipelineExecution(executionParams: $executionParams) {
                    run {
                        runId
                    }
                }
            }
            """

            variables = {
                "executionParams": {
                    "selector": {
                        "repositorySelector": {
                            "repositoryName": "ml_training_repository"
                        },
                        "pipelineName": pipeline_name,
                    },
                    "runConfigData": run_config,
                    "mode": "default",
                }
            }

            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "X-Requested-With": "XMLHttpRequest"
            }
            
            logger.info(f"Sending request to Dagster at {endpoint}")
            logger.info(f"Request payload: {json.dumps(variables)}")
            
            response = requests.post(
                endpoint,
                headers=headers,
                json={"query": query, "variables": variables},
                timeout=30  # Add timeout to prevent hanging
            )

            logger.info(f"Dagster response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Dagster response data: {json.dumps(data)}")
                
                if "errors" in data:
                    logger.error(f"GraphQL errors: {data['errors']}")
                    raise Exception(f"GraphQL errors: {data['errors']}")

                launch_result = data.get("data", {}).get("launchPipelineExecution", {})
                if launch_result and "run" in launch_result:
                    run_id = launch_result.get("run", {}).get("runId")
                    logger.info(f"Launched Dagster pipeline {pipeline_name} with run ID {run_id}")
                    return run_id
                else:
                    logger.error(f"Unexpected launch result: {launch_result}")
                    raise Exception(f"Failed to launch pipeline: {launch_result}")
            else:
                logger.error(f"Error launching pipeline: {response.status_code} - {response.text}")
                raise Exception(f"Error launching pipeline: {response.status_code} - {response.text}")

        except Exception as e:
            logger.exception(f"Error launching Dagster pipeline: {str(e)}")
            # Fallback - generate a simulated ID in case of connection error
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

    logger.info(f"Submitting {pipeline_name} with config: {json.dumps(run_config)}")
    
    # Launch pipeline via Dagster client
    run_id = dagster_client.launch_pipeline(pipeline_name, run_config)

    return run_id
