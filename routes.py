import os
import zipfile
import logging
import uuid
import json
from datetime import datetime
from flask import render_template, redirect, url_for, flash, request, jsonify, send_file
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from app import app, db
from models import Dataset, TrainingJob, ModelArtifact, User
from forms import DatasetUploadForm, YOLOConfigForm, RFDETRConfigForm
from config import YOLO_MODEL_CONFIGS, RF_DETR_MODEL_CONFIGS, VALID_IMAGE_EXTENSIONS
from ml_pipelines import start_training_job, get_job_status, cancel_training_job
from ml_utils import validate_dataset, get_dataset_stats

logger = logging.getLogger(__name__)

# Create a default user if it doesn't exist
def ensure_default_user():
    default_user = User.query.filter_by(id=1).first()
    if default_user is None:
        default_user = User()
        default_user.id = 1
        default_user.username = "demo_user"
        default_user.email = "demo@example.com"
        default_user.set_password("demo12345")
        db.session.add(default_user)
        db.session.commit()
        logger.info("Created default user")
    return default_user.id

def register_routes(app):

    # Ensure default user exists when app starts
    with app.app_context():
        default_user_id = ensure_default_user()

    @app.route('/')
    def index():
        # Always use the default user
        user_id = ensure_default_user()
        recent_datasets = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.created_at.desc()).limit(5).all()
        recent_jobs = TrainingJob.query.filter_by(user_id=user_id).order_by(TrainingJob.created_at.desc()).limit(5).all()

        # Get stats for active jobs
        active_jobs = TrainingJob.query.filter(
            TrainingJob.user_id == user_id,
            TrainingJob.status.in_(['pending', 'running'])
        ).all()

        return render_template('index.html', 
                            recent_datasets=recent_datasets, 
                            recent_jobs=recent_jobs,
                            active_jobs=active_jobs)

    @app.route('/upload', methods=['GET', 'POST'])
    # Removed login_required decorator
    def upload_dataset():
        form = DatasetUploadForm()

        if form.validate_on_submit():
            try:
                # Save uploaded zip file
                zip_file = form.dataset_zip.data
                filename = secure_filename(zip_file.filename)
                # Use a default user ID for demonstration purposes
                user_id = 1
                dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 
                                        f"dataset_{user_id}_{uuid.uuid4().hex}")
                os.makedirs(dataset_dir, exist_ok=True)
                zip_path = os.path.join(dataset_dir, filename)
                zip_file.save(zip_path)

                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)

                # Validate the dataset 
                # Assicurati di usare il formato selezionato dall'utente
                format_type = form.format_type.data
                logger.info(f"Formato selezionato dall'utente: {format_type}")

                # Aggiungi log aggiuntivi per debug dettagliato
                logger.debug(f"Form data completo: {request.form.to_dict()}")
                logger.debug(f"Format type dal form: {form.format_type.data}")
                logger.debug(f"Format type raw: {request.form.get('format_type')}")
                logger.debug(f"Files: {request.files}")
                logger.debug(f"Headers: {request.headers}")

                validation_result = validate_dataset(dataset_dir, format_type)
                if not validation_result['valid']:
                    flash(f"Invalid dataset: {validation_result['error']}", 'danger')
                    return redirect(url_for('upload_dataset'))

                # Create dataset record
                dataset = Dataset()
                dataset.name = form.dataset_name.data
                dataset.description = form.description.data
                dataset.data_path = dataset_dir
                dataset.format_type = form.format_type.data
                dataset.image_count = validation_result['image_count']
                dataset.user_id = user_id  # Use our default user_id

                # Debug log for dataset creation
                logger.info(f"Creating dataset: {dataset.name}, path: {dataset.data_path}, user_id: {dataset.user_id}")

                # Set class names
                dataset.set_class_names(validation_result['classes'])

                # Save to database
                db.session.add(dataset)
                db.session.commit()

                flash(f"Dataset '{form.dataset_name.data}' uploaded successfully!", 'success')
                return redirect(url_for('index'))

            except Exception as e:
                logger.exception("Error uploading dataset")
                flash(f"Error uploading dataset: {str(e)}", 'danger')
                return redirect(url_for('upload_dataset'))

        return render_template('upload.html', title='Upload Dataset', form=form)

    @app.route('/datasets')
    # Removed login_required
    def list_datasets():
        # Use default user
        user_id = ensure_default_user()
        datasets = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.created_at.desc()).all()
        logger.info(f"Found {len(datasets)} datasets for user_id {user_id}")

        if len(datasets) == 0:
            # Check if any datasets exist in the filesystem
            datasets_in_directory = os.listdir(app.config['UPLOAD_FOLDER'])
            logger.info(f"Datasets in directory: {datasets_in_directory}")

            # Recreate dataset records if they exist in filesystem but not in database
            if datasets_in_directory:
                for dataset_dir in datasets_in_directory:
                    full_path = os.path.join(app.config['UPLOAD_FOLDER'], dataset_dir)
                    if os.path.isdir(full_path) and dataset_dir.startswith('dataset_'):
                        try:
                            # Try to determine format type from directory contents
                            format_type = 'coco'  # Default to COCO
                            if os.path.exists(os.path.join(full_path, 'data.yaml')):
                                format_type = 'yolo'

                            # Create a new dataset record
                            dataset = Dataset()
                            dataset.name = f"Recovered Dataset {dataset_dir}"
                            dataset.description = "Automatically recovered dataset"
                            dataset.data_path = full_path
                            dataset.format_type = format_type
                            dataset.user_id = user_id

                            # Run validation to get the image count and classes
                            try:
                                validation_result = validate_dataset(full_path, format_type)
                                if validation_result['valid']:
                                    dataset.image_count = validation_result['image_count']
                                    dataset.set_class_names(validation_result['classes'])
                                    logger.info(f"Validated dataset with {validation_result['image_count']} images and classes: {validation_result['classes']}")
                                else:
                                    logger.warning(f"Dataset validation failed: {validation_result['error']}")
                                    # Count images manually as fallback
                                    total_images = 0
                                    for subdir in ['train', 'test', 'valid']:
                                        subdir_path = os.path.join(full_path, subdir)
                                        if os.path.exists(subdir_path):
                                            image_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                            total_images += len(image_files)
                                    dataset.image_count = total_images
                                    # Try to get class names from README or other files
                                    readme_files = [f for f in os.listdir(full_path) if 'readme' in f.lower()]
                                    if readme_files:
                                        with open(os.path.join(full_path, readme_files[0]), 'r') as f:
                                            content = f.read()
                                            # Simple extraction of possible class names
                                            if 'class' in content.lower() and ':' in content:
                                                class_text = content.lower().split('class')[1].split('\n')[0]
                                                class_names = [c.strip() for c in class_text.split(':')[1].split(',')]
                                                dataset.set_class_names(class_names)
                            except Exception as e:
                                logger.error(f"Error validating dataset: {str(e)}")
                                # Emergency count of images
                                total_images = 0
                                try:
                                    for root, dirs, files in os.walk(full_path):
                                        for file in files:
                                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                                total_images += 1
                                    dataset.image_count = total_images
                                    logger.info(f"Counted {total_images} images as emergency fallback")
                                except:
                                    dataset.image_count = 0
                                dataset.set_class_names([])

                            db.session.add(dataset)
                            db.session.commit()
                            logger.info(f"Recovered dataset record for {dataset_dir}")
                        except Exception as e:
                            logger.exception(f"Error recovering dataset {dataset_dir}: {str(e)}")

                # Fetch datasets again after recovery
                datasets = Dataset.query.filter_by(user_id=user_id).order_by(Dataset.created_at.desc()).all()
                logger.info(f"After recovery: Found {len(datasets)} datasets for user_id {user_id}")

        return render_template('datasets.html', title='My Datasets', datasets=datasets)

    @app.route('/dataset/<int:dataset_id>')
    # Removed login_required
    def view_dataset(dataset_id):
        dataset = Dataset.query.get_or_404(dataset_id)

        # Get dataset statistics
        stats = get_dataset_stats(dataset)

        # Get associated training jobs
        jobs = TrainingJob.query.filter_by(dataset_id=dataset_id).order_by(TrainingJob.created_at.desc()).all()

        return render_template('dataset_details.html', 
                            title=f'Dataset: {dataset.name}',
                            dataset=dataset,
                            stats=stats,
                            jobs=jobs)

    @app.route('/configure', methods=['GET', 'POST'])
    # Removed login_required
    def configure_training():
        model_type = request.args.get('model_type', 'yolo')

        # Get list of datasets for dropdown
        user_id = ensure_default_user()
        user_datasets = Dataset.query.filter_by(user_id=user_id).all()
        if not user_datasets:
            flash("You need to upload a dataset before configuring training.", "warning")
            return redirect(url_for('upload_dataset'))

        dataset_choices = [(d.id, d.name) for d in user_datasets]

        if model_type == 'yolo':
            form = YOLOConfigForm()
            form.dataset_id.choices = dataset_choices
            config_data = YOLO_MODEL_CONFIGS

            if form.validate_on_submit():
                # Create training job for YOLO
                hyperparams = {
                    'epochs': form.epochs.data,
                    'batch_size': form.batch_size.data,
                    'img_size': form.img_size.data,
                    'learning_rate': form.learning_rate.data,
                    'pretrained': form.pretrained.data
                }

                job = TrainingJob(
                    job_name=form.job_name.data,
                    model_type='yolo',
                    model_variant=form.model_variant.data,
                    user_id=user_id,  # Use default user
                    dataset_id=form.dataset_id.data,
                    status='pending'
                )
                job.set_hyperparameters(hyperparams)

                db.session.add(job)
                db.session.commit()

                # Start the training job asynchronously
                try:
                    start_training_job(job.id)
                    flash("Training job submitted successfully!", "success")
                except Exception as e:
                    logger.exception("Error starting training job")
                    job.status = 'failed'
                    job.error_message = str(e)
                    db.session.commit()
                    flash(f"Error starting training job: {str(e)}", "danger")

                return redirect(url_for('monitor_training', job_id=job.id))

        elif model_type == 'rf-detr':
            form = RFDETRConfigForm()
            form.dataset_id.choices = dataset_choices
            config_data = RF_DETR_MODEL_CONFIGS

            if form.validate_on_submit():
                # Create training job for RF-DETR
                hyperparams = {
                    'epochs': form.epochs.data,
                    'batch_size': form.batch_size.data,
                    'img_size': form.img_size.data,
                    'learning_rate': form.learning_rate.data,
                    'pretrained': form.pretrained.data
                }

                job = TrainingJob(
                    job_name=form.job_name.data,
                    model_type='rf-detr',
                    model_variant=form.model_variant.data,
                    user_id=user_id,  # Use default user
                    dataset_id=form.dataset_id.data,
                    status='pending'
                )
                job.set_hyperparameters(hyperparams)

                db.session.add(job)
                db.session.commit()

                # Start the training job asynchronously
                try:
                    start_training_job(job.id)
                    flash("Training job submitted successfully!", "success")
                except Exception as e:
                    logger.exception("Error starting training job")
                    job.status = 'failed'
                    job.error_message = str(e)
                    db.session.commit()
                    flash(f"Error starting training job: {str(e)}", "danger")

                return redirect(url_for('monitor_training', job_id=job.id))
        else:
            flash("Invalid model type selected", "danger")
            return redirect(url_for('index'))

        # Pre-fill form with default values for selected model
        if request.method == 'GET' and 'model_variant' in request.args:
            variant = request.args.get('model_variant')
            if variant in config_data:
                form.model_variant.data = variant
                form.epochs.data = config_data[variant]['default_epochs']
                form.batch_size.data = config_data[variant]['default_batch_size']
                form.img_size.data = config_data[variant]['default_img_size']
                form.learning_rate.data = config_data[variant]['default_learning_rate']

        return render_template('configure.html', 
                            title='Configure Training', 
                            form=form, 
                            model_type=model_type,
                            config_data=config_data)

    @app.route('/jobs')
    # Removed login_required
    def list_jobs():
        user_id = ensure_default_user()
        jobs = TrainingJob.query.filter_by(user_id=user_id).order_by(TrainingJob.created_at.desc()).all()
        return render_template('jobs.html', title='Training Jobs', jobs=jobs)

    @app.route('/monitor/<int:job_id>')
    # Removed login_required
    def monitor_training(job_id):
        job = TrainingJob.query.get_or_404(job_id)
        return render_template('monitor.html', 
                            title=f'Monitor: {job.job_name}',
                            job=job)

    @app.route('/api/jobs/<int:job_id>/status')
    # Removed login_required
    def job_status(job_id):
        job = TrainingJob.query.get_or_404(job_id)
        # Get latest job status from MLFlow/Dagster
        status_data = get_job_status(job)
        return jsonify(status_data)

    @app.route('/api/job/<int:job_id>/cancel', methods=['POST'])
    # Removed login_required
    def cancel_job(job_id):
        job = TrainingJob.query.get_or_404(job_id)

        # Check if job can be cancelled
        if job.status not in ['pending', 'running']:
            return jsonify({'error': 'Job cannot be cancelled in its current state'}), 400

        # Cancel the job
        success = cancel_training_job(job)

        if success:
            job.status = 'cancelled'
            db.session.commit()
            return jsonify({'success': True, 'message': 'Job cancelled successfully'})
        else:
            return jsonify({'error': 'Failed to cancel job'}), 500

    @app.route('/results/<int:job_id>')
    # Removed login_required
    def training_results(job_id):
        job = TrainingJob.query.get_or_404(job_id)
        dataset = Dataset.query.get(job.dataset_id) if job.dataset_id else None

        # Check if job is completed
        if job.status != 'completed':
            flash("This job has not completed yet.", 'warning')
            return redirect(url_for('monitor_training', job_id=job_id))

        # Get model artifacts
        artifacts = ModelArtifact.query.filter_by(training_job_id=job_id).all()
        
        # Initialize metrics
        metrics = {}
        
        # Extract metrics from artifact if available
        for artifact in artifacts:
            if artifact.artifact_type == 'weights' and artifact.metrics:
                metrics = artifact.get_metrics()
                break
        
        # Get model path and size from artifacts
        model_path = None
        model_size = None
        for artifact in artifacts:
            if artifact.artifact_type == 'weights':
                model_path = artifact.artifact_path
                if model_path and os.path.exists(model_path):
                    model_size = round(os.path.getsize(model_path) / (1024 * 1024), 2)  # Size in MB
                break

        # Get hyperparameters from job
        hyperparameters = job.get_hyperparameters()
        
        # Prepare metrics history (dummy data for now)
        metrics_history = None
        
        return render_template('results.html', 
                            title=f'Results: {job.job_name}',
                            job=job,
                            dataset=dataset,
                            artifacts=artifacts,
                            metrics=metrics,
                            hyperparameters=hyperparameters,
                            model_path=model_path,
                            model_size=model_size,
                            metrics_history=metrics_history)

    @app.route('/download/<int:artifact_id>')
    # Removed login_required
    def download_artifact(artifact_id):
        artifact = ModelArtifact.query.get_or_404(artifact_id)
        job = TrainingJob.query.get_or_404(artifact.training_job_id)

        # Check if file exists
        if not os.path.exists(artifact.artifact_path):
            flash("Artifact file not found.", 'danger')
            return redirect(url_for('training_results', job_id=job.id))

        # Determine file type and set download name
        filename = os.path.basename(artifact.artifact_path)

        return send_file(artifact.artifact_path, 
                        as_attachment=True,
                        download_name=filename)

    # Error handlers
    @app.errorhandler(404)
    def not_found_error(error):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return render_template('500.html'), 500