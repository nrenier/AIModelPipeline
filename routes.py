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
from models import Dataset, TrainingJob, ModelArtifact
from forms import DatasetUploadForm, YOLOConfigForm, RFDETRConfigForm
from config import YOLO_MODEL_CONFIGS, RF_DETR_MODEL_CONFIGS, VALID_IMAGE_EXTENSIONS
from ml_pipelines import start_training_job, get_job_status, cancel_training_job
from ml_utils import validate_dataset, get_dataset_stats

logger = logging.getLogger(__name__)

def register_routes(app):
    
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            recent_datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).limit(5).all()
            recent_jobs = TrainingJob.query.filter_by(user_id=current_user.id).order_by(TrainingJob.created_at.desc()).limit(5).all()
            
            # Get stats for active jobs
            active_jobs = TrainingJob.query.filter(
                TrainingJob.user_id == current_user.id,
                TrainingJob.status.in_(['pending', 'running'])
            ).all()
            
            return render_template('index.html', 
                                recent_datasets=recent_datasets, 
                                recent_jobs=recent_jobs,
                                active_jobs=active_jobs)
        else:
            return render_template('index.html')
    
    @app.route('/upload', methods=['GET', 'POST'])
    @login_required
    def upload_dataset():
        form = DatasetUploadForm()
        
        if form.validate_on_submit():
            try:
                # Save uploaded zip file
                zip_file = form.dataset_zip.data
                filename = secure_filename(zip_file.filename)
                dataset_dir = os.path.join(app.config['UPLOAD_FOLDER'], 
                                        f"dataset_{current_user.id}_{uuid.uuid4().hex}")
                os.makedirs(dataset_dir, exist_ok=True)
                zip_path = os.path.join(dataset_dir, filename)
                zip_file.save(zip_path)
                
                # Extract the zip file
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                
                # Validate the dataset
                validation_result = validate_dataset(dataset_dir, form.format_type.data)
                if not validation_result['valid']:
                    flash(f"Invalid dataset: {validation_result['error']}", 'danger')
                    return redirect(url_for('upload_dataset'))
                
                # Create dataset record
                dataset = Dataset(
                    name=form.dataset_name.data,
                    description=form.description.data,
                    data_path=dataset_dir,
                    format_type=form.format_type.data,
                    image_count=validation_result['image_count'],
                    user_id=current_user.id
                )
                
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
    @login_required
    def list_datasets():
        datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).all()
        return render_template('datasets.html', title='My Datasets', datasets=datasets)
    
    @app.route('/dataset/<int:dataset_id>')
    @login_required
    def view_dataset(dataset_id):
        dataset = Dataset.query.get_or_404(dataset_id)
        
        # Ensure the dataset belongs to the current user
        if dataset.user_id != current_user.id:
            flash("You don't have permission to view this dataset.", 'danger')
            return redirect(url_for('list_datasets'))
        
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
    @login_required
    def configure_training():
        model_type = request.args.get('model_type', 'yolo')
        
        # Get list of datasets for dropdown
        user_datasets = Dataset.query.filter_by(user_id=current_user.id).all()
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
                    user_id=current_user.id,
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
                    user_id=current_user.id,
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
    @login_required
    def list_jobs():
        jobs = TrainingJob.query.filter_by(user_id=current_user.id).order_by(TrainingJob.created_at.desc()).all()
        return render_template('jobs.html', title='Training Jobs', jobs=jobs)
    
    @app.route('/monitor/<int:job_id>')
    @login_required
    def monitor_training(job_id):
        job = TrainingJob.query.get_or_404(job_id)
        
        # Ensure the job belongs to the current user
        if job.user_id != current_user.id:
            flash("You don't have permission to view this training job.", 'danger')
            return redirect(url_for('list_jobs'))
        
        return render_template('monitor.html', 
                            title=f'Monitor: {job.job_name}',
                            job=job)
    
    @app.route('/api/job/<int:job_id>/status')
    @login_required
    def job_status(job_id):
        job = TrainingJob.query.get_or_404(job_id)
        
        # Ensure the job belongs to the current user
        if job.user_id != current_user.id:
            return jsonify({'error': 'Permission denied'}), 403
        
        # Get latest job status from MLFlow/Dagster
        status_data = get_job_status(job)
        
        return jsonify(status_data)
    
    @app.route('/api/job/<int:job_id>/cancel', methods=['POST'])
    @login_required
    def cancel_job(job_id):
        job = TrainingJob.query.get_or_404(job_id)
        
        # Ensure the job belongs to the current user
        if job.user_id != current_user.id:
            return jsonify({'error': 'Permission denied'}), 403
        
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
    @login_required
    def training_results(job_id):
        job = TrainingJob.query.get_or_404(job_id)
        
        # Ensure the job belongs to the current user
        if job.user_id != current_user.id:
            flash("You don't have permission to view these results.", 'danger')
            return redirect(url_for('list_jobs'))
        
        # Check if job is completed
        if job.status != 'completed':
            flash("This job has not completed yet.", 'warning')
            return redirect(url_for('monitor_training', job_id=job_id))
        
        # Get model artifacts
        artifacts = ModelArtifact.query.filter_by(training_job_id=job_id).all()
        
        return render_template('results.html', 
                            title=f'Results: {job.job_name}',
                            job=job,
                            artifacts=artifacts)
    
    @app.route('/download/<int:artifact_id>')
    @login_required
    def download_artifact(artifact_id):
        artifact = ModelArtifact.query.get_or_404(artifact_id)
        job = TrainingJob.query.get_or_404(artifact.training_job_id)
        
        # Ensure the artifact belongs to the current user
        if job.user_id != current_user.id:
            flash("You don't have permission to download this artifact.", 'danger')
            return redirect(url_for('list_jobs'))
        
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
