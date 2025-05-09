
// Main JavaScript functions for the application

// Handle MLFlow synchronization
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    $('[data-toggle="tooltip"]').tooltip();
    
    // MLFlow sync button
    const syncMlflowBtn = document.getElementById('sync-mlflow-btn');
    if (syncMlflowBtn) {
        syncMlflowBtn.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            syncMlflowMetrics(jobId);
        });
    }
});

// Function to sync MLFlow metrics
function syncMlflowMetrics(jobId) {
    // Disable button and show loading
    const btn = document.getElementById('sync-mlflow-btn');
    if (btn) {
        const originalText = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sincronizzazione...';
    
        // Call API endpoint
        fetch(`/api/job/${jobId}/sync_mlflow`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Show success message
                showToast('Sincronizzazione completata', 'Le metriche e gli artefatti sono stati sincronizzati con MLFlow', 'success');
                
                // Open MLFlow in new tab
                const mlflowUrl = `/mlflow/#/experiments?{%22searchFilter%22:%22run_id='${data.mlflow_run_id}'%22}`;
                window.open(mlflowUrl, '_blank');
            } else {
                showToast('Errore di sincronizzazione', data.error || 'Errore durante la sincronizzazione con MLFlow', 'error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showToast('Errore di sincronizzazione', 'Si è verificato un errore durante la richiesta', 'error');
        })
        .finally(() => {
            // Re-enable button
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = originalText;
            }
        });
    }
}

// Show toast notification
function showToast(title, message, type = 'info') {
    // Check if we have a toast container, if not create one
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed bottom-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toastId = 'toast-' + Date.now();
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type === 'error' ? 'danger' : type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    toast.id = toastId;
    
    // Toast content
    toast.innerHTML = `
    <div class="d-flex">
        <div class="toast-body">
            <strong>${title}</strong><br>${message}
        </div>
        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
    </div>`;
    
    // Add toast to container
    toastContainer.appendChild(toast);
    
    // Initialize and show the toast
    const bsToast = new bootstrap.Toast(toast, {
        autohide: true,
        delay: 5000
    });
    bsToast.show();
    
    // Remove toast from DOM after hiding
    toast.addEventListener('hidden.bs.toast', function() {
        toast.remove();
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips and popovers
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });
    
    // Handle model type selection
    const modelTypeSelectors = document.querySelectorAll('.model-type-selector');
    if (modelTypeSelectors) {
        modelTypeSelectors.forEach(button => {
            button.addEventListener('click', function() {
                const modelType = this.getAttribute('data-model-type');
                window.location.href = `/configure?model_type=${modelType}`;
            });
        });
    }
    
    // Handle model variant selection
    const variantSelectors = document.querySelectorAll('.variant-selector');
    if (variantSelectors) {
        variantSelectors.forEach(card => {
            card.addEventListener('click', function() {
                // Remove selected class from all cards
                variantSelectors.forEach(c => c.classList.remove('border-primary'));
                
                // Add selected class to clicked card
                this.classList.add('border-primary');
                
                // Get variant and update form value
                const variant = this.getAttribute('data-variant');
                document.getElementById('model_variant').value = variant;
                
                // Get default values for this variant
                const defaultValues = JSON.parse(this.getAttribute('data-defaults'));
                
                // Update form fields with default values
                if (defaultValues) {
                    if (defaultValues.default_epochs) {
                        document.getElementById('epochs').value = defaultValues.default_epochs;
                    }
                    
                    if (defaultValues.default_batch_size) {
                        document.getElementById('batch_size').value = defaultValues.default_batch_size;
                    }
                    
                    if (defaultValues.default_img_size) {
                        document.getElementById('img_size').value = defaultValues.default_img_size;
                    }
                    
                    if (defaultValues.default_learning_rate) {
                        document.getElementById('learning_rate').value = defaultValues.default_learning_rate;
                    }
                }
            });
        });
    }
    
    // Handle cancel job button
    const cancelButtons = document.querySelectorAll('.cancel-job-btn');
    if (cancelButtons) {
        cancelButtons.forEach(button => {
            button.addEventListener('click', function() {
                const jobId = this.getAttribute('data-job-id');
                
                if (confirm('Are you sure you want to cancel this training job?')) {
                    // Send request to cancel job
                    fetch(`/api/job/${jobId}/cancel`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Job cancelled successfully');
                            window.location.reload();
                        } else {
                            alert(`Error: ${data.error}`);
                        }
                    })
                    .catch(error => {
                        console.error('Error cancelling job:', error);
                        alert('An error occurred while cancelling the job');
                    });
                }
            });
        });
    }
});
