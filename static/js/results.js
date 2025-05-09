
/**
 * JavaScript for the results page to handle MLFlow sync
 */
document.addEventListener('DOMContentLoaded', function() {
    // Handle MLFlow sync button click
    const syncBtn = document.getElementById('sync-mlflow-btn');
    if (syncBtn) {
        syncBtn.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            
            // Disable button during sync
            this.disabled = true;
            this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Syncing...';
            
            // Make API call to sync
            fetch(`/api/job/${jobId}/sync_mlflow`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // Show success alert
                    const alertDiv = document.createElement('div');
                    alertDiv.className = 'alert alert-success alert-dismissible fade show mt-3';
                    alertDiv.innerHTML = `<i class="fas fa-check-circle me-2"></i> ${data.message} <button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
                    
                    // Insert alert before the button's parent element
                    this.closest('.card-header').after(alertDiv);
                    
                    // Re-enable button
                    this.disabled = false;
                    this.innerHTML = '<i class="fas fa-sync"></i> Sync MLFlow';
                } else {
                    throw new Error(data.error || 'Unknown error occurred');
                }
            })
            .catch(error => {
                // Show error alert
                const alertDiv = document.createElement('div');
                alertDiv.className = 'alert alert-danger alert-dismissible fade show mt-3';
                alertDiv.innerHTML = `<i class="fas fa-exclamation-circle me-2"></i> ${error.message} <button type="button" class="btn-close" data-bs-dismiss="alert"></button>`;
                
                // Insert alert before the button's parent element
                this.closest('.card-header').after(alertDiv);
                
                // Re-enable button
                this.disabled = false;
                this.innerHTML = '<i class="fas fa-sync"></i> Sync MLFlow';
            });
        });
    }
});
