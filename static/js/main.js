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
