
// File: test.js
document.addEventListener('DOMContentLoaded', function() {
    const testForm = document.getElementById('test-form');
    const testImage = document.getElementById('test-image');
    const modelSelect = document.getElementById('model-select');
    const confidenceThreshold = document.getElementById('confidence-threshold');
    const thresholdValue = document.getElementById('threshold-value');
    const runInferenceBtn = document.getElementById('run-inference-btn');
    const imagePreviewContainer = document.getElementById('image-preview-container');
    const resultsCard = document.getElementById('results-card');
    const resultsImage = document.getElementById('results-image');
    const detectionList = document.getElementById('detection-list');
    const inferenceTime = document.getElementById('inference-time');
    const totalObjects = document.getElementById('total-objects');
    const classFilterSection = document.getElementById('class-filter-section');
    const classFilterContainer = document.getElementById('class-filter-container');
    const selectAllClassesBtn = document.getElementById('select-all-classes');
    const clearAllClassesBtn = document.getElementById('clear-all-classes');
    
    let availableClasses = [];
    let lastInferenceData = null;
    
    // Update threshold value display
    confidenceThreshold.addEventListener('input', function() {
        thresholdValue.textContent = this.value;
    });
    
    // Class filter event handlers
    selectAllClassesBtn.addEventListener('click', function() {
        const checkboxes = classFilterContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => checkbox.checked = true);
        if (lastInferenceData) {
            applyClassFilter();
        }
    });
    
    clearAllClassesBtn.addEventListener('click', function() {
        const checkboxes = classFilterContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => checkbox.checked = false);
        if (lastInferenceData) {
            applyClassFilter();
        }
    });
    
    // Display model info when selected
    modelSelect.addEventListener('change', function() {
        const selectedOption = this.options[this.selectedIndex];
        const modelInfo = document.getElementById('model-info');
        
        if (selectedOption.value) {
            const modelType = selectedOption.getAttribute('data-model-type');
            
            if (modelType === 'rf-detr') {
                modelInfo.innerHTML = 
                    '<i class="fas fa-info-circle"></i> RF-DETR models use transformer architecture for higher accuracy.';
            } else if (modelType === 'yolo') {
                modelInfo.innerHTML = 
                    '<i class="fas fa-info-circle"></i> YOLO models are optimized for real-time detection.';
            } else {
                modelInfo.innerHTML = 'Select a trained model to perform object detection.';
            }
        } else {
            modelInfo.innerHTML = 'Select a trained model to perform object detection.';
        }
    });
    
    // Display image preview when selected
    testImage.addEventListener('change', function(e) {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreviewContainer.innerHTML = `
                    <img src="${e.target.result}" class="img-fluid" alt="Preview">
                `;
            };
            reader.readAsDataURL(file);
        } else {
            imagePreviewContainer.innerHTML = `
                <p class="text-muted">Upload an image to see preview</p>
            `;
        }
    });
    
    // Handle form submission
    testForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Validate form
        if (!testImage.files[0]) {
            alert('Please select an image to test');
            return;
        }
        
        if (!modelSelect.value) {
            alert('Please select a model');
            return;
        }
        
        // Show loading state
        runInferenceBtn.disabled = true;
        runInferenceBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Processing...';
        
        // Create FormData and submit
        const formData = new FormData();
        formData.append('image', testImage.files[0]);
        formData.append('model_id', modelSelect.value);
        formData.append('threshold', confidenceThreshold.value);
        
        // Send request to server
        fetch('/api/test/inference', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset button state
            runInferenceBtn.disabled = false;
            runInferenceBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>Run Inference';
            
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Display results
            displayResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            runInferenceBtn.disabled = false;
            runInferenceBtn.innerHTML = '<i class="fas fa-play-circle me-2"></i>Run Inference';
            alert('An error occurred while processing the image.');
        });
    });
    
    // Function to display results
    function displayResults(data) {
        // Store the data for filtering
        lastInferenceData = data;
        
        // Extract unique classes from detections
        availableClasses = [...new Set(data.detections.map(det => det.class))].sort();
        
        // Show and populate class filter section
        if (availableClasses.length > 0) {
            populateClassFilter();
            classFilterSection.style.display = 'block';
        }
        
        // Show results card
        resultsCard.style.display = 'block';
        
        // Set results image
        resultsImage.src = data.image_url + '?t=' + new Date().getTime();
        
        // Set inference time and total objects
        inferenceTime.textContent = data.inference_time + ' ms';
        totalObjects.textContent = data.detections.length;
        
        // Display detection results
        displayDetectionResults(data.detections);
        
        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }
    
    // Function to populate class filter checkboxes
    function populateClassFilter() {
        classFilterContainer.innerHTML = '';
        
        availableClasses.forEach(className => {
            const checkboxDiv = document.createElement('div');
            checkboxDiv.className = 'form-check';
            checkboxDiv.innerHTML = `
                <input class="form-check-input" type="checkbox" value="${className}" id="class-${className}" checked>
                <label class="form-check-label" for="class-${className}">
                    ${className}
                </label>
            `;
            classFilterContainer.appendChild(checkboxDiv);
        });
        
        // Add change event listeners to checkboxes
        const checkboxes = classFilterContainer.querySelectorAll('input[type="checkbox"]');
        checkboxes.forEach(checkbox => {
            checkbox.addEventListener('change', applyClassFilter);
        });
    }
    
    // Function to apply class filter
    function applyClassFilter() {
        if (!lastInferenceData) return;
        
        const selectedClasses = getSelectedClasses();
        
        // Filter detections based on selected classes
        const filteredDetections = lastInferenceData.detections.filter(detection => 
            selectedClasses.length === 0 || selectedClasses.includes(detection.class)
        );
        
        // Re-request filtered inference image
        requestFilteredImage(selectedClasses);
        
        // Update detection results display
        displayDetectionResults(filteredDetections);
        
        // Update total objects count
        totalObjects.textContent = filteredDetections.length;
    }
    
    // Function to get selected classes from checkboxes
    function getSelectedClasses() {
        const checkboxes = classFilterContainer.querySelectorAll('input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(cb => cb.value);
    }
    
    // Function to request filtered image from server
    function requestFilteredImage(selectedClasses) {
        if (!testImage.files[0] || !lastInferenceData) return;
        
        const formData = new FormData();
        formData.append('image', testImage.files[0]);
        formData.append('model_id', modelSelect.value);
        formData.append('threshold', confidenceThreshold.value);
        formData.append('filter_classes', JSON.stringify(selectedClasses));
        
        // Send request for filtered image
        fetch('/api/test/inference', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (!data.error) {
                // Update only the image, keep existing detection data
                resultsImage.src = data.image_url + '?t=' + new Date().getTime();
            }
        })
        .catch(error => {
            console.error('Error filtering image:', error);
        });
    }
    
    // Function to display detection results
    function displayDetectionResults(detections) {
        // Add model type info
        const modelTypeLabel = lastInferenceData.model_info.type === 'rf-detr' ? 'RF-DETR' : 'YOLO';
        const modelVariant = lastInferenceData.model_info.variant || '';
        
        // Clear previous detection list
        detectionList.innerHTML = '';
        
        // Add model info at the top
        detectionList.innerHTML = `
            <div class="alert alert-info mb-3">
                <strong>Model:</strong> ${modelTypeLabel} ${modelVariant}
            </div>
        `;
        
        // Sort detections by confidence (highest first)
        const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);
        
        // Create detection list items
        if (sortedDetections.length === 0) {
            detectionList.innerHTML += '<p class="text-muted">No objects detected with current filters</p>';
        } else {
            // Count objects by class
            const classCounts = {};
            sortedDetections.forEach(detection => {
                if (classCounts[detection.class]) {
                    classCounts[detection.class]++;
                } else {
                    classCounts[detection.class] = 1;
                }
            });
            
            // Create class summary
            const classListHtml = Object.entries(classCounts)
                .map(([className, count]) => {
                    return `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span class="badge bg-primary">${className}</span>
                            <span>${count}</span>
                        </div>
                    `;
                })
                .join('');
            
            detectionList.innerHTML += classListHtml;
            
            // Add detailed list if needed
            if (sortedDetections.length > 0) {
                detectionList.innerHTML += `
                    <hr>
                    <h6>Details:</h6>
                    <div class="detection-details">
                        ${sortedDetections.map((detection, index) => `
                            <div class="detection-item">
                                <div class="d-flex justify-content-between">
                                    <span>#${index + 1} ${detection.class}</span>
                                    <span>${(detection.confidence * 100).toFixed(1)}%</span>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;
            }
        }
    }
});
