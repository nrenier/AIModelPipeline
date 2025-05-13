/**
 * Charts.js for training job metrics visualization
 */

// Function to create a line chart for training metrics
function createMetricsChart(canvasId, data, options = {}) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // Default configuration
    const defaultOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 0 // general animation time
        },
        hover: {
            animationDuration: 0 // duration of animations when hovering an item
        },
        responsiveAnimationDuration: 0, // animation duration after a resize
        elements: {
            line: {
                tension: 0.3 // smoother curves
            },
            point: {
                radius: 2,
                hitRadius: 10,
                hoverRadius: 5
            }
        },
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                mode: 'index',
                intersect: false,
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: '#ffffff',
                bodyColor: '#ffffff',
                borderColor: 'rgba(0, 0, 0, 0.2)',
                borderWidth: 1
            }
        },
        scales: {
            x: {
                display: true,
                title: {
                    display: true,
                    text: 'Epoch'
                },
                grid: {
                    display: false
                }
            },
            y: {
                display: true,
                title: {
                    display: true,
                    text: 'Value'
                },
                grid: {
                    color: 'rgba(200, 200, 200, 0.1)'
                }
            }
        }
    };

    // Merge options
    const chartOptions = { ...defaultOptions, ...options };

    // Create and return chart
    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: chartOptions
    });
}

// Function to update chart with new data
function updateMetricsChart(chart, newData) {
    chart.data = newData;
    chart.update();
}

// Function to create a dataset object for Chart.js
function createDataset(label, data, color) {
    return {
        label: label,
        data: data,
        backgroundColor: color,
        borderColor: color,
        fill: false,
        borderWidth: 2
    };
}

// Function to prepare data for metrics chart
function prepareChartData(metrics, epochKey = 'epoch') {
    // Extract epochs
    const epochs = metrics[epochKey] || [];

    // Define metrics to show and their colors
    const metricsConfig = {
        'loss': 'rgba(255, 99, 132, 1)',
        'precision': 'rgba(54, 162, 235, 1)',
        'recall': 'rgba(75, 192, 192, 1)',
        'mAP50': 'rgba(153, 102, 255, 1)',
        'mAP50-95': 'rgba(255, 159, 64, 1)'
    };

    // Prepare datasets
    const datasets = [];

    for (const [metric, color] of Object.entries(metricsConfig)) {
        if (metrics[metric]) {
            datasets.push(createDataset(metric, metrics[metric], color));
        }
    }

    return {
        labels: Array.from({length: epochs.length}, (_, i) => i + 1),
        datasets: datasets
    };
}

// Function to fetch metrics for a job and update the chart
function fetchAndUpdateMetrics(jobId, chart) {
    fetch(`/api/jobs/${jobId}/status`)
        .then(response => response.json())
        .then(data => {
            if (data.metrics) {
                // Prepare chart data
                const chartData = prepareChartData(data.metrics);

                // Update chart
                updateMetricsChart(chart, chartData);

                // Update progress
                const progressBar = document.getElementById('trainingProgress');
                if (progressBar && data.progress) {
                    progressBar.style.width = `${data.progress}%`;
                    progressBar.setAttribute('aria-valuenow', data.progress);

                    // Aggiorna il testo della barra di progresso
                    const progressText = document.getElementById('progressText');
                    if (progressText) {
                        progressText.textContent = `${Math.round(data.progress)}%`;
                    }
                }

                // Update status
                const statusBadge = document.getElementById('jobStatus');
                if (statusBadge) {
                    statusBadge.textContent = data.status.toUpperCase();

                    // Reset all classes
                    statusBadge.className = 'badge rounded-pill';

                    // Add color based on status
                    switch (data.status) {
                        case 'pending':
                            statusBadge.classList.add('bg-secondary');
                            break;
                        case 'running':
                            statusBadge.classList.add('bg-primary');
                            break;
                        case 'completed':
                            statusBadge.classList.add('bg-success');
                            break;
                        case 'failed':
                            statusBadge.classList.add('bg-danger');
                            break;
                        case 'cancelled':
                            statusBadge.classList.add('bg-warning');
                            break;
                        default:
                            statusBadge.classList.add('bg-secondary');
                    }
                }

                // Se il job è completato, mostra il messaggio e attiva il pulsante
                if (data.status === 'completed') {
                    // Show complete message and enable view results button
                    const completeMsg = document.getElementById('trainingCompleteMessage');
                    if (completeMsg) {
                        completeMsg.classList.remove('d-none');
                    }

                    const resultsBtn = document.getElementById('viewResultsBtn');
                    if (resultsBtn) {
                        resultsBtn.classList.remove('disabled');
                    }
                } else if (data.status === 'failed') {
                    // Show error message
                    const errorMsg = document.getElementById('trainingErrorMessage');
                    if (errorMsg) {
                        errorMsg.classList.remove('d-none');
                        if (data.error_message) {
                            const errorDetails = errorMsg.querySelector('.error-details');
                            if (errorDetails) {
                                errorDetails.textContent = data.error_message;
                            }
                        }
                    }
                }
            }
        })
        .catch(error => {
            console.error('Error fetching metrics:', error);
        });
}

// Initialize monitoring for a job
function initJobMonitoring(jobId) {
    const progressBar = document.getElementById('trainingProgress');
    const progressText = document.getElementById('progressText');
    const jobStatus = document.getElementById('jobStatus');
    const trainingCompleteMessage = document.getElementById('trainingCompleteMessage');
    const trainingErrorMessage = document.getElementById('trainingErrorMessage');
    const metricsChart = document.getElementById('metricsChart');

    // Store metrics history for the chart
    let metricsHistory = {
        epochs: [],
        precision: [],
        recall: [],
        mAP50: []
    };

    let chart = null;
    if (metricsChart) {
        chart = new Chart(metricsChart, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Precision',
                        borderColor: 'rgb(54, 162, 235)',
                        data: [],
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'Recall',
                        borderColor: 'rgb(255, 99, 132)',
                        data: [],
                        fill: false,
                        tension: 0.1
                    },
                    {
                        label: 'mAP50',
                        borderColor: 'rgb(75, 192, 192)',
                        data: [],
                        fill: false,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Epoch'
                        }
                    },
                    y: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Value'
                        },
                        min: 0,
                        max: 1
                    }
                }
            }
        });
    }

    function updateJobStatus() {
        fetch(`/api/jobs/${jobId}/status`)
            .then(response => response.json())
            .then(data => {
                console.log('Job status data:', data);

                // Update progress bar
                if (progressBar && progressText) {
                    const progress = Math.round(data.progress || 0);
                    progressBar.style.width = `${progress}%`;
                    progressBar.setAttribute('aria-valuenow', progress);
                    progressText.innerText = `${progress}%`;
                }

                // Update status badge
                if (jobStatus) {
                    const statusClass = getStatusClass(data.status);
                    // Remove all possible status classes
                    jobStatus.classList.remove('bg-success', 'bg-danger', 'bg-primary', 'bg-warning', 'bg-secondary');
                    // Add the current status class
                    jobStatus.classList.add(statusClass);
                    jobStatus.innerText = data.status.toUpperCase();
                }

                // Show completion or error message
                if (data.status === 'completed' && trainingCompleteMessage) {
                    trainingCompleteMessage.classList.remove('d-none');
                } else if (data.status === 'failed' && trainingErrorMessage) {
                    trainingErrorMessage.classList.remove('d-none');
                    const errorDetails = trainingErrorMessage.querySelector('.error-details');
                    if (errorDetails && data.error_message) {
                        errorDetails.innerText = data.error_message;
                    }
                }

                // Update metrics chart
                if (chart && data.metrics) {
                    // Check for epoch information
                    let currentEpoch = data.metrics.epoch;
                    if (currentEpoch !== undefined) {
                        // Only add new data points
                        if (!metricsHistory.epochs.includes(currentEpoch)) {
                            metricsHistory.epochs.push(currentEpoch);
                            metricsHistory.precision.push(data.metrics.precision || 0);
                            metricsHistory.recall.push(data.metrics.recall || 0);
                            metricsHistory.mAP50.push(data.metrics.mAP50 || 0);

                            // Simulate intermediate epochs if this is the first data point and epoch > 1
                            if (metricsHistory.epochs.length === 1 && currentEpoch > 1) {
                                const precision = data.metrics.precision || 0;
                                const recall = data.metrics.recall || 0;
                                const mAP50 = data.metrics.mAP50 || 0;

                                // Insert previous epochs with interpolated values
                                for (let i = 1; i < currentEpoch; i++) {
                                    const progress = i / currentEpoch;
                                    metricsHistory.epochs.unshift(i);
                                    metricsHistory.precision.unshift(precision * progress);
                                    metricsHistory.recall.unshift(recall * progress);
                                    metricsHistory.mAP50.unshift(mAP50 * progress);
                                }
                            }

                            // Update chart with all history
                            chart.data.labels = [...metricsHistory.epochs];
                            chart.data.datasets[0].data = [...metricsHistory.precision];
                            chart.data.datasets[1].data = [...metricsHistory.recall];
                            chart.data.datasets[2].data = [...metricsHistory.mAP50];
                            chart.update();
                        }
                    }
                    // If we don't have epoch info but have other metrics, update single points
                    else if ('precision' in data.metrics || 'recall' in data.metrics || 'mAP50' in data.metrics) {
                        const newEpoch = chart.data.labels.length + 1;
                        chart.data.labels.push(newEpoch);
                        chart.data.datasets[0].data.push(data.metrics.precision || 0);
                        chart.data.datasets[1].data.push(data.metrics.recall || 0);
                        chart.data.datasets[2].data.push(data.metrics.mAP50 || 0);
                        chart.update();
                    }
                }

                // Continue polling if job is still running
                if (data.status === 'running' || data.status === 'pending') {
                    setTimeout(updateJobStatus, 3000); // Poll every 3 seconds
                }
            })
            .catch(error => {
                console.error('Error fetching job status:', error);
                // Retry after a delay in case of error
                setTimeout(updateJobStatus, 10000);
            });
    }

    function getStatusClass(status) {
        switch (status) {
            case 'completed': return 'bg-success';
            case 'failed': return 'bg-danger';
            case 'running': return 'bg-primary';
            case 'cancelled': return 'bg-warning';
            default: return 'bg-secondary';
        }
    }

    // Start monitoring
    updateJobStatus();
}