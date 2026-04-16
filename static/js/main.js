document.addEventListener('DOMContentLoaded', () => {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const uploadPanel = document.getElementById('uploadPanel');
    const resultsPanel = document.getElementById('resultsPanel');
    const resetBtn = document.getElementById('resetBtn');
    const loadingState = document.getElementById('loadingState');

    // Elements to populate
    const resultImage = document.getElementById('resultImage');
    const originalImage = document.getElementById('originalImage');
    const labelsImage = document.getElementById('labelsImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const sceneClass = document.getElementById('sceneClass');
    const cleanlinessLevel = document.getElementById('cleanlinessLevel');
    const scoreValue = document.getElementById('scoreValue');
    const scoreCircle = document.getElementById('scoreCircle');
    const recommendationText = document.getElementById('recommendationText');
    const rawCount = document.getElementById('rawCount');
    const contextScore = document.getElementById('contextScore');
    const semanticScore = document.getElementById('semanticScore');
    const hotspotContainer = document.getElementById('hotspotContainer');
    const noHotspotsMsg = document.getElementById('noHotspotsMsg');

    // Tab switching removed (Dual view enabled)

    // Reset View
    resetBtn.addEventListener('click', () => {
        resultsPanel.classList.add('hidden');
        uploadPanel.style.display = 'block';
        fileInput.value = '';
    });

    // Drag and Drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, () => {
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    fileInput.addEventListener('change', function () {
        if (this.files.length > 0) {
            handleFile(this.files[0]);
        }
    });

    // Simulated Loading Steps Animation
    let stepInterval;
    function startLoadingAnimation() {
        loadingState.classList.add('active');
        const steps = document.querySelectorAll('.loading-steps .step');
        let currentStep = 0;

        // Reset steps
        steps.forEach(s => {
            s.classList.remove('active', 'done');
        });
        steps[0].classList.add('active');

        stepInterval = setInterval(() => {
            if (currentStep < steps.length - 1) {
                steps[currentStep].classList.remove('active');
                steps[currentStep].classList.add('done');
                currentStep++;
                steps[currentStep].classList.add('active');
            }
        }, 800);
    }

    function stopLoadingAnimation() {
        clearInterval(stepInterval);
        loadingState.classList.remove('active');
    }

    // Colors mapping
    const getLevelColor = (level) => {
        const colors = {
            'Excellent': '#10b981', // emerald
            'Good': '#84cc16',      // lime
            'Average': '#f59e0b',   // amber
            'Poor': '#f97316',      // orange
            'Critical': '#ef4444',  // red
        };
        return colors[level] || '#94a3b8';
    };

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file (JPG/PNG).');
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        startLoadingAnimation();

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                stopLoadingAnimation();

                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }

                populateResults(data);

                // Switch views
                uploadPanel.style.display = 'none';
                resultsPanel.classList.remove('hidden');
            })
            .catch(error => {
                stopLoadingAnimation();
                console.error('Error:', error);
                alert('An error occurred during analysis.');
            });
    }

    function populateResults(data) {
        // Images
        resultImage.src = data.image_data;
        originalImage.src = data.original_data;
        labelsImage.src = data.labels_data;
        heatmapImage.src = data.heatmap_data;

        const res = data.results;

        // Header Values
        sceneClass.textContent = res.scene_class.charAt(0).toUpperCase() + res.scene_class.slice(1);
        cleanlinessLevel.textContent = res.cleanliness_level;

        // Configure Level Badge Color
        const color = getLevelColor(res.cleanliness_level);
        cleanlinessLevel.style.color = color;
        cleanlinessLevel.style.backgroundColor = color + '20';
        cleanlinessLevel.style.borderColor = color + '50';

        // Dial Animation
        scoreValue.textContent = res.final_cleanliness_score.toFixed(1);
        const percentage = (res.final_cleanliness_score / 5.0) * 100;
        scoreCircle.setAttribute('stroke-dasharray', `${percentage}, 100`);
        scoreCircle.style.stroke = color;

        recommendationText.innerHTML = `<strong>Recommendation:</strong> ${res.recommendation}`;

        // Sub Metrics
        rawCount.textContent = res.litter_count;
        contextScore.textContent = res.context_aware_score.toFixed(1);
        semanticScore.textContent = res.weighted_semantic_score.toFixed(1);

        // Hotspots
        hotspotContainer.innerHTML = '';
        if (res.hotspots && res.hotspots.length > 0) {
            noHotspotsMsg.classList.add('hidden');
            res.hotspots.forEach((hs, idx) => {
                const div = document.createElement('div');
                div.className = `hotspot-item severity-${hs.severity}`;
                div.innerHTML = `
                    <div>
                        <strong>Cluster ${idx + 1}</strong>
                        <div class="stat-sub" style="margin-top:2px;">${hs.litter_count} items detected</div>
                    </div>
                    <div style="font-weight:600; font-size:0.85rem;">Severity: ${hs.severity}</div>
                `;
                hotspotContainer.appendChild(div);
            });
        } else {
            noHotspotsMsg.classList.remove('hidden');
        }
    }
});
