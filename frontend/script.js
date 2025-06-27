const fileInput = document.getElementById('csvFile');
const fileName = document.getElementById('fileName');
const processButton = document.getElementById('processButton');
const serverUrlInput = document.getElementById('serverUrl');
const results = document.getElementById('results');
let fileProcessed = false; // Track if file is successfully processed
let currentSessionId = null;
let currentPage = 1;
let totalPages = 1;
let totalRecords = 0;
const plotSelector = document.getElementById('plot-selector');
const dropdownContent = document.getElementById('dropdownContent');
const selectedText = document.querySelector('.selected-text');
const edaPlotsDiv = document.getElementById('eda-plots');
let edaLoaded = false;
let selectedIndex = 0; // Keep track of selected index
const graphDescription = document.getElementById('description');
let elbowLoaded = false;
let elbowData = null;
// Array to store the selected event values
let selectedEvents = [];
const canvas = document.getElementById('pieChart');
const ctx = canvas.getContext('2d');
const legendDiv = document.getElementById('chartLegend');


// Get references to DOM elements
const selectedEventsContainer = document.getElementById('selected-events-container');
const noEventsMessage = document.getElementById('no-events-message');
const clearAllButton = document.getElementById('clear-all-button');
const predictButton = document.getElementById('predict-button');

// File input handling
fileInput.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (file) {
        fileName.textContent = `Selected: ${file.name}`;
        updateProcessButton();
    } else {
        fileName.textContent = 'No file selected';
        updateProcessButton();
    }
});

// Server URL input handling
serverUrlInput.addEventListener('input', updateProcessButton);

function updateProcessButton() {
    const hasFile = fileInput.files.length > 0;
    const hasServerUrl = serverUrlInput.value.trim() !== '';
    processButton.disabled = !(hasFile && hasServerUrl);
}

// Function to enable tabs and Go to EDA button
function enableTabs() {
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('disabled');
    });
}

// Function to render the table from CSV data
function displayResultTable(data) {
    document.querySelector("#placeholder-data").style.display = "none";
    const tableContainer = document.createElement('div');
    tableContainer.className = 'result-table-container';

    const table = document.createElement('table');
    table.className = 'result-table';

    // Define columns with natural names (mapped to CSV keys)
    const columns = [
        { key: 'User_ID', label: 'User' },
        { key: 'Product_ID', label: 'Product' },
        { key: 'Category_ID', label: 'Category' },
        { key: 'Behavior', label: 'Action' },
        { key: 'Datetime', label: 'Date & Time' },
        { key: 'Day_of_Week', label: 'Day' },
        { key: 'Hour_of_Day', label: 'Hour' },
        { key: 'Date', label: 'Date Only' }
    ];

    // Create table header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(col => {
        const th = document.createElement('th');
        th.textContent = col.label;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);

    // Create table body
    const tbody = document.createElement('tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(col => {
            const td = document.createElement('td');
            td.textContent = row[col.key] || ''; // Handle missing or null values
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);

    tableContainer.appendChild(table);

    // Clear previous table (if any) and append new one
    const existingTable = results.querySelector('.result-table-container');
    if (existingTable) {
        existingTable.remove();
    }
    var firstChild = results.firstChild;
    if (firstChild) {
        results.insertBefore(tableContainer, firstChild);
    } else {
        results.appendChild(tableContainer);
    }

}

// Function to show temporary popup notification
function showPopup(message, type, duration = 3000) {
    const popup = document.createElement('div');
    popup.className = `popup-notification ${type}`;
    popup.textContent = message;
    document.body.appendChild(popup);

    // Remove after 3 seconds
    setTimeout(() => {
        popup.remove();
    }, duration);
}

// Process button click handler
processButton.addEventListener('click', async function () {
    const file = fileInput.files[0];
    const serverUrl = serverUrlInput.value.trim().replace(/\/$/, ''); // Remove trailing slash

    if (!file || !serverUrl) {
        showPopup('Please select a file and enter server URL', 'error');
        return;
    }

    // Show loading popup
    showPopup('Processing file...', 'loading');
    processButton.disabled = true;

    // Try fetch first, then XMLHttpRequest as fallback
    try {
        await tryFetchMethod(file, serverUrl);
    } catch (fetchError) {
        console.log('Fetch method failed, trying XMLHttpRequest...', fetchError);
        try {
            await tryXHRMethod(file, serverUrl);
        } catch (xhrError) {
            console.error('Both methods failed:', { fetchError, xhrError });
            showPopup(`Both methods failed. Fetch: ${fetchError.message}, XHR: ${xhrError.message}`, 'error');
        }
    } finally {
        updateProcessButton();
    }
});

async function tryFetchMethod(file, serverUrl) {
    // Create FormData
    const formData = new FormData();
    formData.append('file', file);

    console.log(`Making fetch request to: ${serverUrl}/upload_and_process`);

    // Make the request to your Flask server
    const response = await fetch(`${serverUrl}/upload_and_process`, {
        method: 'POST',
        body: formData,
        mode: 'cors'
    });

    console.log('Response status:', response.status);
    console.log('Response headers:', [...response.headers.entries()]);

    if (response.ok) {
        // Get the response as text (CSV content)
        try {
            const csvText = await response.text();
            console.log('CSV text length:', csvText.length);

            if (csvText.length === 0) {
                throw new Error('Empty CSV response');
            }

            // Parse CSV to JSON using Papa Parse
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    const data = results.data;
                    if (data.length === 0) {
                        showPopup('Parsed CSV is empty', 'error');
                        return;
                    }

                    // Display table
                    displayResultTable(data);
                    // Extract pagination info from headers
                    const sessionId = response.headers.get('X-Session-ID');
                    const totalPgs = parseInt(response.headers.get('X-Total-Pages'));
                    const currentPg = parseInt(response.headers.get('X-Current-Page'));
                    const totalRecs = parseInt(response.headers.get('X-Total-Records'));

                    if (sessionId && totalPgs > 1) {
                        showPagination(sessionId, currentPg, totalPgs, totalRecs);
                    }
                    fileProcessed = true; // Mark as processed
                    enableTabs(); // Enable other tabs
                    let edaLoaded = false;
                    showPopup('✅ File processed successfully!', 'success');


                },
                error: (err) => {
                    showPopup(`CSV parsing error: ${err.message}`, 'error');
                }
            });

        } catch (textError) {
            console.error('Error processing CSV response:', textError);
            throw textError;
        }
    } else {
        // Handle HTTP error
        const errorText = await response.text();
        console.error('Error response:', errorText);

        let errorMessage;
        try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.error || 'Unknown error';
        } catch (e) {
            errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        }

        throw new Error(errorMessage);
    }
}

function tryXHRMethod(file, serverUrl) {
    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();
        const formData = new FormData();
        formData.append('file', file);

        xhr.open('POST', `${serverUrl}/upload_and_process`, true);
        xhr.responseType = 'blob';

        xhr.onload = function () {
            if (xhr.status === 200) {
                console.log('XHR success, blob size:', xhr.response.size);

                if (xhr.response.size === 0) {
                    showPopup('Empty response from XHR', 'error');
                    reject(new Error('Empty response from XHR'));
                    return;
                }

                // Convert blob to text for CSV parsing
                const reader = new FileReader();
                reader.onload = function () {
                    const csvText = reader.result;
                    Papa.parse(csvText, {
                        header: true,
                        skipEmptyLines: true,
                        complete: (results) => {
                            const data = results.data;
                            if (data.length === 0) {
                                showPopup('Parsed CSV is empty', 'error');
                                reject(new Error('Parsed CSV is empty'));
                                return;
                            }

                            // Display table
                            displayResultTable(data);
                            fileProcessed = true; // Mark as processed
                            enableTabs(); // Enable other tabs
                            let edaLoaded = false;
                            showPopup('✅ File processed successfully!', 'success');
                            resolve();
                        },
                        error: (err) => {
                            showPopup(`CSV parsing error: ${err.message}`, 'error');
                            reject(new Error(`CSV parsing error: ${err.message}`));
                        }
                    });
                };
                reader.onerror = function () {
                    showPopup('Error reading XHR blob', 'error');
                    reject(new Error('Error reading XHR blob'));
                };
                reader.readAsText(xhr.response);
            } else {
                showPopup(`XHR failed: ${xhr.status} ${xhr.statusText}`, 'error');
                reject(new Error(`XHR failed: ${xhr.status} ${xhr.statusText}`));
            }
        };

        xhr.onerror = function () {
            showPopup('XHR network error', 'error');
            reject(new Error('XHR network error'));
        };

        xhr.send(formData);
    });
}

// Drag and drop functionality
const fileInputButton = document.querySelector('.file-input-button');

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    fileInputButton.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    fileInputButton.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    fileInputButton.addEventListener(eventName, unhighlight, false);
});

function highlight(e) {
    fileInputButton.style.borderColor = '#5a67d8';
    fileInputButton.style.background = '#3c3c3c';
}

function unhighlight(e) {
    fileInputButton.style.borderColor = '#667eea';
    fileInputButton.style.background = '#333333';
}

fileInputButton.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;

    if (files.length > 0 && files[0].name.toLowerCase().endsWith('.csv')) {
        fileInput.files = files;
        fileName.textContent = `Selected: ${files[0].name}`;
        updateProcessButton();
    }
}

// Initialize
updateProcessButton();

function showPagination(sessionId, page, totalPgs, totalRecs) {
    currentSessionId = sessionId;
    currentPage = page;
    totalPages = totalPgs;
    totalRecords = totalRecs;

    document.getElementById('paginationSection').style.display = 'block';
    updatePaginationUI();
}

function updatePaginationUI() {
    document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
    document.getElementById('pageNumberInput').max = totalPages;

    // Enable/disable buttons
    document.getElementById('firstPageBtn').disabled = currentPage === 1;
    document.getElementById('prevPageBtn').disabled = currentPage === 1;
    document.getElementById('nextPageBtn').disabled = currentPage === totalPages;
    document.getElementById('lastPageBtn').disabled = currentPage === totalPages;
}

// Pagination event listeners
document.getElementById('firstPageBtn').addEventListener('click', () => navigateToPage('first'));
document.getElementById('prevPageBtn').addEventListener('click', () => navigateToPage('previous'));
document.getElementById('nextPageBtn').addEventListener('click', () => navigateToPage('next'));
document.getElementById('lastPageBtn').addEventListener('click', () => navigateToPage('last'));
document.getElementById('goToPageBtn').addEventListener('click', () => {
    const pageNum = parseInt(document.getElementById('pageNumberInput').value);
    if (pageNum >= 1 && pageNum <= totalPages) {
        navigateToPage('goto', pageNum);
    }
});

async function navigateToPage(action, pageNum = null) {
    if (!currentSessionId) return;

    const serverUrl = serverUrlInput.value.trim().replace(/\/$/, '');
    let url;

    switch (action) {
        case 'first':
            url = `${serverUrl}/first_page/${currentSessionId}`;
            break;
        case 'last':
            url = `${serverUrl}/last_page/${currentSessionId}`;
            break;
        case 'next':
            url = `${serverUrl}/next_page/${currentSessionId}/${currentPage}`;
            break;
        case 'previous':
            url = `${serverUrl}/previous_page/${currentSessionId}/${currentPage}`;
            break;
        case 'goto':
            url = `${serverUrl}/go_to_page/${currentSessionId}/${pageNum}`;
            break;
    }

    try {
        const response = await fetch(url);
        if (response.ok) {
            const csvText = await response.text();
            const newPage = response.headers.get('X-Current-Page');
            const newTotalPages = response.headers.get('X-Total-Pages');

            currentPage = parseInt(newPage);
            totalPages = parseInt(newTotalPages);

            // Parse and display new data
            Papa.parse(csvText, {
                header: true,
                skipEmptyLines: true,
                complete: (results) => {
                    displayResultTable(results.data);
                    updatePaginationUI();
                }
            });
        }
    } catch (error) {
        showPopup(`Navigation error: ${error.message}`, 'error');
    }
}

// ...existing code...

//let sessionId = null; // Store sessionId after upload

// Example: After successful upload, set sessionId = response.session_id

// Function to handle option selection
function selectOption(optionText, index) {
    selectedText.textContent = optionText;
    selectedIndex = index;
    closeDropdown();
    showImage(index);
    graphDescription.textContent = data.plot_descriptions[index];
}

// Toggle dropdown function
function toggleDropdown() {
    const dropdown = document.getElementById('dropdownContent');
    const button = document.querySelector('.dropdown-button');

    dropdown.classList.toggle('show');
    button.classList.toggle('active');
}

// Close dropdown function
function closeDropdown() {
    const dropdown = document.getElementById('dropdownContent');
    const button = document.querySelector('.dropdown-button');

    dropdown.classList.remove('show');
    button.classList.remove('active');
}

// Your existing showImage function (unchanged)
const showImage = (index) => {
    edaPlotsDiv.innerHTML = ''; // Clear previous content
    const base64Image = data.plots[index][1];
    const img = document.createElement('img');
    img.src = `data:image/png;base64,${base64Image}`;
    img.alt = `EDA Plot ${index + 1}`;
    img.classList.add('eda-image-default');
    img.addEventListener('click', () => {
        img.classList.toggle('eda-image-default');
        img.classList.toggle('eda-image-enlarged');
    });
    edaPlotsDiv.appendChild(img);
};


// Close dropdown when clicking outside
window.addEventListener('click', function (event) {
    if (!event.target.matches('.dropdown-button') &&
        !event.target.closest('.dropdown-container')) {
        closeDropdown();
    }
});

// Handle keyboard navigation
document.addEventListener('keydown', function (event) {
    const dropdown = document.getElementById('dropdownContent');

    if (event.key === 'Escape' && dropdown.classList.contains('show')) {
        closeDropdown();
        plotSelector.focus();
    }
});

document.querySelector('[data-tab="eda"]').addEventListener('click', async function () {
    if (!currentSessionId) {
        showPopup('Please process a file first to perform EDA.', 'error');
        return;
    }
    if (edaLoaded) {
        return;
    }
    showPopup('Performing exploratory data analysis ...', 'loading', 7000);
    const serverUrl = document.getElementById('serverUrl').value;
    const edaTab = document.getElementById('eda');

    const edaPlotsDiv = document.getElementById('eda-plots');
    const plotSelector = document.getElementById('plot-selector');
    //edaTab.innerHTML = '<div class="container"><p>Loading EDA results...</p></div>';
    try {
        const response = await fetch(`${serverUrl}/eda/${currentSessionId}`);
        if (!response.ok) throw new Error('Failed to fetch EDA results');
        // const html = await response.text();
        // edaTab.innerHTML = html;
        data = await response.json();
        edaLoaded = true;
        //edaResultsDiv.innerHTML = data.text_output;
        if (data.plots && data.plots.length > 0) {
            data.plots.forEach((plotArray, index) => {
                const graphName = plotArray[0];
                const dropdownItem = document.createElement('div');
                dropdownItem.className = 'dropdown-item';
                dropdownItem.textContent = graphName;
                dropdownItem.onclick = () => selectOption(graphName, index);
                dropdownContent.appendChild(dropdownItem);
            });

            if (data.plots && data.plots.length > 0) {
                selectedText.textContent = data.plots[0][0];
                showImage(0);
                graphDescription.textContent = data.plot_descriptions[0];
            }

            // Update image on dropdown change
            plotSelector.addEventListener('change', (e) => {
                const selectedIndex = parseInt(e.target.value);
                showImage(selectedIndex);
            });

        } else {
            edaPlotsDiv.innerHTML = '<p>No plots were generated for this session.</p>';
        }


    } catch (err) {
        edaTab.innerHTML = `<div class="container"><p>Error loading EDA: ${err.message}</p></div>`;
    }
});


// Function to display RFM clustering results
function displayRFMResults(data) {
    const rfmTab = document.getElementById('rfm');

    // Create the RFM results HTML structure
    const rfmHTML = `
        <div class="container">
            <h3>RFM Clustering Analysis</h3>
            
            <!-- K-Value Selection -->
            <div class="rfm-controls">
                <label for="kValueSelect">Number of Clusters (K):</label>
                <select id="kValueSelect" class="k-value-select">
                    <option value="2">2</option>
                    <option value="3" selected>3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                    <option value="6">6</option>
                    <option value="7">7</option>
                    <option value="8">8</option>
                    <option value="9">9</option>
                    <option value="10">10</option>
                </select>
                <button id="updateClustering" class="update-btn">Update Clustering</button>
            </div>
            
            <!-- Clustering Summary -->
            <div class="clustering-summary">
                <h4>Clustering Summary</h4>
                <div class="summary-stats">
                    <div class="stat-item">
                        <span class="stat-label">Total Customers:</span>
                        <span class="stat-value" id="totalCustomers">${data.clustering_summary.total_customers}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Number of Clusters:</span>
                        <span class="stat-value" id="numClusters">${data.clustering_summary.num_clusters}</span>
                    </div>
                </div>
            </div>
            
            <!-- Pie Chart -->
            <div class="rfm-visualization">
                <h4>Customer Segments Distribution</h4>
                <div class="chart-container">
                    <img id="rfmPieChart" src="data:image/png;base64,${data.pie_chart}" alt="RFM Clustering Pie Chart" class="rfm-chart">
                </div>
            </div>
            
            <!-- Cluster Means Table -->
            <div class="cluster-analysis">
                <h4>Cluster Characteristics</h4>
                <div class="table-container">
                    <table class="cluster-table" id="clusterMeansTable">
                        <thead>
                            <tr>
                                <th>Cluster</th>
                                <th>Avg Recency</th>
                                <th>Avg Frequency</th>
                                <th>Avg Monetary</th>
                                <th>Customer Count</th>
                            </tr>
                        </thead>
                        <tbody id="clusterMeansBody">
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Customer Classification -->
            <div class="customer-classification">
                <h4>Customer Segments</h4>
                <div class="table-container">
                    <table class="classification-table" id="customerClassTable">
                        <thead>
                            <tr>
                                <th>Cluster</th>
                                <th>Segment Name</th>
                                <th>Description</th>
                                <th>Customer Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody id="customerClassBody">
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    rfmTab.innerHTML = rfmHTML;

    // Populate the tables
    populateClusterMeansTable(data.cluster_means, data.clustering_summary.cluster_distribution);
    populateCustomerClassTable(data.customer_class, data.clustering_summary);


    // Add click event to enlarge chart
    document.getElementById('rfmPieChart').addEventListener('click', function () {
        this.classList.toggle('enlarged-chart');
    });
}

// Function to populate cluster means table
function populateClusterMeansTable(clusterMeans, clusterDistribution) {
    const tbody = document.getElementById('clusterMeansBody');
    tbody.innerHTML = '';

    clusterMeans.forEach((cluster, index) => {
        const row = document.createElement('tr');
        const customerCount = clusterDistribution[index] || 0;

        row.innerHTML = `
            <td>Cluster ${index}</td>
            <td>${cluster.Recency?.toFixed(2) || 'N/A'}</td>
            <td>${cluster.Frequency?.toFixed(2) || 'N/A'}</td>
            <td>${cluster.Monetary?.toFixed(2) || 'N/A'}</td>
            <td>${customerCount}</td>
        `;
        tbody.appendChild(row);
    });
}

// Function to populate customer classification table
function populateCustomerClassTable(customerClass, clusteringSummary) {
    const tbody = document.getElementById('customerClassBody');
    tbody.innerHTML = '';

    customerClass.forEach(segment => {
        const row = document.createElement('tr');
        const percentage = ((segment.customer_count / clusteringSummary.total_customers) * 100).toFixed(1);

        row.innerHTML = `
            <td>Cluster ${segment.cluster}</td>
            <td>${segment.segment_name}</td>
            <td>${segment.description}</td>
            <td>${segment.customer_count}</td>
            <td>${percentage}%</td>
        `;
        tbody.appendChild(row);
    });
}

// Function to get selected K value
function getSelectedKValue() {
    const select = document.getElementById('kValueSelect');
    return select ? parseInt(select.value) : null;
}

// Function to update RFM display with new data
function updateRFMDisplay(data) {
    // Update summary stats
    document.getElementById('totalCustomers').textContent = data.clustering_summary.total_customers;
    document.getElementById('numClusters').textContent = data.clustering_summary.num_clusters;

    // Update pie chart
    document.getElementById('rfmPieChart').src = `data:image/png;base64,${data.pie_chart}`;

    // Update tables
    populateClusterMeansTable(data.cluster_means, data.clustering_summary.cluster_distribution);
    populateCustomerClassTable(data.customer_class, data.clustering_summary);
}

// Reset RFM loaded state when processing new file
const originalProcessButtonHandler = processButton.onclick;
processButton.addEventListener('click', function () {
    rfmLoaded = false;
    rfmData = null;
});


// RFM Elbow Tab Event Listener - Add this to your existing code
document.querySelector('[data-tab="rfm"]').addEventListener('click', async function () {
    if (!currentSessionId) {
        showPopup('Please process a file first to perform RFM analysis.', 'error');
        return;
    }

    if (elbowLoaded) {
        return;
    }

    showPopup('Performing RFM analysis and elbow method...', 'loading', 3000);
    const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');
    const rfmTab = document.getElementById('rfm');

    try {
        const response = await fetch(`${serverUrl}/rfm_elbow/${currentSessionId}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to fetch RFM elbow analysis');
        }

        elbowData = await response.json();
        elbowLoaded = true;

        // Display RFM elbow analysis results
        displayElbowResults(elbowData);

        showPopup('✅ RFM elbow analysis completed successfully!', 'success');

    } catch (err) {
        console.error('RFM elbow analysis error:', err);
        rfmTab.innerHTML = `<div class="container"><p class="error-message">Error loading RFM analysis: ${err.message}</p></div>`;
        showPopup(`RFM elbow analysis failed: ${err.message}`, 'error');
    }
});

// Function to display RFM elbow analysis results
function displayElbowResults(data) {
    const rfmTab = document.getElementById('rfm');

    // Create the RFM elbow analysis HTML structure
    const elbowHTML = `
        <div class="container">
            <h2>RFM and Clustering Analysis</h2>
            
            <!-- RFM Summary -->
            <div class="rfm-summary-section">
                <h3>RFM Summary</h3>
                <div class="summary-stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Customers</div>
                        <div class="stat-value">${data.rfm_summary.total_customers}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average Recency</div>
                        <div class="stat-value">${data.rfm_summary.avg_recency.toFixed(2)} days</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Average Frequency</div>
                        <div class="stat-value">${data.rfm_summary.avg_frequency.toFixed(2)}</div>
                    </div>
                </div>
                
                <div class="range-stats">
                    <div class="range-item">
                        <span class="range-label">Recency Range:</span>
                        <span class="range-value">${data.rfm_summary.recency_range[0].toFixed(0)} - ${data.rfm_summary.recency_range[1].toFixed(0)} days</span>
                    </div>
                    <div class="range-item">
                        <span class="range-label">Frequency Range:</span>
                        <span class="range-value">${data.rfm_summary.frequency_range[0].toFixed(0)} - ${data.rfm_summary.frequency_range[1].toFixed(0)} purchases</span>
                    </div>
                </div>
            </div>
            
            <!-- Elbow Method -->
            <div class="elbow-analysis-section">
                <h3>Elbow Method for Optimal K</h3>
                <div class="elbow-description">
                    <p>The elbow method helps determine the optimal number of clusters (K) for K-means clustering. 
                    Look for the "elbow" point where the rate of decrease in WCSS (Within-Cluster Sum of Squares) 
                    starts to level off.</p>
                </div>
                
                <div class="elbow-chart-container">
                    <img id="elbowChart" 
                         src="data:image/png;base64,${data.elbow_plot}" 
                         alt="Elbow Method Chart" 
                         class="elbow-chart">
                </div>
                
            
            <!-- Clustering Action -->
            <div class="clustering-action-section">
                <h3>Proceed to Clustering</h3>
                <div class="clustering-controls">
                    <label for="selectedK">Select Number of Clusters (K):</label>
                    <select id="selectedK" class="k-select">
                        <option value="2">2</option>
                        <option value="3" selected>3</option>
                        <option value="4">4</option>
                        <option value="5">5</option>
                        <option value="6">6</option>
                        <option value="7">7</option>
                        <option value="8">8</option>
                        <option value="9">9</option>
                        <option value="10">10</option>
                    </select>
                    <button id="performClustering" class="clustering-btn">Perform K-Means Clustering</button>
                </div>
                <div class="clustering-note">
                    <p><strong>Tip:</strong> Based on the elbow chart above, choose the K value where the curve starts to flatten out (the "elbow" point).</p>
                </div>
            </div>
        </div>
    `;

    rfmTab.innerHTML = elbowHTML;

    // Populate WCSS table
    populateWCSSTable(data.wcss_values);

    // Add event listeners
    addElbowEventListeners();
}

// Function to populate WCSS values table
function populateWCSSTable(wcssValues) {

}

// Function to add event listeners for elbow analysis
function addElbowEventListeners() {
    // Add click event to enlarge elbow chart
    document.getElementById('elbowChart').addEventListener('click', function () {
        this.classList.toggle('enlarged-chart');
    });

    // Add event listener for clustering button
    document.getElementById('performClustering').addEventListener('click', performKMeansClustering);
}

// Function to perform K-means clustering (calls your existing RFM clustering API)
async function performKMeansClustering() {
    const kValue = parseInt(document.getElementById('selectedK').value);
    if (!kValue || !currentSessionId) {
        showPopup('Invalid K value or session', 'error');
        return;
    }

    showPopup(`Performing K-means clustering with K=${kValue}...`, 'loading', 2000);
    const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');

    try {
        const response = await fetch(`${serverUrl}/rfm_clustering/${currentSessionId}/${kValue}`);
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to perform clustering');
        }

        const clusteringData = await response.json();

        // Display clustering results alongside elbow analysis
        displayClusteringResults(clusteringData, kValue);

        showPopup('✅ K-means clustering completed successfully!', 'success');
        document.getElementById('performClustering').disabled = true;
        document.getElementById('performClustering').classList.add('disabled-button');
    } catch (err) {
        console.error('K-means clustering error:', err);
        showPopup(`K-means clustering failed: ${err.message}`, 'error');
    }
}

// Function to display clustering results (appends to existing elbow analysis)
function displayClusteringResults(data, kValue) {
    const rfmTab = document.getElementById('rfm');

    // Create clustering results HTML
    const clusteringHTML = `
        <div class="clustering-results-section">
            <h3>K-Means Clustering Results (K=${kValue})</h3>
            
            <!-- Clustering Summary -->
            <div class="clustering-summary">
                <h4>Clustering Summary</h4>
                <div class="summary-stats-grid">
                    <div class="stat-card">
                        <div class="stat-label">Total Customers</div>
                        <div class="stat-value">${data.clustering_summary.total_customers}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Number of Clusters</div>
                        <div class="stat-value">${data.clustering_summary.num_clusters}</div>
                    </div>
                </div>
            </div>
            
            <!-- Pie Chart -->
            <div class="pie-chart-section">
                <h4>Customer Segments Distribution</h4>
                <div class="chart-container">
                    <img id="clusteringPieChart" 
                         src="data:image/png;base64,${data.pie_chart}" 
                         alt="Clustering Pie Chart" 
                         class="clustering-chart">
                </div>
            </div>
            
            <!-- Cluster Characteristics -->
            <div class="cluster-characteristics">
                <h4>Cluster Characteristics</h4>
                <div class="table-container">
                    <table class="cluster-table">
                        <thead>
                            <tr>
                                <th>Cluster</th>
                                <th>Avg Recency</th>
                                <th>Avg Frequency</th>
                                <th>Avg Monetary</th>
                                <th>Customer Count</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody id="clusterCharacteristicsBody">
                        </tbody>
                    </table>
                </div>
            </div>

        </div>
    `;

    // Append clustering results to the container
    const container = rfmTab.querySelector('.container');
    container.insertAdjacentHTML('beforeend', clusteringHTML);

    // Populate tables
    populateClusterCharacteristics(data.cluster_means, data.clustering_summary);


    // Add event listeners for new elements
    document.getElementById('clusteringPieChart').addEventListener('click', function () {
        this.classList.toggle('enlarged-chart');
    });

}

// Function to populate cluster characteristics table
function populateClusterCharacteristics(clusterMeans, clusteringSummary) {
    const tbody = document.getElementById('clusterCharacteristicsBody');
    tbody.innerHTML = '';

    clusterMeans.forEach((cluster, index) => {
        const row = document.createElement('tr');
        const customerCount = clusteringSummary.cluster_distribution[index] || 0;
        const percentage = ((customerCount / clusteringSummary.total_customers) * 100).toFixed(1);

        row.innerHTML = `
            <td>Cluster ${index}</td>
            <td>${cluster.Recency?.toFixed(1) || 'N/A'} days</td>
            <td>${cluster.Frequency?.toFixed(1) || 'N/A'}</td>
            <td>${cluster.Monetary?.toFixed(2) || 'N/A'}</td>
            <td>${customerCount}</td>
            <td>${percentage}%</td>
        `;
        tbody.appendChild(row);
    });
}



// Helper function to generate business actions based on segment names
function generateBusinessAction(segmentName) {
    const actions = {
        'Champions': 'Reward & retain with exclusive offers',
        'Loyal Customers': 'Maintain engagement with personalized content',
        'Potential Loyalists': 'Nurture with targeted campaigns',
        'New Customers': 'Welcome series and onboarding',
        'Promising': 'Encourage repeat purchases',
        'Need Attention': 'Re-engagement campaigns',
        'About to Sleep': 'Win-back offers and reminders',
        'At Risk': 'Urgent retention efforts',
        'Cannot Lose Them': 'High-touch personal outreach',
        'Hibernating': 'Reactivation campaigns',
        'Lost': 'Final win-back attempts'
    };

    return actions[segmentName] || 'Analyze further for targeted strategy';
}

// Reset elbow loaded state when processing new file
// Add this to your existing processButton event listener
const originalProcessHandler = processButton.onclick;
processButton.addEventListener('click', function () {
    elbowLoaded = false;
    elbowData = null;

    // Also reset RFM loaded state
    rfmLoaded = false;
    rfmData = null;
});


// Time Series Forecasting functionality
let selectedMethod = null;

document.querySelector('[data-tab="forecast"]').addEventListener('click', async function () {
    if (!currentSessionId) {
        showPopup('Please process a file first to perform Time Series Forecasting.', 'error');
        return;
    }
});
    

// Method selection handlers
document.querySelectorAll('.forecast-method').forEach(method => {
    method.addEventListener('click', function () {
        // Remove selection from all methods
        document.querySelectorAll('.forecast-method').forEach(m => {
            m.classList.remove('selected');
            m.querySelector('.method-radio').classList.remove('selected');
            m.querySelector('.input-group').style.display = 'none';
        });

        // Select current method
        this.classList.add('selected');
        this.querySelector('.method-radio').classList.add('selected');
        this.querySelector('.input-group').style.display = 'flex';

        selectedMethod = this.dataset.method;
        updateForecastButton();
    });
});

// Update forecast button state
function updateForecastButton() {
    const button = document.getElementById('startForecast');
    const hasSession = currentSessionId !== null;
    const hasMethod = selectedMethod !== null;

    button.disabled = !(hasSession && hasMethod);

    if (!hasSession) {
        button.textContent = 'Process a dataset first';
    } else if (!hasMethod) {
        button.textContent = 'Select a forecasting method';
    } else {
        button.textContent = 'Start Forecasting';
    }
}

// Start forecasting
document.getElementById('startForecast').addEventListener('click', async function () {
    if (!selectedMethod || !currentSessionId) {
        showPopup('Please select a method and ensure you have a processed dataset', 'error');
        return;
    }

    const button = this;
    const originalText = button.textContent;

    try {
        button.disabled = true;
        button.innerHTML = '<span class="loading-spinner"></span>Processing...';

        let hours, endpoint, payload;
        const serverUrl = document.getElementById('serverUrl').value.trim().replace(/\/$/, '');

        if (selectedMethod === 'existing') {
            hours = parseInt(document.getElementById('existingHours').value) || 24;
            endpoint = `${serverUrl}/predict_future/${hours}`;
            payload = {
                model_path: "Trained_Model/prophet_model.pkl",
                behavior_type: "buy"
            };
        } else {
            hours = parseInt(document.getElementById('updateHours').value) || 168;
            endpoint = `${serverUrl}/update_and_predict/${currentSessionId}/${hours}`;
            payload = {
                new_csv_path: null
            };
        }

        // Validate hours
        if (hours < 1 || hours > 8760) {
            throw new Error('Hours must be between 1 and 8760 (1 year)');
        }

        showPopup(`Starting ${selectedMethod === 'existing' ? 'existing model' : 'updated model'} forecasting for ${hours} hours...`, 'loading', 5000);

        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP ${response.status}: ${response.statusText}`);
        }

        const result = await response.json();

        if (!result.success) {
            throw new Error(result.error || 'Forecasting failed');
        }

        displayForecastResults(result, selectedMethod);
        showPopup('✅ Forecasting completed successfully!', 'success');

    } catch (error) {
        console.error('Forecasting error:', error);
        showPopup(`Forecasting failed: ${error.message}`, 'error');
        displayError(error.message);
    } finally {
        button.disabled = false;
        button.innerHTML = originalText;
        updateForecastButton();
    }
});

// Display forecast results
function displayForecastResults(data, method) {
    const resultsDiv = document.getElementById('forecastResults');

    // Update metadata
    document.getElementById('methodUsed').textContent =
        method === 'existing' ? 'Existing Model' : 'Updated Model';
    document.getElementById('hoursPredicted').textContent = data.hours_predicted;
    document.getElementById('behaviorType').textContent = data.behavior_type;
    document.getElementById('totalPredictions').textContent = data.total_predictions;

    // Display plots
    if (data.forecast_plot) {
        const forecastImg = document.getElementById('forecastPlot');
        forecastImg.src = `data:image/png;base64,${data.forecast_plot}`;
        forecastImg.style.display = 'block';
    }

    if (data.components_plot) {
        const componentsImg = document.getElementById('componentsPlot');
        componentsImg.src = `data:image/png;base64,${data.components_plot}`;
        componentsImg.style.display = 'block';
    }

    // Populate predictions table
    populatePredictionsTable(data.predictions);

    // Show results
    resultsDiv.classList.add('show');

    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

// Populate predictions table
function populatePredictionsTable(predictions) {
    const tbody = document.getElementById('predictionsTableBody');
    tbody.innerHTML = '';

    predictions.forEach(pred => {
        const row = document.createElement('tr');

        const dateTime = new Date(pred.ds).toLocaleString();
        const predictedValue = pred.yhat ? pred.yhat.toFixed(4) : 'N/A';
        const lowerBound = pred.yhat_lower ? pred.yhat_lower.toFixed(4) : 'N/A';
        const upperBound = pred.yhat_upper ? pred.yhat_upper.toFixed(4) : 'N/A';

        row.innerHTML = `
                    <td>${dateTime}</td>
                    <td>${predictedValue}</td>
                    <td>${lowerBound}</td>
                    <td>${upperBound}</td>
                `;

        tbody.appendChild(row);
    });
}

// Display error message
function displayError(message) {
    const resultsDiv = document.getElementById('forecastResults');
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = `Error: ${message}`;

    resultsDiv.innerHTML = '';
    resultsDiv.appendChild(errorDiv);
    resultsDiv.classList.add('show');
}

// Image enlargement functionality
document.addEventListener('click', function (e) {
    if (e.target.classList.contains('forecast-plot')) {
        const overlay = document.getElementById('overlay');
        const img = e.target.cloneNode(true);

        img.classList.add('enlarged');
        overlay.appendChild(img);
        overlay.classList.add('show');

        // Close on overlay click
        overlay.addEventListener('click', function () {
            overlay.classList.remove('show');
            setTimeout(() => {
                overlay.innerHTML = '';
            }, 300);
        });
    }
});

// Update forecast button when session changes
const originalEnableTabs = enableTabs;
enableTabs = function () {
    originalEnableTabs();
    updateForecastButton();
};

// Initialize
updateForecastButton();


function renderEvents() {
    // Clear existing tags except the "No events selected" message and clear button
    const existingTags = selectedEventsContainer.querySelectorAll('.event-tag');
    existingTags.forEach(tag => tag.remove());

    if (selectedEvents.length === 0) {
        noEventsMessage.classList.remove('hidden'); // Show "No events selected" message
        clearAllButton.classList.add('hidden'); // Hide clear all button
    } else {
        noEventsMessage.classList.add('hidden'); // Hide "No events selected" message
        clearAllButton.classList.remove('hidden'); // Show clear all button

        selectedEvents.forEach((event, index) => {
            // Create a span element for each event tag
            const eventTag = document.createElement('span');
            eventTag.className = 'event-tag flex items-center bg-blue-200 text-blue-800 text-sm font-medium px-3 py-1 rounded-full cursor-pointer hover:bg-blue-300 transition-colors duration-200';
            eventTag.textContent = event;
            eventTag.dataset.index = index; // Store index for removal

            // Add a small 'x' icon within the tag itself for removal
            const closeIcon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            closeIcon.setAttribute("class", "ml-1 -mr-1 h-3 w-3 text-blue-600");
            closeIcon.setAttribute("fill", "currentColor");
            closeIcon.setAttribute("viewBox", "0 0 20 20");
            const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
            path.setAttribute("fillRule", "evenodd");
            path.setAttribute("d", "M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z");
            path.setAttribute("clipRule", "evenodd");
            closeIcon.appendChild(path);
            eventTag.appendChild(closeIcon);

            // Add click listener to remove the event when clicked
            eventTag.addEventListener('click', (e) => {
                // Prevent the click on the parent container from triggering twice
                e.stopPropagation();
                const indexToRemove = parseInt(eventTag.dataset.index);
                removeEvent(indexToRemove);
            });

            // Insert the new tag before the clear all button
            selectedEventsContainer.insertBefore(eventTag, clearAllButton);
        });
    }
}

// Function to add an event tag to the selectedEvents array
function addEvent(eventValue) {
    selectedEvents.push(eventValue);
    renderEvents(); // Re-render the UI
}

// Function to remove a specific event tag
function removeEvent(indexToRemove) {
    selectedEvents.splice(indexToRemove, 1);
    renderEvents(); // Re-render the UI
}

// Function to clear all event tags
function clearAllEvents() {
    selectedEvents = [];
    renderEvents(); // Re-render the UI
}

// Function to handle the "Predict" button click
async function handlePredict() {
    console.log("Current selected events for prediction:", selectedEvents);

    try {
        const serverUrl = serverUrlInput.value.trim().replace(/\/$/, '');
        body_data = JSON.stringify(selectedEvents);
        console.log("Sending data for prediction:", body_data);
        const response = await fetch(`${serverUrl}/predict_next_user_behavior`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: body_data,
            mode: 'cors'
        });
        if (!response.ok) throw new Error('Failed to fetch EDA results');
        data = await response.json();
        console.log("Prediction response data:", data);

        const resizeCanvas = () => {
            const parentWidth = canvas.parentElement.clientWidth;
            // Set max width to 400px or parent width, whichever is smaller
            const size = Math.min(400, parentWidth);
            canvas.width = size;
            canvas.height = size;
            drawPieChart(data["predictions"]); // Redraw chart on resize
        };
        resizeCanvas();

    } catch (error) {
        console.error('Error fetching prediction:', error);
        showPopup(`Prediction failed: ${error.message}`, 'error');
        return;
    }

}

// Add event listeners to the action buttons
document.getElementById('view-page-button').addEventListener('click', () => addEvent("PageView"));
document.getElementById('favorite-button').addEventListener('click', () => addEvent("Favorite"));
document.getElementById('add-to-cart-button').addEventListener('click', () => addEvent("AddToCart"));
document.getElementById('purchase-button').addEventListener('click', () => addEvent("Buy"));

// Add event listener to the clear all button
clearAllButton.addEventListener('click', clearAllEvents);

// Add event listener to the predict button
predictButton.addEventListener('click', handlePredict);

// Initial render when the page loads
document.addEventListener('DOMContentLoaded', renderEvents);

function drawPieChart(eventData) {
    // Clear the canvas before redrawing
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    legendDiv.innerHTML = '<h2 class="text-xl font-semibold mb-4">Legend:</h2>'; // Clear legend

    const totalProbability = eventData.reduce((sum, item) => sum + item.probability, 0);
    let startAngle = 0;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) * 0.7; // Use 70% of the smaller dimension for radius


    eventData.forEach((item, index) => {
        const sliceAngle = (item.probability / totalProbability) * 2 * Math.PI;
        const endAngle = startAngle + sliceAngle;
        const color = getColor(item.event);

        ctx.beginPath();
        ctx.moveTo(centerX, centerY); // Move to the center of the pie
        ctx.arc(centerX, centerY, radius, startAngle, endAngle); // Draw arc
        ctx.closePath();
        ctx.fillStyle = color;
        ctx.fill();

        // Draw a border for each slice to make them more distinct
        ctx.strokeStyle = '#ffffff'; // White border
        ctx.lineWidth = 2; // Border width
        ctx.stroke();

        // Add text label for probability percentage inside the slice
        // Calculate the midpoint of the slice for text placement
        const textAngle = startAngle + sliceAngle / 2;
        const textX = centerX + radius * 0.6 * Math.cos(textAngle);
        const textY = centerY + radius * 0.6 * Math.sin(textAngle);

        ctx.fillStyle = '#333333'; // Darker text color
        ctx.font = 'bold 14px Inter, sans-serif'; // Bold font for visibility
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const percentage = (item.probability * 100).toFixed(1);
        // Only draw percentage if slice is large enough to prevent overcrowding
        if (sliceAngle > 0.1) { // Arbitrary threshold for drawing text
            ctx.fillText(`${percentage}%`, textX, textY);
        }

        // Add item to legend
        const legendItem = document.createElement('div');
        legendItem.className = 'flex items-center mb-2';
        legendItem.innerHTML = `
                    <span class="inline-block w-4 h-4 rounded-sm mr-3" style="background-color: ${color};"></span>
                    <span class="text-gray-700">${item.event}: <span class="font-semibold">${(item.probability * 100).toFixed(2)}%</span></span>
                `;
        legendDiv.appendChild(legendItem);

        startAngle = endAngle; // Update start angle for the next slice
    });
}
function getColor(type){
    switch(type) {
        case 'PageView':
            return 'hsl(0.0, 70%, 60%)';
        case 'Favorite':
            return 'hsl(90.0, 70%, 60%)'; 
        case 'AddToCart':
            return 'hsl(210.0, 70%, 60%)';
        case 'Buy':
            return 'hsl(270.0, 70%, 60%)'; 
    }
}

document.querySelector('[data-tab="recommend"]').addEventListener('click', function () {
    // Reset input fields and results on tab click
    const recommendTab = document.getElementById('recommend');
    const placeholder = recommendTab.querySelector('.placeholder-content');
    if (!placeholder.querySelector('#recommend-user-id')) {
        placeholder.innerHTML = `
            <div style="display: flex; gap: 16px; margin-bottom: 16px;">
                <input id="recommend-user-id" type="text" placeholder="User ID" class="server-url-input" style="width: 140px;">
                <input id="recommend-product-id" type="text" placeholder="Product ID" class="server-url-input" style="width: 140px;">
                <button id="get-recommend-btn" class="process-button">Get Recommendations</button>
            </div>
            <div id="recommend-results" style="width: 100%;"></div>
        `;
    } else {
        // Clear previous results
        document.getElementById('recommend-results').innerHTML = '';
    }
});

// Handle Get Recommendations button click
document.addEventListener('click', async function (e) {
    if (e.target && e.target.id === 'get-recommend-btn') {
        const userId = document.getElementById('recommend-user-id').value.trim();
        const productId = document.getElementById('recommend-product-id').value.trim();
        const resultsDiv = document.getElementById('recommend-results');
        if (!userId || !productId) {
            resultsDiv.innerHTML = `<p style="color:red;">Please enter both User ID and Product ID.</p>`;
            return;
        }
        resultsDiv.innerHTML = '<p>Loading recommendations...</p>';
        // Send request as before (no change to sending part)
        const serverUrl = serverUrlInput.value.trim().replace(/\/$/, '');
        try {
            const response = await fetch(`${serverUrl}/recommend_products`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: currentSessionId, user_id: userId, product_id: productId }),
                mode: 'cors'
            });
            if (!response.ok) throw new Error('Failed to fetch recommendations');
            const data = await response.json();
            if (data.recommended_products && data.recommended_products.length > 0) {
                resultsDiv.innerHTML = `
                    <h3 style="margin-bottom: 1em; color: #667eea;">Recommended Products</h3>
                    <div style="
                        display: flex;
                        flex-wrap: wrap;
                        gap: 12px;
                        justify-content: flex-start;
                        align-items: center;
                        margin-bottom: 1em;
                    ">
                        ${data.recommended_products.map(pid => `
                            <button style="
                                background: linear-gradient(90deg, #667eea 60%, #764ba2 100%);
                                color: #fff;
                                border: none;
                                border-radius: 20px;
                                padding: 10px 22px;
                                font-size: 1.1em;
                                font-weight: 600;
                                box-shadow: 0 2px 8px rgba(102,126,234,0.08);
                                cursor: pointer;
                                transition: background 0.2s, transform 0.2s;
                            "
                            onmouseover="this.style.background='#5a67d8';this.style.transform='scale(1.07)'"
                            onmouseout="this.style.background='linear-gradient(90deg, #667eea 60%, #764ba2 100%)';this.style.transform='scale(1)'"
                            >${pid}</button>
                        `).join('')}
                    </div>
                `;
            } else {
                resultsDiv.innerHTML = '<p>No recommendations found.</p>';
            }
        } catch (err) {
            resultsDiv.innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
        }
    }
});