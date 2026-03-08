// Global variables
let columns = [];
let numericColumns = [];
let categoricalColumns = [];

// Utility Functions
function showLoading() {
    document.getElementById('loadingOverlay').classList.add('active');
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.remove('active');
}

function showAlert(message, type = 'success') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type}`;
    alertDiv.innerHTML = `<i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i> ${message}`;
    
    const container = document.querySelector('.main-container');
    container.insertBefore(alertDiv, container.firstChild);
    
    setTimeout(() => alertDiv.remove(), 5000);
}

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabName).classList.add('active');
    event.target.classList.add('active');
}

// File Upload
document.getElementById('fileInput').addEventListener('change', uploadFile);

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading();
    document.getElementById('uploadStatus').innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading...';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            document.getElementById('uploadStatus').innerHTML = 
                `<i class="fas fa-check-circle" style="color: green;"></i> ${data.message} (${data.rows} rows, ${data.columns} columns)`;
            
            document.getElementById('uploadSection').style.display = 'none';
            document.getElementById('mainContent').style.display = 'block';
            
            await loadOverview();
            showAlert('File uploaded successfully!', 'success');
        } else {
            document.getElementById('uploadStatus').innerHTML = 
                `<i class="fas fa-times-circle" style="color: red;"></i> ${data.error}`;
            showAlert(data.error, 'error');
        }
    } catch (error) {
        document.getElementById('uploadStatus').innerHTML = 
            `<i class="fas fa-times-circle" style="color: red;"></i> Upload failed`;
        showAlert('Upload failed: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Load Overview
async function loadOverview() {
    showLoading();
    try {
        const response = await fetch('/api/overview');
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        columns = data.column_names;
        numericColumns = data.numeric_columns;
        categoricalColumns = data.categorical_columns;
        
        let html = `
            <div class="info-grid">
                <div class="info-card">
                    <strong>Total Rows</strong>
                    <div class="value">${data.rows.toLocaleString()}</div>
                </div>
                <div class="info-card">
                    <strong>Total Columns</strong>
                    <div class="value">${data.columns}</div>
                </div>
                <div class="info-card">
                    <strong>Numeric Columns</strong>
                    <div class="value">${numericColumns.length}</div>
                </div>
                <div class="info-card">
                    <strong>Categorical Columns</strong>
                    <div class="value">${categoricalColumns.length}</div>
                </div>
                <div class="info-card">
                    <strong>Memory Usage</strong>
                    <div class="value">${data.memory_usage}</div>
                </div>
                <div class="info-card">
                    <strong>Duplicate Rows</strong>
                    <div class="value">${data.duplicates}</div>
                </div>
                <div class="info-card">
                    <strong>Total Missing Values</strong>
                    <div class="value">${data.total_missing.toLocaleString()}</div>
                </div>
                <div class="info-card">
                    <strong>Missing Data %</strong>
                    <div class="value">${data.missing_percentage.toFixed(2)}%</div>
                </div>
            </div>
            
            <h3><i class="fas fa-columns"></i> Column Information</h3>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>
                            <th>Column Name</th>
                            <th>Data Type</th>
                            <th>Non-Null</th>
                            <th>Null</th>
                            <th>Unique Values</th>
                            <th>Sample Values</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${data.column_details.map(col => `
                            <tr>
                                <td><strong>${col.name}</strong></td>
                                <td>${col.dtype}</td>
                                <td>${col.non_null}</td>
                                <td>${col.null}</td>
                                <td>${col.unique}</td>
                                <td>${col.sample_values.join(', ')}</td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
            
            <h3><i class="fas fa-table"></i> Data Preview (First 20 Rows)</h3>
            <div class="table-container">
                <table>
                    <thead>
                        <tr>${columns.map(col => `<th>${col}</th>`).join('')}</tr>
                    </thead>
                    <tbody>
                        ${data.head.map(row => 
                            `<tr>${columns.map(col => `<td>${row[col] !== null && row[col] !== undefined ? row[col] : '<span style="color: #999;">null</span>'}</td>`).join('')}</tr>`
                        ).join('')}
                    </tbody>
                </table>
            </div>
        `;
        
        if (Object.keys(data.missing).length > 0) {
            html += '<h3><i class="fas fa-exclamation-triangle"></i> Columns with Missing Values</h3><div class="info-grid">';
            for (const [col, count] of Object.entries(data.missing)) {
                const percentage = ((count / data.rows) * 100).toFixed(2);
                html += `
                    <div class="stats-card">
                        <h4>${col}</h4>
                        <p><strong>${count}</strong> missing values (${percentage}%)</p>
                    </div>
                `;
            }
            html += '</div>';
        } else {
            html += '<div class="alert alert-success"><i class="fas fa-check-circle"></i> No missing values detected!</div>';
        }
        
        document.getElementById('overviewData').innerHTML = html;
        updateAllSelects();
        
    } catch (error) {
        showAlert('Error loading overview: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Update all select dropdowns
function updateAllSelects() {
    // Drop columns
    const dropSelect = document.getElementById('dropColumns');
    dropSelect.innerHTML = columns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    // Fill missing
    const fillSelect = document.getElementById('fillColumn');
    fillSelect.innerHTML = columns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    // Outlier removal
    const outlierSelect = document.getElementById('outlierColumn');
    outlierSelect.innerHTML = numericColumns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    // Duplicate columns
    const duplicateSelect = document.getElementById('duplicateColumns');
    duplicateSelect.innerHTML = columns.map(col => `<option value="${col}">${col}</option>`).join('');
    
    updatePlotOptions();
}

// Fill method change handler
document.getElementById('fillMethod').addEventListener('change', (e) => {
    document.getElementById('customValue').style.display = 
        e.target.value === 'custom' ? 'block' : 'none';
});

// Data Cleaning Functions
async function dropColumns() {
    const select = document.getElementById('dropColumns');
    const selectedColumns = Array.from(select.selectedOptions).map(opt => opt.value);
    
    if (selectedColumns.length === 0) {
        showAlert('Please select columns to drop', 'error');
        return;
    }
    
    if (!confirm(`Are you sure you want to drop ${selectedColumns.length} column(s)?`)) return;
    
    showLoading();
    try {
        const response = await fetch('/api/clean/drop_columns', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({columns: selectedColumns})
        });
        
        const data = await response.json();
        if (response.ok) {
            showAlert(data.message, 'success');
            await loadOverview();
        } else {
            showAlert(data.error, 'error');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function fillMissing() {
    const column = document.getElementById('fillColumn').value;
    const method = document.getElementById('fillMethod').value;
    const customValue = document.getElementById('customValue').value;
    
    const body = {column, method};
    if (method === 'custom') {
        if (!customValue) {
            showAlert('Please enter a custom value', 'error');
            return;
        }
        body.value = customValue;
    }
    
    showLoading();
    try {
        const response = await fetch('/api/clean/fill_missing', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        
        const data = await response.json();
        if (response.ok) {
            showAlert(data.message, 'success');
            await loadOverview();
        } else {
            showAlert(data.error, 'error');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function removeDuplicates() {
    const select = document.getElementById('duplicateColumns');
    const selectedColumns = Array.from(select.selectedOptions).map(opt => opt.value);
    
    const message = selectedColumns.length > 0 
        ? `Remove duplicates based on columns: ${selectedColumns.join(', ')}?`
        : 'Remove all duplicate rows (based on all columns)?';
    
    if (!confirm(message)) return;
    
    showLoading();
    try {
        const response = await fetch('/api/clean/remove_duplicates', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({columns: selectedColumns})
        });
        
        const data = await response.json();
        if (response.ok) {
            showAlert(data.message, 'success');
            await loadOverview();
        } else {
            showAlert(data.error, 'error');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function removeOutliers() {
    const column = document.getElementById('outlierColumn').value;
    
    if (!confirm(`Remove outliers from ${column} using IQR method?`)) return;
    
    showLoading();
    try {
        const response = await fetch('/api/clean/remove_outliers', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({column})
        });
        
        const data = await response.json();
        if (response.ok) {
            showAlert(data.message, 'success');
            await loadOverview();
        } else {
            showAlert(data.error, 'error');
        }
    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Analysis Functions
async function showStatistics() {
    showLoading();
    try {
        const response = await fetch('/api/statistics', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({columns: numericColumns})
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        
        let html = '<div class="table-container"><table><thead><tr><th>Statistic</th>';
        for (const col of Object.keys(data)) {
            html += `<th>${col}</th>`;
        }
        html += '</tr></thead><tbody>';
        
        const stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skewness', 'kurtosis'];
        for (const stat of stats) {
            html += `<tr><td><strong>${stat}</strong></td>`;
            for (const col of Object.keys(data)) {
                const value = data[col][stat];
                html += `<td>${value !== undefined ? value.toFixed(2) : 'N/A'}</td>`;
            }
            html += '</tr>';
        }
        html += '</tbody></table></div>';
        
        document.getElementById('statisticsData').innerHTML = html;
        
    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

async function showCorrelation() {
    const method = document.getElementById('corrMethod').value;
    
    showLoading();
    try {
        const response = await fetch(`/api/correlation?method=${method}`);
        const data = await response.json();
        
        if (!response.ok) throw new Error(data.error);
        
        document.getElementById('correlationPlot').innerHTML = 
            `<img src="data:image/png;base64,${data.image}" alt="Correlation Matrix">`;
        
    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Visualization Functions
function updatePlotOptions() {
    const plotType = document.getElementById('plotType').value;
    let html = '';
    
    if (plotType === 'histogram') {
        html = `
            <select id="plotColumn" class="form-control">
                ${numericColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
            </select>
            <label>
                <span>Bins:</span>
                <input type="number" id="plotBins" class="form-control" value="30" min="5" max="100">
            </label>
        `;
    } else if (plotType === 'boxplot') {
        html = `
            <select id="plotColumn" class="form-control">
                ${numericColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
            </select>
        `;
    } else if (plotType === 'scatter' || plotType === 'line') {
        html = `
            <label>
                <span>X-axis:</span>
                <select id="plotX" class="form-control">
                    ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
                </select>
            </label>
            <label>
                <span>Y-axis:</span>
                <select id="plotY" class="form-control">
                    ${numericColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
                </select>
            </label>
        `;
    } else if (plotType === 'bar') {
        html = `
            <select id="plotColumn" class="form-control">
                ${columns.map(col => `<option value="${col}">${col}</option>`).join('')}
            </select>
            <label>
                <span>Top N:</span>
                <input type="number" id="plotTopN" class="form-control" value="10" min="5" max="50">
            </label>
        `;
    } else if (plotType === 'distribution') {
        html = `
            <select id="plotColumn" class="form-control">
                ${numericColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
            </select>
        `;
    }
    
    document.getElementById('plotOptions').innerHTML = html;
}

async function generatePlot() {
    const plotType = document.getElementById('plotType').value;
    const body = {type: plotType};
    
    if (plotType === 'histogram') {
        body.column = document.getElementById('plotColumn').value;
        body.bins = parseInt(document.getElementById('plotBins').value);
    } else if (plotType === 'boxplot' || plotType === 'distribution') {
        body.column = document.getElementById('plotColumn').value;
    } else if (plotType === 'scatter' || plotType === 'line') {
        body.x = document.getElementById('plotX').value;
        body.y = document.getElementById('plotY').value;
    } else if (plotType === 'bar') {
        body.column = document.getElementById('plotColumn').value;
        body.top_n = parseInt(document.getElementById('plotTopN').value);
    }
    
    showLoading();
    try {
        const response = await fetch('/api/plot', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        
        document.getElementById('plotArea').innerHTML = 
            `<img src="data:image/png;base64,${data.image}" alt="Plot">`;
        
    } catch (error) {
        showAlert('Error: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Download Data
async function downloadData() {
    window.location.href = '/api/download';
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    console.log('CSV Data Analyzer - B.Tech Project Initialized');
});
