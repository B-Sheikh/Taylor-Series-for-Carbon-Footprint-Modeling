/**
 * Dashboard Module for Carbon Footprint
 * Handles data visualization and entry management
 */

// Import or recreate the CarbonFootprintModel class
class CarbonFootprintModel {
    constructor() {
        this.e = [0.5, 2.7, 2.0, 0.12, 1.5, 0.5, -0.3];
        this.a = [0.0001, 0.0005, 0.0003, 0.00001, 0.001, 0.0002, 0.00005];
        this.B = [
            [0,      0.0001, 0.00005, 0.00002, 0.0003,  0.00001, -0.00005],
            [0.0001, 0,      0.0002,  0.00003, 0.0001,  0.00005, -0.00002],
            [0.00005,0.0002, 0,       0.00001, 0.00015, 0.00002, -0.00001],
            [0.00002,0.00003,0.00001, 0,       0.00008, 0.00001, -0.00003],
            [0.0003, 0.0001, 0.00015, 0.00008, 0,       0.0002,  -0.00004],
            [0.00001,0.00005,0.00002, 0.00001, 0.0002,  0,       0.00001],
            [-0.00005,-0.00002,-0.00001,-0.00003,-0.00004,0.00001,0]
        ];
        this.var_names = ['Electricity', 'Fossil Fuel', 'Natural Gas', 'Transportation', 'Industrial', 'Waste', 'Renewable'];
    }

    computeEmissions(x) {
        let linear = 0;
        for (let i = 0; i < 7; i++) linear += this.e[i] * x[i];
        
        let quadratic = 0;
        for (let i = 0; i < 7; i++) quadratic += 0.5 * this.a[i] * x[i] * x[i];
        
        let interaction = 0;
        for (let i = 0; i < 7; i++) {
            for (let j = 0; j < 7; j++) {
                interaction += 0.5 * x[i] * this.B[i][j] * x[j];
            }
        }
        return linear + quadratic + interaction;
    }
}

class DashboardManager {
    constructor() {
        this.model = new CarbonFootprintModel();
        this.charts = {};
        this.init();
    }

    init() {
        this.checkAuth();
        this.bindEvents();
        
        // Set default month to current month
        const monthInput = document.getElementById('entryDate');
        if (monthInput) {
            const now = new Date();
            monthInput.value = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}`;
        }
    }

    checkAuth() {
        const userData = localStorage.getItem('ecocalc_current_user');
        const guestMessage = document.getElementById('guestMessage');
        const dashboardContent = document.getElementById('dashboardContent');

        if (!userData) {
            // Show guest message
            if (guestMessage) guestMessage.classList.remove('hidden');
            if (dashboardContent) dashboardContent.classList.add('hidden');
        } else {
            // Show dashboard
            if (guestMessage) guestMessage.classList.add('hidden');
            if (dashboardContent) {
                dashboardContent.classList.remove('hidden');
                this.loadDashboardData();
            }
        }
    }

    bindEvents() {
        // Entry form submission
        const entryForm = document.getElementById('entryForm');
        if (entryForm) {
            entryForm.addEventListener('submit', (e) => this.handleEntrySubmit(e));
        }

        // Clear entry button
        const clearBtn = document.getElementById('clearEntryBtn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearEntryForm());
        }

        // Guest buttons
        const guestLoginBtn = document.getElementById('guestLoginBtn');
        const guestSignupBtn = document.getElementById('guestSignupBtn');

        if (guestLoginBtn) {
            guestLoginBtn.addEventListener('click', () => {
                if (typeof authManager !== 'undefined') {
                    authManager.showLoginModal();
                }
            });
        }

        if (guestSignupBtn) {
            guestSignupBtn.addEventListener('click', () => {
                if (typeof authManager !== 'undefined') {
                    authManager.showSignupModal();
                }
            });
        }
    }

    handleEntrySubmit(e) {
        e.preventDefault();

        if (typeof authManager === 'undefined' || !authManager.isLoggedIn()) {
            this.showNotification('Please log in to save entries', 'error');
            return;
        }

        const date = document.getElementById('entryDate').value;
        if (!date) {
            this.showNotification('Please select a month', 'error');
            return;
        }

        const entry = {
            date: date,
            electricity: parseFloat(document.getElementById('dashElectricity').value) || 0,
            fossilFuel: parseFloat(document.getElementById('dashFossilFuel').value) || 0,
            naturalGas: parseFloat(document.getElementById('dashNaturalGas').value) || 0,
            transportation: parseFloat(document.getElementById('dashTransportation').value) || 0,
            industrial: parseFloat(document.getElementById('dashIndustrial').value) || 0,
            waste: parseFloat(document.getElementById('dashWaste').value) || 0,
            renewable: parseFloat(document.getElementById('dashRenewable').value) || 0,
            timestamp: new Date().toISOString()
        };

        // Calculate emissions
        const values = [
            entry.electricity,
            entry.fossilFuel,
            entry.naturalGas,
            entry.transportation,
            entry.industrial,
            entry.waste,
            entry.renewable
        ];
        entry.totalEmissions = this.model.computeEmissions(values);

        // Save entry
        if (authManager.saveEntry(entry)) {
            this.showNotification('Entry saved successfully!', 'success');
            this.clearEntryForm();
            this.loadDashboardData();
        } else {
            this.showNotification('Failed to save entry', 'error');
        }
    }

    clearEntryForm() {
        document.getElementById('dashElectricity').value = '';
        document.getElementById('dashFossilFuel').value = '';
        document.getElementById('dashNaturalGas').value = '';
        document.getElementById('dashTransportation').value = '';
        document.getElementById('dashIndustrial').value = '';
        document.getElementById('dashWaste').value = '';
        document.getElementById('dashRenewable').value = '';
    }

    loadDashboardData() {
        if (typeof authManager === 'undefined') return;
        
        const data = authManager.getCurrentUserData();
        
        this.updateStats(data);
        this.updateCharts(data);
        this.updateHistoryTable(data);
        this.updateProgress(data);
    }

    updateStats(data) {
        const totalEntries = data.length;
        document.getElementById('totalEntries').textContent = totalEntries;

        if (totalEntries === 0) {
            document.getElementById('latestFootprint').textContent = '0 kg';
            document.getElementById('bestMonth').textContent = '-';
            document.getElementById('trendValue').textContent = '-';
            return;
        }

        // Latest footprint
        const latest = data[data.length - 1];
        document.getElementById('latestFootprint').textContent = latest.totalEmissions.toFixed(1) + ' kg';

        // Best month (lowest emissions)
        const best = data.reduce((min, curr) => curr.totalEmissions < min.totalEmissions ? curr : min);
        const bestDate = new Date(best.date);
        document.getElementById('bestMonth').textContent = bestDate.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });

        // Trend (comparing first and last)
        if (data.length >= 2) {
            const first = data[0].totalEmissions;
            const last = data[data.length - 1].totalEmissions;
            const change = ((last - first) / first * 100).toFixed(1);
            const trendEl = document.getElementById('trendValue');
            
            if (change < 0) {
                trendEl.textContent = `${Math.abs(change)}% Decrease`;
                trendEl.style.color = '#43A047';
            } else {
                trendEl.textContent = `${change}% Increase`;
                trendEl.style.color = '#E53935';
            }
        } else {
            document.getElementById('trendValue').textContent = 'N/A';
        }
    }

    updateCharts(data) {
        if (data.length === 0) return;

        // Trend Chart
        const trendCtx = document.getElementById('trendChart');
        if (trendCtx) {
            if (this.charts.trend) this.charts.trend.destroy();

            const labels = data.map(d => {
                const date = new Date(d.date);
                return date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' });
            });
            const values = data.map(d => d.totalEmissions);

            this.charts.trend = new Chart(trendCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Total Emissions (kg CO₂e)',
                        data: values,
                        borderColor: '#2E7D32',
                        backgroundColor: 'rgba(46, 125, 50, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 5,
                        pointHoverRadius: 7
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: { display: true, text: 'kg CO₂e' }
                        }
                    }
                }
            });
        }

        // Category Chart (latest entry)
        const categoryCtx = document.getElementById('categoryChart');
        if (categoryCtx) {
            if (this.charts.category) this.charts.category.destroy();

            const latest = data[data.length - 1];
            const categories = [
                this.model.e[0] * latest.electricity,
                this.model.e[1] * latest.fossilFuel,
                this.model.e[2] * latest.naturalGas,
                this.model.e[3] * latest.transportation,
                this.model.e[4] * latest.industrial,
                this.model.e[5] * latest.waste
            ].filter(v => v > 0);

            const labels = ['Electricity', 'Fossil Fuel', 'Natural Gas', 'Transport', 'Industrial', 'Waste'].filter((_, i) => categories[i] > 0);

            this.charts.category = new Chart(categoryCtx, {
                type: 'doughnut',
                data: {
                    labels: labels,
                    datasets: [{
                        data: categories,
                        backgroundColor: ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#95a5a6']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        legend: {
                            position: 'right',
                            labels: { boxWidth: 12, font: { size: 11 } }
                        }
                    }
                }
            });
        }

        // Update targets list
        this.updateTargetsList(data);
    }

    updateTargetsList(data) {
        const container = document.getElementById('targetsList');
        if (!container || data.length === 0) return;

        const latest = data[data.length - 1];
        const avgEmissions = data.reduce((sum, d) => sum + d.totalEmissions, 0) / data.length;
        const targetEmissions = avgEmissions * 0.9; // 10% reduction target

        const percentToTarget = (latest.totalEmissions / targetEmissions * 100).toFixed(1);
        const isUnderTarget = latest.totalEmissions <= targetEmissions;

        container.innerHTML = `
            <div class="target-item ${isUnderTarget ? 'success' : 'warning'}">
                <div class="target-header">
                    <span class="target-label">Monthly Target (10% below average)</span>
                    <span class="target-value">${targetEmissions.toFixed(0)} kg</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill ${isUnderTarget ? 'success' : 'warning'}" style="width: ${Math.min(percentToTarget, 100)}%"></div>
                </div>
                <p class="target-status">
                    ${isUnderTarget ? 'You are under target!' : `${(latest.totalEmissions - targetEmissions).toFixed(0)} kg above target`}
                </p>
            </div>
            <div class="target-item">
                <div class="target-header">
                    <span class="target-label">Best Month Target</span>
                    <span class="target-value">${Math.min(...data.map(d => d.totalEmissions)).toFixed(0)} kg</span>
                </div>
                <p class="target-hint">Your best recorded month - try to beat it!</p>
            </div>
        `;
    }

    updateHistoryTable(data) {
        const tableBody = document.getElementById('historyTableBody');
        const emptyState = document.getElementById('emptyState');
        const table = document.getElementById('historyTable');

        if (!tableBody) return;

        if (data.length === 0) {
            tableBody.innerHTML = '';
            if (emptyState) emptyState.classList.remove('hidden');
            if (table) table.classList.add('hidden');
            return;
        }

        if (emptyState) emptyState.classList.add('hidden');
        if (table) table.classList.remove('hidden');

        tableBody.innerHTML = data.slice().reverse().map(entry => {
            const date = new Date(entry.date);
            const dateStr = date.toLocaleDateString('en-US', { month: 'short', year: 'numeric' });
            
            return `
                <tr>
                    <td>${dateStr}</td>
                    <td>${entry.electricity.toFixed(0)}</td>
                    <td>${entry.fossilFuel.toFixed(0)}</td>
                    <td>${entry.transportation.toFixed(0)}</td>
                    <td>${entry.waste.toFixed(0)}</td>
                    <td><strong>${entry.totalEmissions.toFixed(1)} kg</strong></td>
                    <td>
                        <button class="btn-icon delete-btn" data-date="${entry.date}" title="Delete">
                            <i class="fas fa-trash"></i>
                        </button>
                    </td>
                </tr>
            `;
        }).join('');

        // Add delete handlers
        tableBody.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const date = e.currentTarget.dataset.date;
                if (confirm('Are you sure you want to delete this entry?')) {
                    if (authManager.deleteEntry(date)) {
                        this.showNotification('Entry deleted', 'success');
                        this.loadDashboardData();
                    }
                }
            });
        });
    }

    updateProgress(data) {
        if (data.length === 0) return;

        const avgEmissions = data.reduce((sum, d) => sum + d.totalEmissions, 0) / data.length;
        document.getElementById('avgEmissions').textContent = avgEmissions.toFixed(0) + ' kg';

        // Reduction from first entry
        const first = data[0].totalEmissions;
        const last = data[data.length - 1].totalEmissions;
        const reduction = ((first - last) / first * 100).toFixed(1);
        
        const reductionEl = document.getElementById('reductionPercent');
        if (reduction > 0) {
            reductionEl.textContent = `${reduction}%`;
            reductionEl.style.color = '#43A047';
        } else {
            reductionEl.textContent = `${reduction}%`;
            reductionEl.style.color = '#E53935';
        }

        // Goal progress (target: 500 kg)
        const goal = 500;
        const goalProgress = Math.min((last / goal * 100), 100).toFixed(0);
        document.getElementById('goalProgress').textContent = goalProgress + '%';
        document.getElementById('goalBar').style.width = goalProgress + '%';

        // Update bars
        document.getElementById('avgBar').style.width = '50%';
        document.getElementById('reductionBar').style.width = Math.min(Math.abs(reduction), 100) + '%';
    }

    showNotification(message, type = 'info') {
        const existing = document.querySelector('.notification');
        if (existing) existing.remove();

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;

        document.body.appendChild(notification);
        setTimeout(() => notification.classList.add('show'), 10);
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    new DashboardManager();
});
