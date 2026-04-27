/**
 * Carbon Footprint Calculator with Taylor Series Expansion
 * 

 * 
 * Mathematical Model:
 * C(x) = Σ(ei * xi) + 0.5 * Σ(ai * xi²) + 0.5 * x' * B * x
 */

// ===== MODEL PARAMETERS =====
const EMISSION_FACTORS = {
    electricity: 0.5,      // kg CO2e per kWh
    fossil_fuel: 2.7,      // kg CO2e per Liter
    natural_gas: 2.0,      // kg CO2e per m³
    transportation: 0.12,  // kg CO2e per km
    industrial: 1.5,       // kg CO2e per kg material
    waste: 0.5,            // kg CO2e per kg
    renewable: -0.3        // kg CO2e per kWh (negative = offset)
};

// Quadratic coefficients (nonlinear scaling)
const QUADRATIC_COEFFS = {
    electricity: 0.0001,
    fossil_fuel: 0.0005,
    natural_gas: 0.0003,
    transportation: 0.00001,
    industrial: 0.001,
    waste: 0.0002,
    renewable: 0.00005
};

// Interaction matrix (symmetric 7x7)
const INTERACTION_MATRIX = [
    [0,      0.0001, 0.00005, 0.00002, 0.0003,  0.00001, -0.00005],
    [0.0001, 0,      0.0002,  0.00003, 0.0001,  0.00005, -0.00002],
    [0.00005,0.0002, 0,       0.00001, 0.00015, 0.00002, -0.00001],
    [0.00002,0.00003,0.00001, 0,       0.00008, 0.00001, -0.00003],
    [0.0003, 0.0001, 0.00015, 0.00008, 0,       0.0002,  -0.00004],
    [0.00001,0.00005,0.00002, 0.00001, 0.0002,  0,       0.00001],
    [-0.00005,-0.00002,-0.00001,-0.00003,-0.00004,0.00001,0]
];

// Baseline values (average monthly consumption)
const BASELINE = {
    electricity: 300,
    fossil_fuel: 80,
    natural_gas: 25,
    transportation: 800,
    industrial: 50,
    waste: 40,
    renewable: 50
};

const SECTOR_NAMES = [
    'Electricity',
    'Fossil Fuel',
    'Natural Gas',
    'Transportation',
    'Industrial',
    'Waste',
    'Renewable'
];

const SECTOR_ICONS = [
    'fa-bolt',
    'fa-gas-pump',
    'fa-fire',
    'fa-car',
    'fa-industry',
    'fa-trash',
    'fa-solar-panel'
];

// ===== CARBON FOOTPRINT MODEL CLASS =====
class CarbonFootprintModel {
    constructor() {
        this.e = Object.values(EMISSION_FACTORS);
        this.a = Object.values(QUADRATIC_COEFFS);
        this.B = INTERACTION_MATRIX;
        this.x0 = Object.values(BASELINE);
        this.var_names = SECTOR_NAMES;
    }

    // Compute total emissions C(x)
    computeEmissions(x) {
        // Linear terms: Σ(ei * xi)
        let linear = 0;
        for (let i = 0; i < 7; i++) {
            linear += this.e[i] * x[i];
        }

        // Quadratic terms: 0.5 * Σ(ai * xi²)
        let quadratic = 0;
        for (let i = 0; i < 7; i++) {
            quadratic += 0.5 * this.a[i] * x[i] * x[i];
        }

        // Interaction terms: 0.5 * x' * B * x
        let interaction = 0;
        for (let i = 0; i < 7; i++) {
            for (let j = 0; j < 7; j++) {
                interaction += 0.5 * x[i] * this.B[i][j] * x[j];
            }
        }

        return linear + quadratic + interaction;
    }

    // Compute gradient ∇C(x) - marginal emissions
    computeGradient(x) {
        let grad = [];
        for (let i = 0; i < 7; i++) {
            // ∂C/∂xi = ei + ai*xi + Σ(Bij * xj)
            let val = this.e[i] + this.a[i] * x[i];
            for (let j = 0; j < 7; j++) {
                val += this.B[i][j] * x[j];
            }
            grad.push(val);
        }
        return grad;
    }

    // Compute Hessian matrix H(x)
    computeHessian() {
        // H_ij = ai (if i==j) + Bij
        let H = [];
        for (let i = 0; i < 7; i++) {
            H[i] = [];
            for (let j = 0; j < 7; j++) {
                if (i === j) {
                    H[i][j] = this.a[i] + this.B[i][j];
                } else {
                    H[i][j] = this.B[i][j];
                }
            }
        }
        return H;
    }

    // First-order Taylor approximation
    taylorFirstOrder(x, x0) {
        const C0 = this.computeEmissions(x0);
        const grad = this.computeGradient(x0);
        
        let dotProduct = 0;
        for (let i = 0; i < 7; i++) {
            dotProduct += grad[i] * (x[i] - x0[i]);
        }
        
        return C0 + dotProduct;
    }

    // Second-order Taylor approximation
    taylorSecondOrder(x, x0) {
        const C0 = this.computeEmissions(x0);
        const grad = this.computeGradient(x0);
        const H = this.computeHessian();
        
        const dx = x.map((xi, i) => xi - x0[i]);
        
        // First-order term
        let firstOrder = 0;
        for (let i = 0; i < 7; i++) {
            firstOrder += grad[i] * dx[i];
        }
        
        // Second-order term: 0.5 * dx' * H * dx
        let secondOrder = 0;
        for (let i = 0; i < 7; i++) {
            for (let j = 0; j < 7; j++) {
                secondOrder += 0.5 * dx[i] * H[i][j] * dx[j];
            }
        }
        
        return C0 + firstOrder + secondOrder;
    }

    // Sensitivity analysis
    sensitivityAnalysis(x) {
        const C = this.computeEmissions(x);
        const grad = this.computeGradient(x);
        
        let results = [];
        for (let i = 0; i < 7; i++) {
            // Normalized sensitivity index
            const sensitivityIndex = grad[i] * (x[i] / C);
            
            // Impact of 10% perturbation
            const xPlus = [...x];
            xPlus[i] *= 1.1;
            const CPlus = this.computeEmissions(xPlus);
            const impact = CPlus - C;
            
            results.push({
                sector: this.var_names[i],
                gradient: grad[i],
                sensitivityIndex: sensitivityIndex,
                impact10pct: impact,
                currentValue: x[i]
            });
        }
        
        // Sort by sensitivity index (descending)
        results.sort((a, b) => b.sensitivityIndex - a.sensitivityIndex);
        return results;
    }

    // Generate optimization recommendations
    generateRecommendations(x, sensitivityData) {
        const C = this.computeEmissions(x);
        const recommendations = [];
        
        // Define realistic reduction targets for each sector
        const reductionTargets = {
            'Electricity': { realistic: 0.15, label: '15% reduction' },
            'Fossil Fuel': { realistic: 0.20, label: '20% reduction' },
            'Natural Gas': { realistic: 0.10, label: '10% reduction' },
            'Transportation': { realistic: 0.25, label: '25% reduction' },
            'Industrial': { realistic: 0.10, label: '10% reduction' },
            'Waste': { realistic: 0.30, label: '30% reduction' },
            'Renewable': { realistic: -0.50, label: '+50% more renewable' }
        };
        
        const actionItems = {
            'Electricity': [
                'Switch to LED bulbs',
                'Use energy-efficient appliances',
                'Unplug devices when not in use'
            ],
            'Fossil Fuel': [
                'Use public transportation',
                'Consider carpooling',
                'Explore electric vehicle options'
            ],
            'Natural Gas': [
                'Improve home insulation',
                'Use programmable thermostats',
                'Consider heat pump systems'
            ],
            'Transportation': [
                'Walk or bike for short trips',
                'Combine errands to reduce trips',
                'Consider remote work options'
            ],
            'Industrial': [
                'Buy products with less packaging',
                'Choose locally-made products',
                'Reduce consumption of goods'
            ],
            'Waste': [
                'Start composting',
                'Recycle properly',
                'Reduce single-use items'
            ],
            'Renewable': [
                'Install solar panels if feasible',
                'Choose green energy provider',
                'Purchase renewable energy credits'
            ]
        };
        
        for (let item of sensitivityData) {
            const sector = item.sector;
            const target = reductionTargets[sector];
            
            if (!target) continue;
            
            let reductionAmount, newValue;
            if (sector === 'Renewable') {
                // For renewable, increase is good (reduces net emissions)
                newValue = item.currentValue * (1 - target.realistic); // negative = increase
                reductionAmount = -item.gradient * item.currentValue * (-target.realistic);
            } else {
                newValue = item.currentValue * (1 - target.realistic);
                reductionAmount = item.gradient * item.currentValue * target.realistic;
            }
            
            if (reductionAmount > 0.1) { // Only show significant reductions
                const priority = item.sensitivityIndex > 0.3 ? 'high' : 
                                item.sensitivityIndex > 0.15 ? 'medium' : 'low';
                
                recommendations.push({
                    sector: sector,
                    priority: priority,
                    target: target.label,
                    reduction: reductionAmount,
                    actions: actionItems[sector],
                    currentValue: item.currentValue,
                    newValue: newValue,
                    gradient: item.gradient
                });
            }
        }
        
        // Sort by potential reduction (descending)
        recommendations.sort((a, b) => b.reduction - a.reduction);
        return recommendations;
    }

    // Calculate scenarios
    calculateScenarios(x) {
        const C = this.computeEmissions(x);
        const scenarios = [];
        
        // Scenario 1: Moderate reduction (realistic targets)
        const moderateReduction = [0.15, 0.20, 0.10, 0.25, 0.10, 0.30, 0]; // 50% more renewable
        const xModerate = x.map((xi, i) => i === 6 ? xi * 1.5 : xi * (1 - moderateReduction[i]));
        const CModerate = this.computeEmissions(xModerate);
        
        // Scenario 2: Aggressive reduction
        const aggressiveReduction = [0.30, 0.40, 0.25, 0.40, 0.25, 0.50, 0]; // 100% more renewable
        const xAggressive = x.map((xi, i) => i === 6 ? xi * 2.0 : xi * (1 - aggressiveReduction[i]));
        const CAggressive = this.computeEmissions(xAggressive);
        
        // Scenario 3: Electricity + Transport focus (highest impact sectors)
        const xFocus = [...x];
        xFocus[0] *= 0.70; // 30% electricity reduction
        xFocus[3] *= 0.70; // 30% transport reduction
        xFocus[6] *= 1.5;  // 50% more renewable
        const CFocus = this.computeEmissions(xFocus);
        
        scenarios.push(
            { name: 'Current', emissions: C, reduction: 0, percent: 0, class: 'neutral' },
            { name: 'Moderate Reduction', emissions: CModerate, reduction: C - CModerate, 
              percent: ((C - CModerate) / C * 100).toFixed(1), class: 'reduction' },
            { name: 'Aggressive Reduction', emissions: CAggressive, reduction: C - CAggressive,
              percent: ((C - CAggressive) / C * 100).toFixed(1), class: 'reduction' },
            { name: 'Focus: Elec + Transport', emissions: CFocus, reduction: C - CFocus,
              percent: ((C - CFocus) / C * 100).toFixed(1), class: 'reduction' }
        );
        
        return scenarios;
    }
}

// ===== UI CONTROLLER =====
class CalculatorUI {
    constructor() {
        this.model = new CarbonFootprintModel();
        this.charts = {};
        this.init();
    }

    init() {
        this.bindEvents();
    }

    bindEvents() {
        document.getElementById('calculateBtn').addEventListener('click', () => this.calculate());
        document.getElementById('resetBtn').addEventListener('click', () => this.reset());
        
        // Add enter key support for inputs
        const inputs = document.querySelectorAll('#carbonForm input');
        inputs.forEach(input => {
            input.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') this.calculate();
            });
        });
    }

    getInputValues() {
        return [
            parseFloat(document.getElementById('electricity').value) || 0,
            parseFloat(document.getElementById('fossil_fuel').value) || 0,
            parseFloat(document.getElementById('natural_gas').value) || 0,
            parseFloat(document.getElementById('transportation').value) || 0,
            parseFloat(document.getElementById('industrial').value) || 0,
            parseFloat(document.getElementById('waste').value) || 0,
            parseFloat(document.getElementById('renewable').value) || 0
        ];
    }

    calculate() {
        const x = this.getInputValues();
        
        // Validate inputs
        if (x.some(val => val < 0)) {
            alert('Please enter non-negative values for all fields.');
            return;
        }
        
        // Show results section
        document.getElementById('resultsSection').classList.remove('hidden');
        
        // Compute results
        const C = this.model.computeEmissions(x);
        const gradient = this.model.computeGradient(x);
        const sensitivityData = this.model.sensitivityAnalysis(x);
        const recommendations = this.model.generateRecommendations(x, sensitivityData);
        const scenarios = this.model.calculateScenarios(x);
        
        // Update total display
        document.getElementById('totalEmissions').textContent = C.toFixed(2);
        document.getElementById('kmEquivalent').textContent = Math.round(C / 0.12); // 0.12 kg/km avg car
        
        // Update Taylor analysis
        this.updateTaylorAnalysis(x);
        
        // Update gradient display
        this.updateGradientDisplay(gradient);
        
        // Create/update charts
        this.createBreakdownChart(x, C);
        this.createSensitivityChart(sensitivityData);
        
        // Update recommendations
        this.updateRecommendations(recommendations, C);
        
        // Update scenarios
        this.updateScenarios(scenarios);
        
        // Scroll to results
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    }

    updateTaylorAnalysis(x) {
        const x0 = this.model.x0;
        const C = this.model.computeEmissions(x);
        const C1 = this.model.taylorFirstOrder(x, x0);
        const C2 = this.model.taylorSecondOrder(x, x0);
        
        const error1 = Math.abs(C - C1);
        const error2 = Math.abs(C - C2);
        
        document.getElementById('taylor1Result').textContent = `${C1.toFixed(2)} kg CO₂e`;
        document.getElementById('taylor1Error').textContent = `Error: ${error1.toFixed(4)} kg (${(error1/C*100).toFixed(2)}%)`;
        
        document.getElementById('taylor2Result').textContent = `${C2.toFixed(2)} kg CO₂e`;
        document.getElementById('taylor2Error').textContent = `Error: ${error2.toFixed(6)} kg (${(error2/C*100).toFixed(4)}%)`;
    }

    updateGradientDisplay(gradient) {
        const container = document.getElementById('gradientDisplay');
        container.innerHTML = '';
        
        gradient.forEach((val, i) => {
            const item = document.createElement('div');
            item.className = 'gradient-item';
            item.innerHTML = `
                <span>${SECTOR_NAMES[i]}</span>
                <span class="gradient-value">${val.toFixed(4)}</span>
            `;
            container.appendChild(item);
        });
    }

    createBreakdownChart(x, total) {
        const ctx = document.getElementById('breakdownChart').getContext('2d');
        
        // Calculate contributions
        const contributions = x.map((xi, i) => this.model.e[i] * xi);
        
        // Filter out renewable (negative) and very small values
        const chartData = [];
        const chartLabels = [];
        const chartColors = [];
        const colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#95a5a6', '#1abc9c'];
        
        contributions.forEach((val, i) => {
            if (val > 0) {
                chartData.push(val);
                chartLabels.push(SECTOR_NAMES[i]);
                chartColors.push(colors[i]);
            }
        });
        
        if (this.charts.breakdown) {
            this.charts.breakdown.destroy();
        }
        
        this.charts.breakdown = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: chartLabels,
                datasets: [{
                    data: chartData,
                    backgroundColor: chartColors,
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'right'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const value = context.raw;
                                const percent = ((value / total) * 100).toFixed(1);
                                return `${context.label}: ${value.toFixed(1)} kg (${percent}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    createSensitivityChart(sensitivityData) {
        const ctx = document.getElementById('sensitivityChart').getContext('2d');
        
        const labels = sensitivityData.map(d => d.sector);
        const data = sensitivityData.map(d => d.impact10pct);
        
        if (this.charts.sensitivity) {
            this.charts.sensitivity.destroy();
        }
        
        this.charts.sensitivity = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Emission Change (kg CO₂e)',
                    data: data,
                    backgroundColor: data.map(v => v > 0 ? '#e74c3c' : '#2ecc71'),
                    borderRadius: 4
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
                        title: {
                            display: true,
                            text: 'kg CO₂e change'
                        }
                    }
                }
            }
        });
    }

    updateRecommendations(recommendations, totalEmissions) {
        const container = document.getElementById('recommendationsList');
        container.innerHTML = '';
        
        let totalReduction = 0;
        
        recommendations.slice(0, 5).forEach(rec => {
            totalReduction += rec.reduction;
            
            const card = document.createElement('div');
            card.className = `recommendation-card priority-${rec.priority}`;
            card.innerHTML = `
                <div class="rec-content">
                    <h4>
                        <span class="badge badge-${rec.priority}">${rec.priority}</span>
                        ${rec.sector}: ${rec.target}
                    </h4>
                    <p>${rec.actions[0]}</p>
                </div>
                <div class="rec-impact">
                    <div class="impact-value">-${rec.reduction.toFixed(1)}</div>
                    <div class="impact-label">kg CO₂e/month</div>
                </div>
            `;
            container.appendChild(card);
        });
        
        // Update summary
        const percentReduction = (totalReduction / totalEmissions * 100).toFixed(1);
        const optimized = totalEmissions - totalReduction;
        
        document.getElementById('potentialReduction').textContent = totalReduction.toFixed(1);
        document.getElementById('percentReduction').textContent = percentReduction + '%';
        document.getElementById('optimizedFootprint').textContent = optimized.toFixed(1);
    }

    updateScenarios(scenarios) {
        const container = document.getElementById('scenarioTable');
        
        let html = `
            <table>
                <thead>
                    <tr>
                        <th>Scenario</th>
                        <th>Emissions (kg CO₂e)</th>
                        <th>Reduction</th>
                        <th>Reduction %</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        scenarios.forEach(scenario => {
            const reductionClass = scenario.reduction > 0 ? 'reduction' : 
                                  scenario.reduction < 0 ? 'increase' : 'neutral';
            html += `
                <tr>
                    <td><strong>${scenario.name}</strong></td>
                    <td>${scenario.emissions.toFixed(2)}</td>
                    <td class="${reductionClass}">${scenario.reduction > 0 ? '-' : ''}${scenario.reduction.toFixed(2)}</td>
                    <td class="${reductionClass}">${scenario.percent > 0 ? '-' : ''}${scenario.percent}%</td>
                </tr>
            `;
        });
        
        html += '</tbody></table>';
        container.innerHTML = html;
    }

    reset() {
        document.getElementById('carbonForm').reset();
        document.getElementById('resultsSection').classList.add('hidden');
        
        // Reset to baseline values
        document.getElementById('electricity').value = BASELINE.electricity;
        document.getElementById('fossil_fuel').value = BASELINE.fossil_fuel;
        document.getElementById('natural_gas').value = BASELINE.natural_gas;
        document.getElementById('transportation').value = BASELINE.transportation;
        document.getElementById('industrial').value = BASELINE.industrial;
        document.getElementById('waste').value = BASELINE.waste;
        document.getElementById('renewable').value = BASELINE.renewable;
    }
}

// ===== INITIALIZE =====
document.addEventListener('DOMContentLoaded', () => {
    new CalculatorUI();
});
