"""
Carbon Footprint Model - Interactive Visualization and Analysis
================================================================

This script provides comprehensive visualizations and interpretable outputs
for carbon footprint modeling using Taylor series expansion.

Features:
- Historical data visualization
- Taylor series approximation comparison
- Sensitivity analysis charts
- Contribution breakdown
- Scenario forecasting
- Interactive dashboards
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PART 1: MODEL PARAMETERS AND FUNCTIONS
# ============================================================================

class CarbonFootprintModel:
    """Carbon footprint model with Taylor series expansion"""
    
    def __init__(self):
        """Initialize model parameters based on realistic emission factors"""
        
        # Variable names
        self.var_names = [
            'Electricity', 'Fossil Fuel', 'Natural Gas',
            'Transportation', 'Industrial Output', 'Waste', 'Renewable Energy'
        ]
        
        self.units = ['kWh', 'Liters', 'm³', 'km', 'Tons', 'kg', 'kWh']
        
        # Linear emission factors (kg CO2e per unit)
        self.e = np.array([
            0.5,    # Electricity: 0.5 kg CO2e/kWh
            2.7,    # Fossil fuel: 2.7 kg CO2e/L
            2.0,    # Natural gas: 2.0 kg CO2e/m³
            0.12,   # Transportation: 0.12 kg CO2e/km
            1.5,    # Industrial output: 1.5 kg CO2e/ton
            0.5,    # Waste: 0.5 kg CO2e/kg
            -0.3    # Renewable energy: -0.3 kg CO2e/kWh (offset)
        ])
        
        # Quadratic coefficients (nonlinear scaling)
        self.a = np.array([
            0.0001,  # Electricity
            0.0005,  # Fossil fuel
            0.0003,  # Natural gas
            0.00001, # Transportation
            0.001,   # Industrial output
            0.0002,  # Waste
            0.00005  # Renewable energy
        ])
        
        # Interaction matrix (symmetric)
        self.B = np.array([
            [0,      0.0001, 0.00005, 0.00002, 0.0003,  0.00001, -0.00005],
            [0.0001, 0,      0.0002,  0.00003, 0.0001,  0.00005, -0.00002],
            [0.00005,0.0002, 0,       0.00001, 0.00015, 0.00002, -0.00001],
            [0.00002,0.00003,0.00001, 0,       0.00008, 0.00001, -0.00003],
            [0.0003, 0.0001, 0.00015, 0.00008, 0,       0.0002,  -0.00004],
            [0.00001,0.00005,0.00002, 0.00001, 0.0002,  0,       0.00001],
            [-0.00005,-0.00002,-0.00001,-0.00003,-0.00004,0.00001,0]
        ])
        
        # Baseline values (typical monthly values)
        self.x0 = np.array([1000, 500, 200, 5000, 100, 50, 300])
        
    def compute_emissions(self, x):
        """
        Compute total emissions using full nonlinear model
        
        Parameters:
        -----------
        x : array (7,) - emission source activities
        
        Returns:
        --------
        C : float - total carbon footprint (kg CO2e)
        """
        linear = np.dot(self.e, x)
        quadratic = 0.5 * np.dot(self.a, x**2)
        interaction = 0.5 * x @ self.B @ x
        return linear + quadratic + interaction
    
    def compute_gradient(self, x):
        """Compute gradient at point x"""
        return self.e + self.a * x + self.B @ x
    
    def compute_hessian(self):
        """Compute Hessian matrix (constant for quadratic model)"""
        H = self.B.copy()
        np.fill_diagonal(H, self.a)
        return H
    
    def taylor_first_order(self, x, x0=None):
        """First-order Taylor approximation"""
        if x0 is None:
            x0 = self.x0
        dx = x - x0
        C0 = self.compute_emissions(x0)
        grad = self.compute_gradient(x0)
        return C0 + np.dot(grad, dx)
    
    def taylor_second_order(self, x, x0=None):
        """Second-order Taylor approximation"""
        if x0 is None:
            x0 = self.x0
        dx = x - x0
        C0 = self.compute_emissions(x0)
        grad = self.compute_gradient(x0)
        H = self.compute_hessian()
        return C0 + np.dot(grad, dx) + 0.5 * dx @ H @ dx
    
    def taylor_reduced(self, x, x0=None):
        """Reduced Taylor model (only significant terms)"""
        if x0 is None:
            x0 = self.x0
        dx = x - x0
        C0 = self.compute_emissions(x0)
        grad = self.compute_gradient(x0)
        
        # All first-order terms
        C = C0 + np.dot(grad, dx)
        
        # Significant diagonal terms (indices 0, 1, 3)
        C += 0.5 * self.a[0] * dx[0]**2  # Electricity
        C += 0.5 * self.a[1] * dx[1]**2  # Fossil fuel
        C += 0.5 * self.a[3] * dx[3]**2  # Transportation
        
        # Significant interaction terms
        C += self.B[0, 1] * dx[0] * dx[1]  # Electricity × Fossil fuel
        C += self.B[0, 3] * dx[0] * dx[3]  # Electricity × Transportation
        C += self.B[1, 3] * dx[1] * dx[3]  # Fossil fuel × Transportation
        C += self.B[3, 6] * dx[3] * dx[6]  # Transportation × Renewable
        
        return C

# ============================================================================
# PART 2: GENERATE HISTORICAL DATA
# ============================================================================

def generate_historical_data(model, months=24, noise_level=0.05):
    """
    Generate synthetic historical data with realistic trends
    
    Parameters:
    -----------
    model : CarbonFootprintModel
    months : int - number of months of history
    noise_level : float - random variation percentage
    
    Returns:
    --------
    df : DataFrame - historical data
    """
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30*months)
    dates = pd.date_range(start=start_date, end=end_date, periods=months)
    
    # Initialize data
    data = {
        'Date': dates,
        'Month': [d.strftime('%Y-%m') for d in dates]
    }
    
    # Generate trends for each variable
    t = np.linspace(0, 1, months)
    
    # Electricity: slight upward trend + seasonal
    electricity = model.x0[0] * (1 + 0.15*t + 0.1*np.sin(2*np.pi*t))
    
    # Fossil fuel: slight downward trend (efficiency improvements)
    fossil_fuel = model.x0[1] * (1 - 0.1*t + 0.05*np.sin(2*np.pi*t))
    
    # Natural gas: seasonal heating/cooling
    natural_gas = model.x0[2] * (1 + 0.3*np.sin(2*np.pi*t))
    
    # Transportation: upward trend
    transportation = model.x0[3] * (1 + 0.2*t + 0.08*np.sin(2*np.pi*t))
    
    # Industrial output: business growth
    industrial = model.x0[4] * (1 + 0.25*t + 0.12*np.sin(2*np.pi*t))
    
    # Waste: slight reduction (waste reduction initiatives)
    waste = model.x0[5] * (1 - 0.05*t + 0.05*np.sin(2*np.pi*t))
    
    # Renewable energy: strong upward trend (green initiatives)
    renewable = model.x0[6] * (1 + 0.5*t + 0.1*np.sin(2*np.pi*t))
    
    # Add noise
    np.random.seed(42)
    for i, arr in enumerate([electricity, fossil_fuel, natural_gas, 
                             transportation, industrial, waste, renewable]):
        noise = np.random.normal(1, noise_level, months)
        arr *= noise
        data[model.var_names[i]] = arr
    
    # Compute emissions for each month
    emissions = []
    for i in range(months):
        x = np.array([electricity[i], fossil_fuel[i], natural_gas[i],
                     transportation[i], industrial[i], waste[i], renewable[i]])
        emissions.append(model.compute_emissions(x))
    
    data['Total_Emissions'] = emissions
    
    df = pd.DataFrame(data)
    return df

# ============================================================================
# PART 3: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_historical_emissions(df, model, save_path='emissions_history.png'):
    """Plot historical emissions with breakdown"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Total emissions over time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['Date'], df['Total_Emissions'], 'o-', linewidth=2, 
             markersize=6, label='Total Emissions', color='#e74c3c')
    ax1.fill_between(df['Date'], df['Total_Emissions'], alpha=0.3, color='#e74c3c')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Carbon Footprint Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add trend line
    z = np.polyfit(range(len(df)), df['Total_Emissions'], 1)
    p = np.poly1d(z)
    ax1.plot(df['Date'], p(range(len(df))), "--", alpha=0.8, 
             linewidth=2, color='black', label=f'Trend: {z[0]:+.2f} kg CO₂e/month')
    ax1.legend(fontsize=11)
    
    # 2. Emission sources stacked area
    ax2 = fig.add_subplot(gs[1, :])
    
    # Calculate contributions
    contributions = []
    for i in range(len(df)):
        x = df.iloc[i][model.var_names].values.astype(float)
        linear_contrib = model.e * x
        contributions.append(np.maximum(linear_contrib, 0))  # Only positive for stacking
    
    contributions = np.array(contributions, dtype=float).T
    
    # Convert dates to numeric for stackplot
    date_nums = range(len(df))
    
    colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71', '#9b59b6', '#95a5a6', '#1abc9c']
    ax2.stackplot(date_nums, *contributions, labels=model.var_names, 
                  colors=colors, alpha=0.8)
    
    # Set x-axis to show actual dates
    tick_indices = np.linspace(0, len(df)-1, min(8, len(df)), dtype=int)
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels([df['Date'].iloc[i].strftime('%Y-%m') for i in tick_indices], rotation=45)
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Emissions Contribution (kg CO₂e)', fontsize=12, fontweight='bold')
    ax2.set_title('Emission Sources Breakdown (Stacked)', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, ncol=3)
    ax2.grid(True, alpha=0.3)
    
    # 3. Average contribution pie chart
    ax3 = fig.add_subplot(gs[2, 0])
    avg_contributions = contributions.mean(axis=1)
    
    # Only show positive contributors
    positive_mask = avg_contributions > 0
    pie_values = avg_contributions[positive_mask]
    pie_labels = [model.var_names[i] for i in range(len(model.var_names)) if positive_mask[i]]
    pie_colors = [colors[i] for i in range(len(colors)) if positive_mask[i]]
    
    wedges, texts, autotexts = ax3.pie(pie_values, labels=pie_labels, autopct='%1.1f%%',
                                        colors=pie_colors, startangle=90)
    ax3.set_title('Average Emission Source Contribution', fontsize=12, fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 4. Month-over-month change
    ax4 = fig.add_subplot(gs[2, 1])
    mom_change = df['Total_Emissions'].diff()
    colors_bar = ['green' if x < 0 else 'red' for x in mom_change]
    ax4.bar(range(len(mom_change)), mom_change, color=colors_bar, alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax4.set_xlabel('Month Index', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Change (kg CO₂e)', fontsize=12, fontweight='bold')
    ax4.set_title('Month-over-Month Emission Changes', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_taylor_approximation_comparison(df, model, save_path='taylor_comparison.png'):
    """Compare Taylor approximations with actual emissions"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Use last 6 months as baseline
    baseline_idx = len(df) - 6
    x0 = df.iloc[baseline_idx][model.var_names].values
    
    # Compute approximations for recent months
    recent_months = 12
    start_idx = max(0, len(df) - recent_months)
    recent_df = df.iloc[start_idx:].copy()
    
    true_emissions = []
    first_order = []
    second_order = []
    reduced_model = []
    
    for i in range(len(recent_df)):
        x = recent_df.iloc[i][model.var_names].values
        true_emissions.append(model.compute_emissions(x))
        first_order.append(model.taylor_first_order(x, x0))
        second_order.append(model.taylor_second_order(x, x0))
        reduced_model.append(model.taylor_reduced(x, x0))
    
    dates = recent_df['Date'].values
    
    # 1. All approximations
    ax1 = axes[0, 0]
    ax1.plot(dates, true_emissions, 'o-', linewidth=2, markersize=8, 
             label='True Emissions', color='black')
    ax1.plot(dates, first_order, 's--', linewidth=2, markersize=6,
             label='1st Order Taylor', color='#3498db', alpha=0.7)
    ax1.plot(dates, second_order, '^--', linewidth=2, markersize=6,
             label='2nd Order Taylor', color='#2ecc71', alpha=0.7)
    ax1.plot(dates, reduced_model, 'd--', linewidth=2, markersize=6,
             label='Reduced Model', color='#e74c3c', alpha=0.7)
    ax1.axvline(x=df.iloc[baseline_idx]['Date'], color='red', 
                linestyle=':', linewidth=2, label='Baseline Point')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
    ax1.set_title('Taylor Series Approximations vs True Emissions', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Absolute errors
    ax2 = axes[0, 1]
    error_1st = np.abs(np.array(true_emissions) - np.array(first_order))
    error_2nd = np.abs(np.array(true_emissions) - np.array(second_order))
    error_reduced = np.abs(np.array(true_emissions) - np.array(reduced_model))
    
    x_pos = np.arange(len(dates))
    width = 0.25
    
    ax2.bar(x_pos - width, error_1st, width, label='1st Order', color='#3498db', alpha=0.7)
    ax2.bar(x_pos, error_2nd, width, label='2nd Order', color='#2ecc71', alpha=0.7)
    ax2.bar(x_pos + width, error_reduced, width, label='Reduced', color='#e74c3c', alpha=0.7)
    
    ax2.set_xlabel('Month Index', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Absolute Error (kg CO₂e)', fontsize=12, fontweight='bold')
    ax2.set_title('Approximation Errors', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Relative errors (percentage)
    ax3 = axes[1, 0]
    rel_error_1st = 100 * error_1st / np.array(true_emissions)
    rel_error_2nd = 100 * error_2nd / np.array(true_emissions)
    rel_error_reduced = 100 * error_reduced / np.array(true_emissions)
    
    ax3.plot(dates, rel_error_1st, 'o-', linewidth=2, markersize=6,
             label='1st Order', color='#3498db')
    ax3.plot(dates, rel_error_2nd, 's-', linewidth=2, markersize=6,
             label='2nd Order', color='#2ecc71')
    ax3.plot(dates, rel_error_reduced, '^-', linewidth=2, markersize=6,
             label='Reduced', color='#e74c3c')
    
    ax3.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Relative Error (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Relative Approximation Errors', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. Error statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_data = [
        ['Model', 'Mean Error (kg)', 'Max Error (kg)', 'Mean Error (%)'],
        ['1st Order', f'{error_1st.mean():.2f}', f'{error_1st.max():.2f}', 
         f'{rel_error_1st.mean():.3f}%'],
        ['2nd Order', f'{error_2nd.mean():.2f}', f'{error_2nd.max():.2f}',
         f'{rel_error_2nd.mean():.3f}%'],
        ['Reduced', f'{error_reduced.mean():.2f}', f'{error_reduced.max():.2f}',
         f'{rel_error_reduced.mean():.3f}%']
    ]
    
    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code rows
    colors_rows = ['#3498db', '#2ecc71', '#e74c3c']
    for i in range(1, 4):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(colors_rows[i-1])
            cell.set_alpha(0.3)
    
    ax4.set_title('Error Statistics Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_sensitivity_analysis(model, save_path='sensitivity_analysis.png'):
    """Perform and visualize sensitivity analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Baseline
    x0 = model.x0.copy()
    C0 = model.compute_emissions(x0)
    grad = model.compute_gradient(x0)
    
    # 1. Gradient bar chart (marginal emissions)
    ax1 = axes[0, 0]
    colors = ['green' if g < 0 else 'red' for g in grad]
    bars = ax1.barh(model.var_names, grad, color=colors, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.set_xlabel('Marginal Emissions (kg CO₂e per unit)', fontsize=12, fontweight='bold')
    ax1.set_title('Sensitivity: Marginal Impact of Each Source', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, grad)):
        ax1.text(val, i, f' {val:.3f}', va='center', 
                ha='left' if val > 0 else 'right', fontweight='bold')
    
    # 2. Percentage change impact
    ax2 = axes[0, 1]
    percent_changes = np.linspace(-20, 20, 9)
    
    for i, var_name in enumerate(model.var_names):
        impacts = []
        for pct in percent_changes:
            x_new = x0.copy()
            x_new[i] *= (1 + pct/100)
            C_new = model.compute_emissions(x_new)
            impacts.append(C_new - C0)
        ax2.plot(percent_changes, impacts, 'o-', linewidth=2, 
                markersize=6, label=var_name, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_xlabel('Change in Activity (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Change in Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
    ax2.set_title('Impact of ±20% Change in Each Source', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Heatmap of interaction effects
    ax3 = axes[1, 0]
    H = model.compute_hessian()
    
    # Mask diagonal for better visualization of interactions
    mask = np.eye(7, dtype=bool)
    H_interactions = H.copy()
    H_interactions[mask] = np.nan
    
    im = ax3.imshow(H_interactions, cmap='RdBu_r', aspect='auto', vmin=-0.0001, vmax=0.0001)
    ax3.set_xticks(range(7))
    ax3.set_yticks(range(7))
    ax3.set_xticklabels([name[:12] for name in model.var_names], rotation=45, ha='right')
    ax3.set_yticklabels([name[:12] for name in model.var_names])
    ax3.set_title('Interaction Effects (Hessian Off-Diagonal)', fontsize=13, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Interaction Coefficient', fontsize=10)
    
    # Add text annotations for significant interactions
    for i in range(7):
        for j in range(7):
            if i != j and abs(H[i,j]) > 0.00005:
                text = ax3.text(j, i, f'{H[i,j]:.5f}',
                              ha="center", va="center", color="black", fontsize=8)
    
    # 4. Scenario comparison
    ax4 = axes[1, 1]
    
    scenarios = {
        'Baseline': x0,
        '+10% All': x0 * 1.1,
        '-10% Fossil': x0 * np.array([1, 0.9, 1, 1, 1, 1, 1]),
        '+20% Renewable': x0 * np.array([1, 1, 1, 1, 1, 1, 1.2]),
        'Efficiency\n(-10% E,F,T)': x0 * np.array([0.9, 0.9, 1, 0.9, 1, 1, 1]),
    }
    
    scenario_names = list(scenarios.keys())
    emissions = [model.compute_emissions(scenarios[name]) for name in scenario_names]
    changes = [e - emissions[0] for e in emissions]
    
    colors_scenario = ['gray' if c == 0 else 'green' if c < 0 else 'red' 
                       for c in changes]
    
    bars = ax4.bar(range(len(scenario_names)), emissions, color=colors_scenario, alpha=0.7)
    ax4.set_xticks(range(len(scenario_names)))
    ax4.set_xticklabels(scenario_names, rotation=0, ha='center')
    ax4.set_ylabel('Total Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
    ax4.set_title('Scenario Analysis', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, em, ch) in enumerate(zip(bars, emissions, changes)):
        ax4.text(i, em + 50, f'{em:.0f}\n({ch:+.0f})', 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def plot_forecast_scenarios(df, model, months_ahead=6, save_path='forecast_scenarios.png'):
    """Forecast future emissions under different scenarios"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Get recent trend
    recent_data = df.tail(12)
    last_values = df.iloc[-1][model.var_names].values
    
    # Calculate growth rates from recent trend
    growth_rates = []
    for var in model.var_names:
        values = recent_data[var].values
        if len(values) > 1:
            trend = np.polyfit(range(len(values)), values, 1)[0]
            growth_rate = trend / values.mean()
        else:
            growth_rate = 0
        growth_rates.append(growth_rate)
    
    growth_rates = np.array(growth_rates)
    
    # Create forecast scenarios
    future_dates = pd.date_range(start=df['Date'].iloc[-1] + timedelta(days=30),
                                 periods=months_ahead, freq='MS')
    
    scenarios = {
        'Business as Usual': growth_rates,
        'Moderate Reduction': growth_rates * 0.5,
        'Aggressive Reduction': growth_rates * 0.2,
        'Green Transition': growth_rates * np.array([0.8, 0.3, 0.7, 0.8, 0.9, 0.5, 1.5])
    }
    
    colors_forecast = ['#e74c3c', '#f39c12', '#2ecc71', '#1abc9c']
    
    # 1. Total emissions forecast
    ax1 = axes[0, 0]
    
    # Historical
    ax1.plot(df['Date'], df['Total_Emissions'], 'o-', linewidth=2,
             markersize=4, label='Historical', color='black', alpha=0.5)
    
    # Forecasts
    for i, (scenario_name, rates) in enumerate(scenarios.items()):
        forecast_emissions = []
        x_current = last_values.copy()
        
        for month in range(months_ahead):
            x_current = x_current * (1 + rates)
            forecast_emissions.append(model.compute_emissions(x_current))
        
        ax1.plot(future_dates, forecast_emissions, 'o--', linewidth=2,
                markersize=6, label=scenario_name, color=colors_forecast[i], alpha=0.8)
    
    ax1.axvline(x=df['Date'].iloc[-1], color='red', linestyle=':', 
                linewidth=2, label='Current')
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
    ax1.set_title('Emission Forecasts Under Different Scenarios', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Cumulative emissions forecast
    ax2 = axes[0, 1]
    
    current_cumulative = df['Total_Emissions'].sum()
    
    for i, (scenario_name, rates) in enumerate(scenarios.items()):
        cumulative = [current_cumulative]
        x_current = last_values.copy()
        
        for month in range(months_ahead):
            x_current = x_current * (1 + rates)
            cumulative.append(cumulative[-1] + model.compute_emissions(x_current))
        
        forecast_dates_with_current = [df['Date'].iloc[-1]] + list(future_dates)
        ax2.plot(forecast_dates_with_current, cumulative, 'o-', linewidth=2,
                markersize=6, label=scenario_name, color=colors_forecast[i], alpha=0.8)
    
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Emissions (kg CO₂e)', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Emissions Forecast', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Emissions reduction potential
    ax3 = axes[1, 0]
    
    bau_total = 0
    x_current = last_values.copy()
    for month in range(months_ahead):
        x_current = x_current * (1 + scenarios['Business as Usual'])
        bau_total += model.compute_emissions(x_current)
    
    reductions = []
    scenario_names = []
    for scenario_name, rates in scenarios.items():
        if scenario_name != 'Business as Usual':
            total = 0
            x_current = last_values.copy()
            for month in range(months_ahead):
                x_current = x_current * (1 + rates)
                total += model.compute_emissions(x_current)
            reduction = bau_total - total
            reductions.append(reduction)
            scenario_names.append(scenario_name)
    
    bars = ax3.barh(scenario_names, reductions, color=colors_forecast[1:], alpha=0.7)
    ax3.set_xlabel('Emissions Reduction vs BAU (kg CO₂e)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Total Reduction Potential ({months_ahead} months)', 
                  fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add values
    for i, (bar, val) in enumerate(zip(bars, reductions)):
        ax3.text(val, i, f' {val:.0f} kg', va='center', 
                ha='left', fontweight='bold')
    
    # 4. Key metrics comparison
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate metrics for each scenario
    metrics_data = [['Scenario', f'{months_ahead}-Month\nTotal (kg CO₂e)', 
                     'vs BAU\n(kg CO₂e)', 'vs BAU\n(%)']]
    
    for i, (scenario_name, rates) in enumerate(scenarios.items()):
        total = 0
        x_current = last_values.copy()
        for month in range(months_ahead):
            x_current = x_current * (1 + rates)
            total += model.compute_emissions(x_current)
        
        diff = total - bau_total
        pct = (diff / bau_total) * 100
        
        metrics_data.append([
            scenario_name,
            f'{total:.0f}',
            f'{diff:+.0f}',
            f'{pct:+.1f}%'
        ])
    
    table = ax4.table(cellText=metrics_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#34495e')
        cell.set_text_props(weight='bold', color='white')
    
    # Color code rows
    for i in range(1, len(metrics_data)):
        for j in range(4):
            cell = table[(i, j)]
            cell.set_facecolor(colors_forecast[i-1])
            cell.set_alpha(0.3)
    
    ax4.set_title('Forecast Metrics Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()

def generate_summary_report(df, model, output_file='carbon_footprint_report.txt'):
    """Generate comprehensive text summary report"""
    
    report = []
    report.append("="*80)
    report.append("CARBON FOOTPRINT ANALYSIS REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Analysis Period: {df['Date'].iloc[0].strftime('%Y-%m-%d')} to {df['Date'].iloc[-1].strftime('%Y-%m-%d')}")
    report.append(f"Total Months Analyzed: {len(df)}")
    report.append("="*80)
    
    # Summary statistics
    report.append("\n1. EMISSIONS SUMMARY")
    report.append("-"*80)
    report.append(f"   Total Cumulative Emissions: {df['Total_Emissions'].sum():,.2f} kg CO₂e")
    report.append(f"   Average Monthly Emissions:  {df['Total_Emissions'].mean():,.2f} kg CO₂e")
    report.append(f"   Minimum Monthly Emissions:  {df['Total_Emissions'].min():,.2f} kg CO₂e")
    report.append(f"   Maximum Monthly Emissions:  {df['Total_Emissions'].max():,.2f} kg CO₂e")
    report.append(f"   Standard Deviation:         {df['Total_Emissions'].std():,.2f} kg CO₂e")
    
    # Trend analysis
    z = np.polyfit(range(len(df)), df['Total_Emissions'], 1)
    trend_direction = "increasing" if z[0] > 0 else "decreasing"
    report.append(f"\n   Trend: {trend_direction.upper()} at {abs(z[0]):.2f} kg CO₂e per month")
    
    # Recent performance
    recent_6 = df.tail(6)['Total_Emissions'].mean()
    previous_6 = df.tail(12).head(6)['Total_Emissions'].mean() if len(df) >= 12 else recent_6
    change_pct = ((recent_6 - previous_6) / previous_6) * 100
    report.append(f"   Recent 6-month avg vs previous 6-month: {change_pct:+.2f}%")
    
    # Source breakdown
    report.append("\n2. AVERAGE SOURCE CONTRIBUTIONS")
    report.append("-"*80)
    
    for i, var_name in enumerate(model.var_names):
        avg_activity = df[var_name].mean()
        linear_contrib = model.e[i] * avg_activity
        report.append(f"   {var_name:20s}: {linear_contrib:8.2f} kg CO₂e/month  (Avg activity: {avg_activity:,.1f} {model.units[i]})")
    
    # Sensitivity analysis
    report.append("\n3. SENSITIVITY ANALYSIS (at current baseline)")
    report.append("-"*80)
    x_current = df.iloc[-1][model.var_names].values
    grad = model.compute_gradient(x_current)
    
    # Sort by absolute magnitude
    sorted_indices = np.argsort(np.abs(grad))[::-1]
    
    report.append("   Marginal Impact (per unit increase):")
    for idx in sorted_indices:
        report.append(f"   {model.var_names[idx]:20s}: {grad[idx]:+8.4f} kg CO₂e per {model.units[idx]}")
    
    # Top reduction opportunities
    report.append("\n4. TOP EMISSION REDUCTION OPPORTUNITIES")
    report.append("-"*80)
    
    # 10% reduction scenarios
    opportunities = []
    for i, var_name in enumerate(model.var_names):
        if grad[i] > 0:  # Only consider positive contributors
            x_reduced = x_current.copy()
            x_reduced[i] *= 0.9  # 10% reduction
            C_current = model.compute_emissions(x_current)
            C_reduced = model.compute_emissions(x_reduced)
            reduction = C_current - C_reduced
            opportunities.append((var_name, reduction, i))
    
    opportunities.sort(key=lambda x: x[1], reverse=True)
    
    report.append("   Impact of 10% reduction in each source:")
    for var_name, reduction, idx in opportunities[:5]:
        report.append(f"   {var_name:20s}: {reduction:8.2f} kg CO₂e reduction")
    
    # Model accuracy
    report.append("\n5. TAYLOR APPROXIMATION ACCURACY")
    report.append("-"*80)
    
    # Test on last 3 months
    baseline_x = df.iloc[-4][model.var_names].values
    test_errors_1st = []
    test_errors_2nd = []
    test_errors_reduced = []
    
    for i in range(-3, 0):
        x_test = df.iloc[i][model.var_names].values
        true_val = model.compute_emissions(x_test)
        
        error_1st = abs(model.taylor_first_order(x_test, baseline_x) - true_val)
        error_2nd = abs(model.taylor_second_order(x_test, baseline_x) - true_val)
        error_reduced = abs(model.taylor_reduced(x_test, baseline_x) - true_val)
        
        test_errors_1st.append(error_1st / true_val * 100)
        test_errors_2nd.append(error_2nd / true_val * 100)
        test_errors_reduced.append(error_reduced / true_val * 100)
    
    report.append(f"   1st Order Taylor:  Mean error = {np.mean(test_errors_1st):.4f}%")
    report.append(f"   2nd Order Taylor:  Mean error = {np.mean(test_errors_2nd):.4f}%")
    report.append(f"   Reduced Model:     Mean error = {np.mean(test_errors_reduced):.4f}%")
    report.append(f"   Reduced model provides {(1 - 14/35)*100:.1f}% computational savings")
    
    # Recommendations
    report.append("\n6. RECOMMENDATIONS")
    report.append("-"*80)
    
    if z[0] > 0:
        report.append("   ⚠ WARNING: Emissions are trending upward")
        report.append("   → Immediate action required to reverse trend")
    else:
        report.append("   ✓ Emissions are trending downward - continue current strategies")
    
    report.append(f"\n   Priority actions (based on highest marginal impact):")
    for i, (var_name, _, idx) in enumerate(opportunities[:3], 1):
        report.append(f"   {i}. Reduce {var_name}")
    
    if grad[6] < 0:  # Renewable energy
        report.append(f"\n   ✓ Renewable energy showing positive offset effect")
        report.append(f"   → Consider increasing renewable energy usage")
    
    report.append("\n" + "="*80)
    report.append("END OF REPORT")
    report.append("="*80)
    
    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved: {output_file}")
    return '\n'.join(report)

# ============================================================================
# PART 4: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("="*80)
    print("CARBON FOOTPRINT MODEL - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print()
    
    # Initialize model
    print("1. Initializing carbon footprint model...")
    model = CarbonFootprintModel()
    print("   ✓ Model initialized with 7 emission sources")
    print()
    
    # Generate historical data
    print("2. Generating historical data (24 months)...")
    df = generate_historical_data(model, months=24, noise_level=0.05)
    print(f"   ✓ Generated {len(df)} months of historical data")
    print()
    
    # Create visualizations
    print("3. Creating visualizations...")
    plot_historical_emissions(df, model, 'emissions_history.png')
    plot_taylor_approximation_comparison(df, model, 'taylor_comparison.png')
    plot_sensitivity_analysis(model, 'sensitivity_analysis.png')
    plot_forecast_scenarios(df, model, months_ahead=6, save_path='forecast_scenarios.png')
    print()
    
    # Generate report
    print("4. Generating summary report...")
    report_text = generate_summary_report(df, model, 'carbon_footprint_report.txt')
    print()
    
    # Display key findings
    print("="*80)
    print("KEY FINDINGS")
    print("="*80)
    print(f"✓ Historical data analyzed: {len(df)} months")
    print(f"✓ Average monthly emissions: {df['Total_Emissions'].mean():.2f} kg CO₂e")
    print(f"✓ Total cumulative emissions: {df['Total_Emissions'].sum():.2f} kg CO₂e")
    
    z = np.polyfit(range(len(df)), df['Total_Emissions'], 1)
    trend = "↑ INCREASING" if z[0] > 0 else "↓ DECREASING"
    print(f"✓ Trend: {trend} at {abs(z[0]):.2f} kg CO₂e/month")
    
    # Top contributor
    x_current = df.iloc[-1][model.var_names].values
    grad = model.compute_gradient(x_current)
    top_idx = np.argmax(np.abs(grad))
    print(f"✓ Highest marginal impact: {model.var_names[top_idx]} ({grad[top_idx]:+.4f} kg CO₂e per unit)")
    
    print()
    print("="*80)
    print("FILES GENERATED:")
    print("="*80)
    print("   📊 emissions_history.png        - Historical emissions and trends")
    print("   📊 taylor_comparison.png        - Taylor approximation accuracy")
    print("   📊 sensitivity_analysis.png     - Sensitivity and scenario analysis")
    print("   📊 forecast_scenarios.png       - Future emission forecasts")
    print("   📄 carbon_footprint_report.txt  - Comprehensive text report")
    print()
    print("="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
