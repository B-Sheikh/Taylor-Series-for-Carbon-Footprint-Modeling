"""
Carbon Footprint Modeling using Taylor Series Expansion
========================================================

Course: Mathematics for Intelligent Systems 2 (25MAT116)
Topic: Taylor Series for Carbon Footprint Modeling - Emission Sensitivity Analysis

Description:
    This module implements a comprehensive carbon footprint model using
    Taylor series expansion to perform emission sensitivity analysis across
    seven major global emission sectors.

Mathematical Framework:
    C(x) = Σ(ei * xi) + 0.5 * Σ(ai * xi²) + 0.5 * x' * B * x

    Where:
        x  = Activity level vector [x1, x2, ..., x7]
        e  = Linear emission factors
        a  = Quadratic coefficients
        B  = Symmetric interaction matrix (7x7)

Taylor Series Expansion:
    First Order:  C(x) ≈ C(x0) + ∇C(x0)·(x - x0)
    Second Order: C(x) ≈ C(x0) + ∇C(x0)·(x - x0) + 0.5*(x-x0)'·H·(x-x0)

Emission Sectors (x1 - x7):
    x1: Electricity & Heat Generation (TWh)
    x2: Transport - Oil Consumption (Million barrels/day)
    x3: Industrial Energy Use (Exajoules)
    x4: Agriculture - Food Production (Peta-calories)
    x5: Buildings & Real Estate Energy (Exajoules)
    x6: Waste Generation (Megatons)
    x7: Other/Fugitive Sources (Normalized index)
"""

import numpy as np
import pandas as pd


# ============================================================================
# CARBON FOOTPRINT MODEL CLASS
# ============================================================================

class CarbonFootprintModel:
    """
    Carbon footprint model using Taylor series expansion for
    global emission sensitivity analysis.

    Attributes
    ----------
    var_names : list
        Names of the seven emission sectors
    units : list
        Units for each sector's activity level
    e : np.ndarray
        Linear emission factors (7,)
    a : np.ndarray
        Quadratic coefficients (7,)
    b : np.ndarray
        Interaction coefficients (21,)
    x0 : np.ndarray
        Baseline activity levels (7,)
    B : np.ndarray
        Symmetric interaction matrix (7x7)
    """

    def __init__(self):
        """Initialize model with seven emission sectors."""

        self.var_names = [
            'Electricity & Heat',
            'Transport',
            'Industrial',
            'Agriculture',
            'Buildings',
            'Waste',
            'Other Sources'
        ]

        self.units = [
            'TWh',
            'Mbarrel/day',
            'EJ',
            'Peta-cal',
            'EJ',
            'Mt',
            'index'
        ]

        # Parameters initialized as None until set_parameters() is called
        self.e = None
        self.a = None
        self.b = None
        self.x0 = None
        self.B = None

    def set_parameters(self, e, a, b, x0):
        """
        Set model parameters.

        Parameters
        ----------
        e : array-like
            Linear emission factors (length 7)
        a : array-like
            Quadratic coefficients (length 7)
        b : array-like
            Interaction coefficients (length 21)
        x0 : array-like
            Baseline activity levels (length 7)
        """
        self.e = np.array(e, dtype=float)
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float) if b is not None else np.zeros(21)
        self.x0 = np.array(x0, dtype=float)
        self.B = self._build_interaction_matrix(self.b)

    def _build_interaction_matrix(self, b):
        """
        Build symmetric 7x7 interaction matrix from 21 coefficients.

        Parameters
        ----------
        b : np.ndarray
            21 upper-triangle interaction coefficients

        Returns
        -------
        B : np.ndarray
            Symmetric interaction matrix of shape (7, 7)
        """
        B = np.zeros((7, 7))
        idx = 0
        for i in range(7):
            for j in range(i + 1, 7):
                B[i, j] = b[idx]
                B[j, i] = b[idx]  # Symmetry: B_ij = B_ji
                idx += 1
        return B

    def compute_emissions(self, x):
        """
        Calculate total carbon emissions C(x).

        C(x) = Σ(ei*xi) + 0.5*Σ(ai*xi²) + 0.5*x'*B*x

        Parameters
        ----------
        x : array-like
            Activity levels for all seven sectors

        Returns
        -------
        float
            Total emissions in Gt CO2e
        """
        x = np.array(x, dtype=float)

        # Linear terms: Σ(ei * xi)
        C = np.dot(self.e, x)

        # Quadratic terms: 0.5 * Σ(ai * xi²)
        C += 0.5 * np.dot(self.a, x ** 2)

        # Interaction terms: 0.5 * x' * B * x
        C += 0.5 * x @ self.B @ x

        return C

    def compute_gradient(self, x):
        """
        Calculate gradient vector ∇C(x).

        ∂C/∂xi = ei + ai*xi + Σj(Bij*xj)

        Parameters
        ----------
        x : array-like
            Activity levels at which to evaluate gradient

        Returns
        -------
        np.ndarray
            Gradient vector of shape (7,)
        """
        x = np.array(x, dtype=float)

        # Linear component: e
        grad = self.e.copy()

        # Quadratic component: a * x
        grad += self.a * x

        # Interaction component: B * x
        grad += self.B @ x

        return grad

    def compute_hessian(self, x=None):
        """
        Calculate Hessian matrix H(x).

        H_ij = ai (if i==j) + B_ij

        Parameters
        ----------
        x : array-like, optional
            Activity levels (not needed for this model but included for generality)

        Returns
        -------
        np.ndarray
            Hessian matrix of shape (7, 7)
        """
        H = self.B.copy()

        # Add diagonal elements (second derivatives of quadratic terms)
        for i in range(7):
            H[i, i] += self.a[i]

        return H

    def taylor_first_order(self, x, x0):
        """
        First-order Taylor series approximation.

        C(x) ≈ C(x0) + ∇C(x0) · (x - x0)

        Parameters
        ----------
        x : array-like
            Query point
        x0 : array-like
            Expansion point (baseline)

        Returns
        -------
        float
            First-order Taylor approximation of C(x)
        """
        x = np.array(x, dtype=float)
        x0 = np.array(x0, dtype=float)

        C0 = self.compute_emissions(x0)
        grad0 = self.compute_gradient(x0)

        return C0 + np.dot(grad0, x - x0)

    def taylor_second_order(self, x, x0):
        """
        Second-order Taylor series approximation.

        C(x) ≈ C(x0) + ∇C(x0)·(x-x0) + 0.5*(x-x0)'·H(x0)·(x-x0)

        Parameters
        ----------
        x : array-like
            Query point
        x0 : array-like
            Expansion point (baseline)

        Returns
        -------
        float
            Second-order Taylor approximation of C(x)
        """
        x = np.array(x, dtype=float)
        x0 = np.array(x0, dtype=float)
        dx = x - x0

        C0 = self.compute_emissions(x0)
        grad0 = self.compute_gradient(x0)
        H = self.compute_hessian(x0)

        return C0 + np.dot(grad0, dx) + 0.5 * dx @ H @ dx

    def sensitivity_analysis(self, perturbation_pct=10.0):
        """
        Perform sensitivity analysis via parameter perturbation.

        Computes normalized sensitivity index:
            Si = (∂C/∂xi) * (xi / C)

        Parameters
        ----------
        perturbation_pct : float
            Percentage by which to perturb each parameter (default: 10%)

        Returns
        -------
        pd.DataFrame
            Sensitivity results sorted by impact
        """
        gradient = self.compute_gradient(self.x0)
        C_baseline = self.compute_emissions(self.x0)

        results = []
        for i in range(7):
            # Normalized sensitivity index
            Si = gradient[i] * (self.x0[i] / C_baseline)

            # Perturbation analysis
            x_plus = self.x0.copy()
            x_minus = self.x0.copy()
            x_plus[i] *= (1 + perturbation_pct / 100)
            x_minus[i] *= (1 - perturbation_pct / 100)

            C_plus = self.compute_emissions(x_plus)
            C_minus = self.compute_emissions(x_minus)
            abs_impact = (C_plus - C_minus) / 2
            pct_impact = (abs_impact / C_baseline) * 100

            results.append({
                'Sector': self.var_names[i],
                'Unit': self.units[i],
                'Baseline_Activity': self.x0[i],
                'Gradient': gradient[i],
                'Sensitivity_Index': Si,
                'Absolute_Impact_Gt': abs_impact,
                'Percent_Impact': pct_impact
            })

        df = pd.DataFrame(results)
        return df.sort_values('Sensitivity_Index', ascending=False).reset_index(drop=True)

    def scenario_analysis(self, scenarios):
        """
        Evaluate multiple emission reduction scenarios.

        Parameters
        ----------
        scenarios : dict
            Dictionary of {scenario_name: activity_vector}

        Returns
        -------
        pd.DataFrame
            Scenario comparison results
        """
        C_baseline = self.compute_emissions(self.x0)
        results = []

        for name, x in scenarios.items():
            C = self.compute_emissions(np.array(x))
            reduction = C_baseline - C
            results.append({
                'Scenario': name,
                'Total_Emissions_Gt': round(C, 4),
                'Reduction_Gt': round(reduction, 4),
                'Reduction_Pct': round((reduction / C_baseline) * 100, 2)
            })

        return pd.DataFrame(results)

    def validate_taylor(self, n_samples=100, max_perturbation=20):
        """
        Validate Taylor approximation accuracy via Monte Carlo sampling.

        Parameters
        ----------
        n_samples : int
            Number of random samples (default: 100)
        max_perturbation : float
            Maximum parameter perturbation percentage (default: 20%)

        Returns
        -------
        dict
            Validation metrics including R² and MAE
        """
        np.random.seed(42)
        actual, pred1, pred2 = [], [], []

        for _ in range(n_samples):
            x_test = self.x0 * (1 + np.random.uniform(
                -max_perturbation / 100,
                max_perturbation / 100,
                size=7
            ))
            x_test = np.maximum(x_test, 0)

            actual.append(self.compute_emissions(x_test))
            pred1.append(self.taylor_first_order(x_test, self.x0))
            pred2.append(self.taylor_second_order(x_test, self.x0))

        actual = np.array(actual)
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)

        ss_tot = np.sum((actual - actual.mean()) ** 2)
        r2_1 = 1 - np.sum((actual - pred1) ** 2) / ss_tot
        r2_2 = 1 - np.sum((actual - pred2) ** 2) / ss_tot

        return {
            'R2_first_order': round(r2_1, 6),
            'R2_second_order': round(r2_2, 6),
            'MAE_first_order': round(np.mean(np.abs(actual - pred1)), 6),
            'MAE_second_order': round(np.mean(np.abs(actual - pred2)), 6)
        }


# ============================================================================
# BASELINE DATA (Global Emissions 2024-2025)
# Sources: IEA 2024, FAO 2024, UNEP 2024, IPCC AR6
# ============================================================================

BASELINE_ACTIVITIES = {
    'electricity_twh': 30500,       # x1: TWh
    'transport_mbd': 102,           # x2: Million barrels/day
    'industrial_ej': 185,           # x3: Exajoules
    'agriculture_pcal': 52800,      # x4: Peta-calories
    'buildings_ej': 135,            # x5: Exajoules
    'waste_mt': 2100,               # x6: Megatons
    'other_index': 1.0              # x7: Normalized index
}

EMISSION_FACTORS = {
    'electricity': 0.000512,        # Gt CO2e per TWh
    'transport': 0.082344,          # Gt CO2e per Mbarrel/day
    'industrial': 0.035100,         # Gt CO2e per EJ
    'agriculture': 0.000112,        # Gt CO2e per Peta-cal
    'buildings': 0.023700,          # Gt CO2e per EJ
    'waste': 0.000810,              # Gt CO2e per Mt
    'other': 6.400000               # Gt CO2e per unit
}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("=" * 70)
    print("CARBON FOOTPRINT MODELING - TAYLOR SERIES SENSITIVITY ANALYSIS")
    print("25MAT116: Mathematics for Intelligent Systems 2")
    print("=" * 70)

    # Initialize model
    model = CarbonFootprintModel()

    x0 = np.array(list(BASELINE_ACTIVITIES.values()))
    e = np.array(list(EMISSION_FACTORS.values()))
    a = np.zeros(7)    # Linear model
    b = np.zeros(21)   # No interaction terms

    model.set_parameters(e, a, b, x0)

    # ----------------------------------------------------------------
    # BASELINE EMISSIONS
    # ----------------------------------------------------------------
    C_baseline = model.compute_emissions(x0)
    contributions = e * x0

    print("\n1. BASELINE EMISSIONS BREAKDOWN")
    print("-" * 70)
    print(f"{'Sector':<25} {'Activity':>12} {'Factor':>12} {'Emissions (Gt)':>15}")
    print("-" * 70)
    for i in range(7):
        print(f"{model.var_names[i]:<25} {x0[i]:>12,.1f} {e[i]:>12.6f} {contributions[i]:>15.2f}")
    print("-" * 70)
    print(f"{'TOTAL':.<25} {'':>12} {'':>12} {C_baseline:>15.2f} Gt CO2e")

    # ----------------------------------------------------------------
    # GRADIENT VECTOR
    # ----------------------------------------------------------------
    gradient = model.compute_gradient(x0)

    print("\n2. GRADIENT VECTOR ∇C(x0) - Marginal Emission Rates")
    print("-" * 70)
    print(f"{'Sector':<25} {'∂C/∂xi':>20} {'Unit':>20}")
    print("-" * 70)
    for i in range(7):
        print(f"{model.var_names[i]:<25} {gradient[i]:>20.6f} {'Gt per ' + model.units[i]:>20}")

    # ----------------------------------------------------------------
    # SENSITIVITY ANALYSIS
    # ----------------------------------------------------------------
    df_sensitivity = model.sensitivity_analysis(perturbation_pct=10)

    print("\n3. SENSITIVITY ANALYSIS (10% Perturbation)")
    print("-" * 70)
    print(df_sensitivity[['Sector', 'Sensitivity_Index', 'Absolute_Impact_Gt', 'Percent_Impact']].to_string(index=False))

    # ----------------------------------------------------------------
    # TAYLOR APPROXIMATION VALIDATION
    # ----------------------------------------------------------------
    test_x = x0 * 1.10  # 10% increase across all sectors
    C_actual = model.compute_emissions(test_x)
    C_taylor1 = model.taylor_first_order(test_x, x0)
    C_taylor2 = model.taylor_second_order(test_x, x0)

    print("\n4. TAYLOR APPROXIMATION VALIDATION (10% increase all sectors)")
    print("-" * 70)
    print(f"  Actual C(x):           {C_actual:.4f} Gt CO2e")
    print(f"  1st-order Taylor:      {C_taylor1:.4f} Gt CO2e  |  Error: {abs(C_actual - C_taylor1):.6f} Gt")
    print(f"  2nd-order Taylor:      {C_taylor2:.4f} Gt CO2e  |  Error: {abs(C_actual - C_taylor2):.6f} Gt")

    val = model.validate_taylor()
    print(f"\n  Monte Carlo Validation (100 samples, ±20% perturbation):")
    print(f"  1st-order R²: {val['R2_first_order']:.6f}  |  MAE: {val['MAE_first_order']:.6f} Gt")
    print(f"  2nd-order R²: {val['R2_second_order']:.6f}  |  MAE: {val['MAE_second_order']:.6f} Gt")

    # ----------------------------------------------------------------
    # HESSIAN MATRIX
    # ----------------------------------------------------------------
    H = model.compute_hessian()
    eigenvalues = np.linalg.eigvalsh(H)

    print("\n5. HESSIAN MATRIX ANALYSIS")
    print("-" * 70)
    print(f"  Hessian (all zeros for linear model - no curvature)")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Model is {'convex' if all(eigenvalues >= 0) else 'non-convex'}")

    # ----------------------------------------------------------------
    # SCENARIO ANALYSIS
    # ----------------------------------------------------------------
    scenarios = {
        'Baseline': x0.copy(),
        '10% reduction - Electricity only': x0.copy(),
        '10% reduction - Transport only': x0.copy(),
        '10% reduction - Top 3 sectors': x0.copy(),
        '20% Electricity + 10% Transport': x0.copy(),
        '20% reduction - All sectors': x0.copy(),
    }

    scenarios['10% reduction - Electricity only'][0] *= 0.9
    scenarios['10% reduction - Transport only'][1] *= 0.9
    scenarios['10% reduction - Top 3 sectors'][0] *= 0.9
    scenarios['10% reduction - Top 3 sectors'][1] *= 0.9
    scenarios['10% reduction - Top 3 sectors'][2] *= 0.9
    scenarios['20% Electricity + 10% Transport'][0] *= 0.8
    scenarios['20% Electricity + 10% Transport'][1] *= 0.9
    scenarios['20% reduction - All sectors'] *= 0.8

    df_scenarios = model.scenario_analysis(scenarios)

    print("\n6. SCENARIO ANALYSIS")
    print("-" * 70)
    print(df_scenarios.to_string(index=False))

    # ----------------------------------------------------------------
    # EXPORT RESULTS
    # ----------------------------------------------------------------
    print("\n7. EXPORTING RESULTS")
    print("-" * 70)

    df_sensitivity.to_csv('sensitivity_results.csv', index=False)
    print("  ✓ sensitivity_results.csv")

    df_scenarios.to_csv('scenarios_results.csv', index=False)
    print("  ✓ scenarios_results.csv")

    # Validation CSV
    pd.DataFrame({
        'Sector': model.var_names,
        'Gradient': gradient,
        'Sensitivity_Index': gradient * (x0 / C_baseline),
        'Emission_Factor': e,
        'Baseline_Contribution_Gt': contributions
    }).to_csv('validation_results.csv', index=False)
    print("  ✓ validation_results.csv")

    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)
    print(f"  Total Global Emissions:     {C_baseline:.2f} Gt CO2e")
    print(f"  Highest sensitivity sector: {df_sensitivity.iloc[0]['Sector']}")
    print(f"  Top 3 sectors contribute:   {df_sensitivity.head(3)['Percent_Impact'].sum():.1f}% of emission sensitivity")
    print(f"  Taylor approximation R²:    {val['R2_first_order']:.6f}")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
