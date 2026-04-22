"""
Carbon Footprint Modeling using Taylor Series Expansion
========================================================

Course: Mathematics for Intelligent Systems 2 (25MAT116)
Topic: Taylor Series for Carbon Footprint Modeling - Emission Sensitivity Analysis

Description:
    This module implements a comprehensive carbon footprint model using
    Taylor series expansion (1st, 2nd, AND 3rd order) to perform emission
    sensitivity analysis across seven major global emission sectors.

Mathematical Framework:
    C(x) = Σ(ei * xi) + 0.5 * Σ(ai * xi²) + 0.5 * x' * B * x

    Where:
        x  = Activity level vector [x1, x2, ..., x7]
        e  = Linear emission factors
        a  = Quadratic coefficients
        B  = Symmetric interaction matrix (7x7)

Taylor Series Expansion (UPGRADED - 3 Orders):
    1st Order:  C(x) ≈ C(x0) + ∇C(x0)·(x - x0)
    2nd Order:  C(x) ≈ C(x0) + ∇C(x0)·(x - x0) + 0.5*(x-x0)'·H·(x-x0)
    3rd Order:  C(x) ≈ [2nd Order] + (1/6) * Σ_ijk T_ijk * dxi * dxj * dxk
                Where T_ijk are third-order tensor coefficients (cubic terms)

Lagrange Remainder Bound:
    |R_n(x)| ≤ M / (n+1)! * ||x - x0||^(n+1)
    Where M bounds the (n+1)-th derivative

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
from itertools import combinations_with_replacement


# ============================================================================
# CARBON FOOTPRINT MODEL CLASS - UPGRADED WITH 3RD ORDER TAYLOR
# ============================================================================

class CarbonFootprintModel:
    """
    Carbon footprint model using Taylor series expansion (1st, 2nd, 3rd order)
    for global emission sensitivity analysis.

    NEW in this version:
    - 3rd order Taylor expansion via cubic emission interaction tensor
    - Lagrange remainder bounds for each order
    - Convergence analysis across orders
    - Multi-point Taylor expansion comparison

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
    T_tensor : np.ndarray
        3rd order tensor (7x7x7) for cubic corrections
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

        self.e = None
        self.a = None
        self.b = None
        self.x0 = None
        self.B = None
        self.T_tensor = None          # 3rd order tensor (NEW)
        self.cubic_coeffs = None      # Diagonal cubic terms (NEW)

    def set_parameters(self, e, a, b, x0, cubic_coeffs=None):
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
        cubic_coeffs : array-like, optional
            Cubic (3rd order) diagonal coefficients (length 7).
            If None, small physically motivated values are used.
        """
        self.e = np.array(e, dtype=float)
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float) if b is not None else np.zeros(21)
        self.x0 = np.array(x0, dtype=float)
        self.B = self._build_interaction_matrix(self.b)

        # 3rd order cubic diagonal coefficients
        # Physical interpretation: diminishing/increasing returns on each sector
        # Signs: positive = accelerating growth, negative = diminishing returns
        if cubic_coeffs is not None:
            self.cubic_coeffs = np.array(cubic_coeffs, dtype=float)
        else:
            # Physically motivated small cubic corrections (6th order of magnitude
            # smaller than linear terms to ensure series converges properly)
            self.cubic_coeffs = np.array([
                1.2e-13,   # Electricity: slight accelerating return
               -2.0e-10,   # Transport: diminishing returns at high consumption
                3.5e-10,   # Industrial: accelerating at high output
                1.5e-16,   # Agriculture: very small cubic effect
               -1.0e-10,   # Buildings: diminishing returns
                2.0e-12,   # Waste: slight acceleration
               -5.0e-13,   # Other: diminishing
            ])

        # Build 3rd order tensor (symmetric, diagonal-dominant)
        self.T_tensor = self._build_cubic_tensor(self.cubic_coeffs)

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
                B[j, i] = b[idx]
                idx += 1
        return B

    def _build_cubic_tensor(self, cubic_diag):
        """
        Build symmetric 3rd order tensor T_ijk from diagonal coefficients.

        For simplicity and physical interpretability, we use a diagonally
        dominant tensor: T_iii = cubic_diag[i], off-diagonal terms are
        1/10th of the geometric mean of involved diagonal terms.

        Parameters
        ----------
        cubic_diag : np.ndarray
            Diagonal cubic coefficients (length 7)

        Returns
        -------
        T : np.ndarray
            Symmetric 3rd order tensor of shape (7, 7, 7)
        """
        n = 7
        T = np.zeros((n, n, n))

        # Diagonal terms: T_iii
        for i in range(n):
            T[i, i, i] = cubic_diag[i]

        # Off-diagonal terms (symmetric, small relative to diagonal)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    # Geometric mean of involved diagonal terms, scaled down
                    val = 0.05 * np.cbrt(abs(cubic_diag[i]) *
                                         abs(cubic_diag[j]) *
                                         abs(cubic_diag[k]))
                    val *= np.sign(cubic_diag[i])
                    # All 6 permutations of (i,j,k) get same value (symmetry)
                    for pi, pj, pk in [(i,j,k),(i,k,j),(j,i,k),
                                        (j,k,i),(k,i,j),(k,j,i)]:
                        T[pi, pj, pk] = val

        return T

    # ======================================================================
    # EMISSION COMPUTATION
    # ======================================================================

    def compute_emissions(self, x):
        """
        Calculate total carbon emissions C(x).

        C(x) = Σ(ei*xi) + 0.5*Σ(ai*xi²) + 0.5*x'*B*x + (1/6)*Σ_ijk T_ijk*xi*xj*xk

        The cubic term provides the 3rd-order nonlinearity that our
        higher-order Taylor series will capture.

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

        # Quadratic diagonal terms: 0.5 * Σ(ai * xi²)
        C += 0.5 * np.dot(self.a, x ** 2)

        # Quadratic interaction terms: 0.5 * x' * B * x
        C += 0.5 * x @ self.B @ x

        # Cubic terms: (1/6) * Σ_ijk T_ijk * xi * xj * xk
        C += (1.0 / 6.0) * np.einsum('ijk,i,j,k', self.T_tensor, x, x, x)

        return C

    def compute_gradient(self, x):
        """
        Calculate gradient vector ∇C(x).

        ∂C/∂xi = ei + ai*xi + Σj(Bij*xj) + (1/2)*Σ_jk T_ijk * xj * xk

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

        # Linear: e
        grad = self.e.copy()

        # Quadratic diagonal: a * x
        grad += self.a * x

        # Quadratic interaction: B * x
        grad += self.B @ x

        # Cubic contribution to gradient: (1/2) * Σ_jk T_ijk * xj * xk
        grad += 0.5 * np.einsum('ijk,j,k->i', self.T_tensor, x, x)

        return grad

    def compute_hessian(self, x=None):
        """
        Calculate Hessian matrix H(x).

        H_ij(x) = a_i*δ_ij + B_ij + Σ_k T_ijk * xk

        UPGRADED: Hessian is now x-dependent due to cubic terms.

        Parameters
        ----------
        x : array-like, optional
            Activity levels at which to evaluate Hessian.
            If None, uses baseline x0.

        Returns
        -------
        np.ndarray
            Hessian matrix of shape (7, 7)
        """
        if x is None:
            x = self.x0
        x = np.array(x, dtype=float)

        # Base Hessian: diagonal quadratic + interaction matrix
        H = self.B.copy()
        np.fill_diagonal(H, np.diag(H) + self.a)

        # Cubic contribution to Hessian: Σ_k T_ijk * xk
        H += np.einsum('ijk,k->ij', self.T_tensor, x)

        return H

    def compute_third_order_tensor_at_x0(self):
        """
        Return the 3rd-order derivative tensor evaluated at x0.

        The 3rd order term in Taylor expansion is:
            (1/6) * Σ_ijk (∂³C/∂xi∂xj∂xk)|_x0 * dxi * dxj * dxk
            = (1/6) * Σ_ijk T_ijk * dxi * dxj * dxk

        For our cubic model, ∂³C/∂xi∂xj∂xk = T_ijk (constant).

        Returns
        -------
        np.ndarray
            3rd order tensor T_ijk of shape (7, 7, 7)
        """
        return self.T_tensor.copy()

    # ======================================================================
    # TAYLOR APPROXIMATIONS - ALL THREE ORDERS
    # ======================================================================

    def taylor_first_order(self, x, x0=None):
        """
        First-order Taylor series approximation.

        C(x) ≈ C(x0) + ∇C(x0) · (x - x0)

        Error bound (Lagrange): |R1| ≤ (M2/2) * ||dx||²
        Where M2 = max eigenvalue of H(x0)

        Parameters
        ----------
        x : array-like
            Query point
        x0 : array-like, optional
            Expansion point (uses self.x0 if None)

        Returns
        -------
        float
            First-order Taylor approximation of C(x)
        """
        if x0 is None:
            x0 = self.x0
        x = np.array(x, dtype=float)
        x0 = np.array(x0, dtype=float)

        C0 = self.compute_emissions(x0)
        grad0 = self.compute_gradient(x0)

        return C0 + np.dot(grad0, x - x0)

    def taylor_second_order(self, x, x0=None):
        """
        Second-order Taylor series approximation.

        C(x) ≈ C(x0) + ∇C(x0)·(x-x0) + 0.5*(x-x0)'·H(x0)·(x-x0)

        Error bound (Lagrange): |R2| ≤ (M3/6) * ||dx||³
        Where M3 bounds the 3rd derivatives

        Parameters
        ----------
        x : array-like
            Query point
        x0 : array-like, optional
            Expansion point (uses self.x0 if None)

        Returns
        -------
        float
            Second-order Taylor approximation of C(x)
        """
        if x0 is None:
            x0 = self.x0
        x = np.array(x, dtype=float)
        x0 = np.array(x0, dtype=float)
        dx = x - x0

        C0 = self.compute_emissions(x0)
        grad0 = self.compute_gradient(x0)
        H0 = self.compute_hessian(x0)

        return C0 + np.dot(grad0, dx) + 0.5 * dx @ H0 @ dx

    def taylor_third_order(self, x, x0=None):
        """
        Third-order Taylor series approximation.  ← NEW

        C(x) ≈ C(x0) + ∇C(x0)·dx + 0.5*dx'·H(x0)·dx
                      + (1/6)*Σ_ijk T_ijk * dxi * dxj * dxk

        Where T_ijk = ∂³C/∂xi∂xj∂xk (3rd partial derivative tensor)

        Error bound (Lagrange): |R3| ≤ (M4/24) * ||dx||⁴
        For our model, 4th derivatives = 0, so this is EXACT at 3rd order.

        Parameters
        ----------
        x : array-like
            Query point
        x0 : array-like, optional
            Expansion point (uses self.x0 if None)

        Returns
        -------
        float
            Third-order Taylor approximation of C(x)
        """
        if x0 is None:
            x0 = self.x0
        x = np.array(x, dtype=float)
        x0 = np.array(x0, dtype=float)
        dx = x - x0

        # Build on 2nd order
        C2 = self.taylor_second_order(x, x0)

        # Add 3rd order correction: (1/6) * Σ_ijk T_ijk * dxi * dxj * dxk
        cubic_correction = (1.0 / 6.0) * np.einsum('ijk,i,j,k',
                                                     self.T_tensor, dx, dx, dx)

        return C2 + cubic_correction

    def lagrange_remainder_bound(self, x, x0=None, order=1):
        """
        Compute Lagrange remainder bound for order n.

        |Rn(x)| ≤ M_(n+1) / (n+1)! * ||x - x0||^(n+1)

        Parameters
        ----------
        x : array-like
            Query point
        x0 : array-like, optional
            Expansion point
        order : int
            Taylor order (1, 2, or 3)

        Returns
        -------
        dict
            {'bound': float, 'dx_norm': float, 'M': float}
        """
        if x0 is None:
            x0 = self.x0
        x = np.array(x, dtype=float)
        x0 = np.array(x0, dtype=float)
        dx = x - x0
        dx_norm = np.linalg.norm(dx)

        if order == 1:
            # M2 = max eigenvalue of Hessian (bounds 2nd derivatives)
            H0 = self.compute_hessian(x0)
            M = np.max(np.abs(np.linalg.eigvalsh(H0)))
            factorial = 2  # 2! = 2
            bound = M / factorial * dx_norm ** 2

        elif order == 2:
            # M3 = max element of 3rd order tensor (bounds 3rd derivatives)
            M = np.max(np.abs(self.T_tensor))
            factorial = 6  # 3! = 6
            bound = M / factorial * dx_norm ** 3

        elif order == 3:
            # 4th derivatives = 0 for our cubic model → exact
            # Return machine epsilon * value as practical bound
            C_true = self.compute_emissions(x)
            M = 0.0
            bound = np.finfo(float).eps * abs(C_true)
            return {'bound': bound, 'dx_norm': dx_norm, 'M': M,
                    'note': 'Exact for cubic model (4th derivatives = 0)'}

        else:
            raise ValueError(f"order must be 1, 2, or 3. Got {order}")

        return {'bound': bound, 'dx_norm': dx_norm, 'M': M}

    def taylor_all_orders(self, x, x0=None):
        """
        Compute all three Taylor approximations and errors together.

        Parameters
        ----------
        x : array-like
            Query point
        x0 : array-like, optional
            Expansion point

        Returns
        -------
        dict
            All approximations, true value, errors, and remainder bounds
        """
        if x0 is None:
            x0 = self.x0

        true_val = self.compute_emissions(x)
        t1 = self.taylor_first_order(x, x0)
        t2 = self.taylor_second_order(x, x0)
        t3 = self.taylor_third_order(x, x0)

        r1 = self.lagrange_remainder_bound(x, x0, order=1)
        r2 = self.lagrange_remainder_bound(x, x0, order=2)
        r3 = self.lagrange_remainder_bound(x, x0, order=3)

        return {
            'true': true_val,
            'order1': t1,
            'order2': t2,
            'order3': t3,
            'error1': abs(true_val - t1),
            'error2': abs(true_val - t2),
            'error3': abs(true_val - t3),
            'rel_error1': abs(true_val - t1) / abs(true_val) * 100,
            'rel_error2': abs(true_val - t2) / abs(true_val) * 100,
            'rel_error3': abs(true_val - t3) / abs(true_val) * 100,
            'remainder_bound1': r1['bound'],
            'remainder_bound2': r2['bound'],
            'remainder_bound3': r3['bound'],
        }

    # ======================================================================
    # ANALYSIS METHODS
    # ======================================================================

    def sensitivity_analysis(self, perturbation_pct=10.0, x0=None):
        """
        Perform sensitivity analysis via parameter perturbation.

        Computes normalized sensitivity index:
            Si = (∂C/∂xi) * (xi / C)

        Parameters
        ----------
        perturbation_pct : float
            Percentage by which to perturb each parameter (default: 10%)
        x0 : array-like, optional
            Point at which to evaluate sensitivity

        Returns
        -------
        pd.DataFrame
            Sensitivity results sorted by impact
        """
        if x0 is None:
            x0 = self.x0
        x0 = np.array(x0, dtype=float)

        gradient = self.compute_gradient(x0)
        C_baseline = self.compute_emissions(x0)

        results = []
        for i in range(7):
            Si = gradient[i] * (x0[i] / C_baseline)

            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] *= (1 + perturbation_pct / 100)
            x_minus[i] *= (1 - perturbation_pct / 100)

            C_plus = self.compute_emissions(x_plus)
            C_minus = self.compute_emissions(x_minus)
            abs_impact = (C_plus - C_minus) / 2
            pct_impact = (abs_impact / C_baseline) * 100

            results.append({
                'Sector': self.var_names[i],
                'Unit': self.units[i],
                'Baseline_Activity': x0[i],
                'Gradient': gradient[i],
                'Sensitivity_Index': abs(Si),
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

    def validate_taylor(self, n_samples=500, max_perturbation=25):
        """
        Validate all three Taylor orders via Monte Carlo sampling.

        UPGRADED: Now validates 3rd order as well, and computes
        Lagrange remainder bounds for sampled points.

        Parameters
        ----------
        n_samples : int
            Number of random samples (default: 500)
        max_perturbation : float
            Maximum parameter perturbation percentage (default: 25%)

        Returns
        -------
        dict
            Validation metrics including R², MAE, and mean remainder bounds
        """
        np.random.seed(42)
        actual, pred1, pred2, pred3 = [], [], [], []
        bounds1, bounds2, bounds3 = [], [], []

        for _ in range(n_samples):
            x_test = self.x0 * (1 + np.random.uniform(
                -max_perturbation / 100,
                max_perturbation / 100,
                size=7
            ))
            x_test = np.maximum(x_test, 0)

            results = self.taylor_all_orders(x_test, self.x0)
            actual.append(results['true'])
            pred1.append(results['order1'])
            pred2.append(results['order2'])
            pred3.append(results['order3'])
            bounds1.append(results['remainder_bound1'])
            bounds2.append(results['remainder_bound2'])

        actual = np.array(actual)
        pred1 = np.array(pred1)
        pred2 = np.array(pred2)
        pred3 = np.array(pred3)

        ss_tot = np.sum((actual - actual.mean()) ** 2)

        def r2(pred):
            return 1 - np.sum((actual - pred) ** 2) / (ss_tot + 1e-30)

        return {
            'n_samples': n_samples,
            'max_perturbation_pct': max_perturbation,
            'R2_first_order': round(r2(pred1), 8),
            'R2_second_order': round(r2(pred2), 8),
            'R2_third_order': round(r2(pred3), 8),
            'MAE_first_order': round(np.mean(np.abs(actual - pred1)), 8),
            'MAE_second_order': round(np.mean(np.abs(actual - pred2)), 8),
            'MAE_third_order': round(np.mean(np.abs(actual - pred3)), 8),
            'MaxAE_first_order': round(np.max(np.abs(actual - pred1)), 8),
            'MaxAE_second_order': round(np.max(np.abs(actual - pred2)), 8),
            'MaxAE_third_order': round(np.max(np.abs(actual - pred3)), 8),
            'Mean_Lagrange_Bound_1st': round(np.mean(bounds1), 8),
            'Mean_Lagrange_Bound_2nd': round(np.mean(bounds2), 8),
        }

    def convergence_analysis(self, perturbation_range=None):
        """
        Analyse how quickly each Taylor order converges as perturbation grows.

        Shows the mathematical convergence: 3rd order error ~ O(||dx||^4),
        2nd order ~ O(||dx||^3), 1st order ~ O(||dx||^2).

        Parameters
        ----------
        perturbation_range : array-like, optional
            Percentage perturbations to test (default: 1% to 50%)

        Returns
        -------
        pd.DataFrame
            Convergence data for all three orders
        """
        if perturbation_range is None:
            perturbation_range = np.linspace(1, 50, 80)

        np.random.seed(0)
        results = []

        for pct in perturbation_range:
            # Average over 50 random directions at each perturbation level
            e1_list, e2_list, e3_list, b1_list, b2_list = [], [], [], [], []
            for _ in range(50):
                direction = np.random.randn(7)
                direction = direction / np.linalg.norm(direction)
                x_test = self.x0 + (pct / 100) * self.x0 * direction
                x_test = np.maximum(x_test, 0)

                r = self.taylor_all_orders(x_test, self.x0)
                e1_list.append(r['error1'])
                e2_list.append(r['error2'])
                e3_list.append(r['error3'])

                rb1 = self.lagrange_remainder_bound(x_test, self.x0, order=1)
                rb2 = self.lagrange_remainder_bound(x_test, self.x0, order=2)
                b1_list.append(rb1['bound'])
                b2_list.append(rb2['bound'])

            results.append({
                'perturbation_pct': pct,
                'error_1st': np.mean(e1_list),
                'error_2nd': np.mean(e2_list),
                'error_3rd': np.mean(e3_list),
                'bound_1st': np.mean(b1_list),
                'bound_2nd': np.mean(b2_list),
                'dx_norm': np.mean([np.linalg.norm((pct/100)*self.x0*d/np.linalg.norm(d))
                                    for d in [np.random.randn(7) for _ in range(10)]])
            })

        return pd.DataFrame(results)

    def multipoint_comparison(self, x, expansion_points_pct=None):
        """
        Compare Taylor 3rd order accuracy from multiple expansion points.

        Demonstrates that choosing x0 closer to x gives better accuracy,
        and that 3rd order captures nonlinearity better than 1st/2nd.

        Parameters
        ----------
        x : array-like
            Target point to approximate
        expansion_points_pct : list of float, optional
            Expansion points as % of x (e.g. [80, 90, 100, 110, 120])

        Returns
        -------
        pd.DataFrame
            Comparison of all orders across expansion points
        """
        if expansion_points_pct is None:
            expansion_points_pct = [70, 80, 90, 100, 110, 120]

        x = np.array(x, dtype=float)
        true_val = self.compute_emissions(x)
        results = []

        for pct in expansion_points_pct:
            x0_test = self.x0 * (pct / 100)
            r = self.taylor_all_orders(x, x0_test)
            results.append({
                'expansion_point_pct': pct,
                'true': r['true'],
                'order1': r['order1'],
                'order2': r['order2'],
                'order3': r['order3'],
                'rel_error1': r['rel_error1'],
                'rel_error2': r['rel_error2'],
                'rel_error3': r['rel_error3'],
            })

        return pd.DataFrame(results)


# ============================================================================
# BASELINE DATA (Global Emissions 2024-2025)
# Sources: IEA 2024, FAO 2024, UNEP 2024, IPCC AR6
# ============================================================================

BASELINE_ACTIVITIES = {
    'electricity_twh': 30500,
    'transport_mbd': 102,
    'industrial_ej': 185,
    'agriculture_pcal': 52800,
    'buildings_ej': 135,
    'waste_mt': 2100,
    'other_index': 1.0
}

EMISSION_FACTORS = {
    'electricity': 0.000512,
    'transport': 0.082344,
    'industrial': 0.035100,
    'agriculture': 0.000112,
    'buildings': 0.023700,
    'waste': 0.000810,
    'other': 6.400000
}


# ============================================================================
# MAIN EXECUTION - DEMONSTRATES NEW CAPABILITIES
# ============================================================================

def main():
    print("=" * 70)
    print("CARBON FOOTPRINT MODELING - HIGHER ORDER TAYLOR SERIES")
    print("25MAT116: Mathematics for Intelligent Systems 2")
    print("UPGRADED: 1st, 2nd, AND 3rd Order Taylor Expansion")
    print("=" * 70)

    model = CarbonFootprintModel()
    x0 = np.array(list(BASELINE_ACTIVITIES.values()))
    e = np.array(list(EMISSION_FACTORS.values()))

    model.set_parameters(e, np.zeros(7), np.zeros(21), x0)

    C_baseline = model.compute_emissions(x0)
    contributions = e * x0

    # ------------------------------------------------------------------
    # 1. BASELINE
    # ------------------------------------------------------------------
    print("\n1. BASELINE EMISSIONS BREAKDOWN")
    print("-" * 70)
    print(f"{'Sector':<25} {'Activity':>12} {'Factor':>12} {'Emissions (Gt)':>15}")
    print("-" * 70)
    for i in range(7):
        print(f"{model.var_names[i]:<25} {x0[i]:>12,.1f} "
              f"{e[i]:>12.6f} {contributions[i]:>15.4f}")
    print("-" * 70)
    print(f"{'TOTAL':.<25} {'':>12} {'':>12} {C_baseline:>15.4f} Gt CO2e")

    # ------------------------------------------------------------------
    # 2. ALL THREE TAYLOR ORDERS vs ACTUAL
    # ------------------------------------------------------------------
    print("\n2. HIGHER-ORDER TAYLOR APPROXIMATION COMPARISON")
    print("-" * 70)

    test_cases = [
        ("5%  increase all", x0 * 1.05),
        ("10% increase all", x0 * 1.10),
        ("20% increase all", x0 * 1.20),
        ("30% increase all", x0 * 1.30),
        ("50% increase all", x0 * 1.50),
        ("Mixed change",     x0 * np.array([1.15, 0.90, 1.25, 1.10, 0.85, 1.20, 0.95])),
    ]

    print(f"\n{'Case':<22} {'Actual':>10} {'1st Ord':>10} {'2nd Ord':>10} "
          f"{'3rd Ord':>10} {'Err1%':>8} {'Err2%':>8} {'Err3%':>8}")
    print("-" * 95)

    for label, x_test in test_cases:
        r = model.taylor_all_orders(x_test, x0)
        print(f"{label:<22} {r['true']:>10.4f} {r['order1']:>10.4f} "
              f"{r['order2']:>10.4f} {r['order3']:>10.4f} "
              f"{r['rel_error1']:>8.4f} {r['rel_error2']:>8.4f} "
              f"{r['rel_error3']:>8.4f}")

    # ------------------------------------------------------------------
    # 3. LAGRANGE REMAINDER BOUNDS
    # ------------------------------------------------------------------
    print("\n3. LAGRANGE REMAINDER BOUNDS")
    print("-" * 70)
    print(f"\n  |Rn(x)| ≤ M_(n+1) / (n+1)! × ||dx||^(n+1)")
    print()

    x_test = x0 * 1.20
    for order in [1, 2, 3]:
        rb = model.lagrange_remainder_bound(x_test, x0, order=order)
        actual_err = abs(model.compute_emissions(x_test) -
                         [model.taylor_first_order,
                          model.taylor_second_order,
                          model.taylor_third_order][order-1](x_test, x0))
        print(f"  Order {order}: Lagrange Bound = {rb['bound']:.6e} Gt   "
              f"| Actual Error = {actual_err:.6e} Gt   "
              f"| Bound holds: {'✓' if rb['bound'] >= actual_err - 1e-15 else '✗'}")

    # ------------------------------------------------------------------
    # 4. MONTE CARLO VALIDATION (all three orders)
    # ------------------------------------------------------------------
    print("\n4. MONTE CARLO VALIDATION (500 samples, ±25% perturbation)")
    print("-" * 70)
    val = model.validate_taylor(n_samples=500, max_perturbation=25)

    print(f"\n  {'Metric':<30} {'1st Order':>14} {'2nd Order':>14} {'3rd Order':>14}")
    print(f"  {'-'*74}")
    print(f"  {'R²':<30} {val['R2_first_order']:>14.8f} "
          f"{val['R2_second_order']:>14.8f} {val['R2_third_order']:>14.8f}")
    print(f"  {'Mean Abs Error (Gt)':<30} {val['MAE_first_order']:>14.8f} "
          f"{val['MAE_second_order']:>14.8f} {val['MAE_third_order']:>14.8f}")
    print(f"  {'Max Abs Error (Gt)':<30} {val['MaxAE_first_order']:>14.8f} "
          f"{val['MaxAE_second_order']:>14.8f} {val['MaxAE_third_order']:>14.8f}")

    # ------------------------------------------------------------------
    # 5. SENSITIVITY ANALYSIS
    # ------------------------------------------------------------------
    df_sensitivity = model.sensitivity_analysis(perturbation_pct=10)
    print("\n5. SENSITIVITY ANALYSIS (10% Perturbation)")
    print("-" * 70)
    print(df_sensitivity[['Sector', 'Sensitivity_Index',
                           'Absolute_Impact_Gt', 'Percent_Impact']].to_string(index=False))

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - See higher_order_taylor.py for full visualizations")
    print("=" * 70)


if __name__ == "__main__":
    main()
