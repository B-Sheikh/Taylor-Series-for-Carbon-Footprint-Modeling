"""
Test Suite - Carbon Footprint Taylor Series Model
==================================================

Course: Mathematics for Intelligent Systems 2 (25MAT116)
Topic: Taylor Series for Carbon Footprint Modeling

Tests:
    - Model initialization and parameter shapes
    - Emission calculations (linear, quadratic, interaction)
    - Gradient accuracy (analytical vs numerical)
    - Taylor approximation accuracy (1st and 2nd order)
    - Sensitivity analysis
    - Scenario analysis
    - Hessian symmetry
    - Edge cases
"""

import numpy as np
import pytest
from carbon_model import CarbonFootprintModel, BASELINE_ACTIVITIES, EMISSION_FACTORS


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def model():
    """Initialize model with global emissions data."""
    m = CarbonFootprintModel()
    x0 = np.array(list(BASELINE_ACTIVITIES.values()))
    e = np.array(list(EMISSION_FACTORS.values()))
    a = np.zeros(7)
    b = np.zeros(21)
    m.set_parameters(e, a, b, x0)
    return m


@pytest.fixture
def baseline(model):
    """Return baseline activity vector."""
    return model.x0.copy()


# ============================================================================
# 1. MODEL INITIALIZATION
# ============================================================================

class TestInitialization:

    def test_var_names_count(self, model):
        """Model should have exactly 7 emission sectors."""
        assert len(model.var_names) == 7

    def test_units_count(self, model):
        """Model should have exactly 7 unit definitions."""
        assert len(model.units) == 7

    def test_parameter_shapes(self, model):
        """All parameter arrays should have correct shapes."""
        assert model.e.shape == (7,)
        assert model.a.shape == (7,)
        assert model.b.shape == (21,)
        assert model.x0.shape == (7,)
        assert model.B.shape == (7, 7)

    def test_emission_factors_positive(self, model):
        """All emission factors should be positive."""
        assert np.all(model.e > 0)

    def test_baseline_positive(self, model):
        """All baseline activity levels should be positive."""
        assert np.all(model.x0 > 0)


# ============================================================================
# 2. EMISSION CALCULATIONS
# ============================================================================

class TestEmissionCalculations:

    def test_baseline_emissions_positive(self, model, baseline):
        """Total emissions should be positive."""
        C = model.compute_emissions(baseline)
        assert C > 0

    def test_baseline_emissions_value(self, model, baseline):
        """Baseline emissions should match sum of linear contributions."""
        C = model.compute_emissions(baseline)
        expected = np.dot(model.e, baseline)
        assert np.isclose(C, expected, rtol=1e-10)

    def test_zero_emissions_at_zero_activity(self, model):
        """Zero activity levels should yield zero emissions."""
        C = model.compute_emissions(np.zeros(7))
        assert np.isclose(C, 0.0, atol=1e-10)

    def test_linearity_scaling(self, model, baseline):
        """Doubling activity should double emissions (linear model)."""
        C_base = model.compute_emissions(baseline)
        C_double = model.compute_emissions(baseline * 2)
        assert np.isclose(C_double, 2 * C_base, rtol=1e-10)

    def test_individual_sector_contribution(self, model, baseline):
        """Each sector's contribution should equal factor × activity."""
        C_total = model.compute_emissions(baseline)
        C_manual = sum(model.e[i] * baseline[i] for i in range(7))
        assert np.isclose(C_total, C_manual, rtol=1e-10)


# ============================================================================
# 3. GRADIENT CALCULATIONS
# ============================================================================

class TestGradient:

    def test_gradient_shape(self, model, baseline):
        """Gradient should have shape (7,)."""
        grad = model.compute_gradient(baseline)
        assert grad.shape == (7,)

    def test_gradient_equals_emission_factors_linear(self, model, baseline):
        """For linear model (a=0, B=0), gradient should equal emission factors."""
        grad = model.compute_gradient(baseline)
        assert np.allclose(grad, model.e, rtol=1e-10)

    def test_gradient_numerical_validation(self, model, baseline):
        """Analytical gradient should match finite difference approximation."""
        epsilon = 1e-6
        analytical = model.compute_gradient(baseline)
        numerical = np.zeros(7)

        for i in range(7):
            x_plus = baseline.copy()
            x_minus = baseline.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            numerical[i] = (model.compute_emissions(x_plus) -
                            model.compute_emissions(x_minus)) / (2 * epsilon)

        assert np.allclose(analytical, numerical, rtol=1e-5)

    def test_gradient_all_positive(self, model, baseline):
        """Gradient components should all be positive (emissions increase with activity)."""
        grad = model.compute_gradient(baseline)
        assert np.all(grad > 0)


# ============================================================================
# 4. HESSIAN MATRIX
# ============================================================================

class TestHessian:

    def test_hessian_shape(self, model):
        """Hessian should be 7x7."""
        H = model.compute_hessian()
        assert H.shape == (7, 7)

    def test_hessian_symmetry(self, model):
        """Hessian should be symmetric: H_ij = H_ji."""
        H = model.compute_hessian()
        assert np.allclose(H, H.T, rtol=1e-10)

    def test_hessian_zero_linear_model(self, model):
        """Hessian should be zero for purely linear model."""
        H = model.compute_hessian()
        assert np.allclose(H, 0, atol=1e-10)


# ============================================================================
# 5. TAYLOR APPROXIMATIONS
# ============================================================================

class TestTaylorApproximation:

    def test_first_order_at_baseline(self, model, baseline):
        """First-order Taylor at baseline should equal actual emissions."""
        C_actual = model.compute_emissions(baseline)
        C_taylor = model.taylor_first_order(baseline, baseline)
        assert np.isclose(C_actual, C_taylor, rtol=1e-10)

    def test_second_order_at_baseline(self, model, baseline):
        """Second-order Taylor at baseline should equal actual emissions."""
        C_actual = model.compute_emissions(baseline)
        C_taylor = model.taylor_second_order(baseline, baseline)
        assert np.isclose(C_actual, C_taylor, rtol=1e-10)

    def test_first_order_exact_linear(self, model, baseline):
        """First-order Taylor should be exact for linear model."""
        for scale in [0.8, 0.9, 1.1, 1.2, 1.5]:
            x_test = baseline * scale
            C_actual = model.compute_emissions(x_test)
            C_taylor = model.taylor_first_order(x_test, baseline)
            assert np.isclose(C_actual, C_taylor, rtol=1e-10)

    def test_second_order_exact_linear(self, model, baseline):
        """Second-order Taylor should be exact for linear model."""
        for scale in [0.8, 0.9, 1.1, 1.2, 1.5]:
            x_test = baseline * scale
            C_actual = model.compute_emissions(x_test)
            C_taylor = model.taylor_second_order(x_test, baseline)
            assert np.isclose(C_actual, C_taylor, rtol=1e-10)

    def test_approximation_error_small_perturbation(self, model, baseline):
        """Taylor error should be very small for 5% perturbation."""
        x_test = baseline * 1.05
        C_actual = model.compute_emissions(x_test)
        C_taylor = model.taylor_first_order(x_test, baseline)
        error_pct = abs(C_actual - C_taylor) / C_actual * 100
        assert error_pct < 0.1  # Less than 0.1%


# ============================================================================
# 6. SENSITIVITY ANALYSIS
# ============================================================================

class TestSensitivity:

    def test_sensitivity_returns_dataframe(self, model):
        """Sensitivity analysis should return a DataFrame."""
        import pandas as pd
        df = model.sensitivity_analysis()
        assert isinstance(df, pd.DataFrame)

    def test_sensitivity_seven_rows(self, model):
        """Sensitivity DataFrame should have 7 rows (one per sector)."""
        df = model.sensitivity_analysis()
        assert len(df) == 7

    def test_sensitivity_sorted_descending(self, model):
        """Sensitivity indices should be sorted in descending order."""
        df = model.sensitivity_analysis()
        indices = df['Sensitivity_Index'].values
        assert all(indices[i] >= indices[i + 1] for i in range(len(indices) - 1))

    def test_sensitivity_all_positive(self, model):
        """All sensitivity indices should be positive."""
        df = model.sensitivity_analysis()
        assert np.all(df['Sensitivity_Index'] > 0)

    def test_electricity_highest_sensitivity(self, model):
        """Electricity sector should have the highest sensitivity index."""
        df = model.sensitivity_analysis()
        assert 'Electricity' in df.iloc[0]['Sector']


# ============================================================================
# 7. SCENARIO ANALYSIS
# ============================================================================

class TestScenario:

    def test_reduction_decreases_emissions(self, model, baseline):
        """Reducing activity should decrease total emissions."""
        C_base = model.compute_emissions(baseline)
        x_reduced = baseline.copy()
        x_reduced[0] *= 0.9  # 10% reduction in electricity
        C_reduced = model.compute_emissions(x_reduced)
        assert C_reduced < C_base

    def test_10pct_electricity_reduction(self, model, baseline):
        """10% electricity reduction should save ~1.56 Gt CO2e."""
        C_base = model.compute_emissions(baseline)
        x_reduced = baseline.copy()
        x_reduced[0] *= 0.9
        C_reduced = model.compute_emissions(x_reduced)
        reduction = C_base - C_reduced
        # Should be ~10% of electricity contribution
        expected = 0.1 * model.e[0] * baseline[0]
        assert np.isclose(reduction, expected, rtol=1e-5)

    def test_scenario_analysis_output(self, model, baseline):
        """Scenario analysis should return correct DataFrame structure."""
        import pandas as pd
        scenarios = {
            'Baseline': baseline.copy(),
            'Test Scenario': baseline * 0.9
        }
        df = model.scenario_analysis(scenarios)
        assert isinstance(df, pd.DataFrame)
        assert 'Scenario' in df.columns
        assert 'Total_Emissions_Gt' in df.columns
        assert 'Reduction_Gt' in df.columns

    def test_baseline_scenario_zero_reduction(self, model, baseline):
        """Baseline scenario should show zero reduction."""
        scenarios = {'Baseline': baseline.copy()}
        df = model.scenario_analysis(scenarios)
        assert df.iloc[0]['Reduction_Gt'] == 0.0


# ============================================================================
# 8. EDGE CASES
# ============================================================================

class TestEdgeCases:

    def test_large_perturbation(self, model, baseline):
        """Model should handle large perturbations gracefully."""
        x_large = baseline * 10
        C_large = model.compute_emissions(x_large)
        C_base = model.compute_emissions(baseline)
        # Linear model: 10x input = 10x output
        assert np.isclose(C_large, 10 * C_base, rtol=1e-10)

    def test_single_sector_perturbation(self, model, baseline):
        """Perturbing one sector should only affect that sector's contribution."""
        C_base = model.compute_emissions(baseline)
        for i in range(7):
            x_test = baseline.copy()
            x_test[i] *= 1.1
            C_test = model.compute_emissions(x_test)
            delta = C_test - C_base
            expected = 0.1 * model.e[i] * baseline[i]
            assert np.isclose(delta, expected, rtol=1e-5)

    def test_interaction_matrix_symmetry(self, model):
        """Interaction matrix should be symmetric."""
        assert np.allclose(model.B, model.B.T, rtol=1e-10)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
