#  Copyright © 2026 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for variogram visualization functions."""

import unittest

import numpy as np

from evo.objects.notebooks.variogram_plot import (
    _evaluate_structure,
    _generate_ellipsoid_wireframe,
    _get_variogram_data,
    _rotation_matrix,
    plot_ellipsoids_comparison,
    plot_search_ellipsoid,
    plot_variogram,
    plot_variogram_2d,
    plot_variogram_ellipsoids,
    plot_variogram_model,
)


class TestRotationMatrix(unittest.TestCase):
    """Tests for rotation matrix calculation."""

    def test_identity_rotation(self):
        """No rotation should give identity matrix."""
        rot = _rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(rot, np.eye(3))

    def test_dip_azimuth_90(self):
        """90 degree azimuth rotates X to Y."""
        rot = _rotation_matrix(90, 0, 0)
        # Apply to unit X vector
        result = rot @ np.array([1, 0, 0])
        # Should point in -Y direction (clockwise rotation)
        np.testing.assert_array_almost_equal(result, [0, -1, 0], decimal=5)

    def test_dip_90(self):
        """90 degree dip rotates Z to -Y."""
        rot = _rotation_matrix(0, 90, 0)
        result = rot @ np.array([0, 0, 1])
        np.testing.assert_array_almost_equal(result, [0, -1, 0], decimal=5)

    def test_rotation_is_orthogonal(self):
        """Rotation matrix should be orthogonal (R^T R = I)."""
        rot = _rotation_matrix(45, 30, 60)
        np.testing.assert_array_almost_equal(rot.T @ rot, np.eye(3), decimal=10)


class TestEllipsoidWireframe(unittest.TestCase):
    """Tests for ellipsoid wireframe generation."""

    def test_wireframe_shape(self):
        """Wireframe should return coordinate arrays."""
        ranges = (100, 50, 25)
        rotation = np.eye(3)
        x, y, z = _generate_ellipsoid_wireframe(ranges, rotation, n_points=20)

        self.assertEqual(len(x), len(y))
        self.assertEqual(len(y), len(z))
        self.assertGreater(len(x), 0)

    def test_wireframe_bounds(self):
        """Wireframe points should be within ellipsoid bounds."""
        ranges = (100, 50, 25)
        rotation = np.eye(3)
        x, y, z = _generate_ellipsoid_wireframe(ranges, rotation, n_points=30)

        # Filter out NaN separators
        valid = ~np.isnan(x)
        self.assertTrue(np.all(np.abs(x[valid]) <= ranges[0] * 1.01))
        self.assertTrue(np.all(np.abs(y[valid]) <= ranges[1] * 1.01))
        self.assertTrue(np.all(np.abs(z[valid]) <= ranges[2] * 1.01))


class TestEvaluateStructure(unittest.TestCase):
    """Tests for variogram structure evaluation."""

    def test_spherical_at_zero(self):
        """Spherical model should be 0 at origin."""
        h = np.array([0])
        gamma = _evaluate_structure("spherical", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [0])

    def test_spherical_at_range(self):
        """Spherical model should reach sill at range."""
        h = np.array([100])
        gamma = _evaluate_structure("spherical", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [1.0])

    def test_spherical_beyond_range(self):
        """Spherical model should stay at sill beyond range."""
        h = np.array([150, 200, 500])
        gamma = _evaluate_structure("spherical", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [1.0, 1.0, 1.0])

    def test_exponential_approaches_sill(self):
        """Exponential model should approach sill asymptotically."""
        h = np.array([0, 100, 300])
        gamma = _evaluate_structure("exponential", h, contribution=1.0, range_val=100)

        # At origin, should be 0
        self.assertAlmostEqual(gamma[0], 0, places=5)
        # At range, should be close to sill (practical range is ~3x range)
        self.assertGreater(gamma[1], 0.5)
        # At 3x range, should be very close to sill
        self.assertAlmostEqual(gamma[2], 1.0, places=2)

    def test_gaussian_smooth_at_origin(self):
        """Gaussian model should have smooth behavior near origin."""
        h = np.array([0, 1, 5, 10])
        gamma = _evaluate_structure("gaussian", h, contribution=1.0, range_val=100)

        # At origin, should be 0
        self.assertAlmostEqual(gamma[0], 0, places=5)
        # Near origin, should increase slowly (parabolic)
        self.assertLess(gamma[1], 0.01)

    def test_cubic_reaches_sill(self):
        """Cubic model should reach sill at range."""
        h = np.array([100, 150])
        gamma = _evaluate_structure("cubic", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [1.0, 1.0])

    def test_linear_unbounded(self):
        """Linear model should increase without bound."""
        h = np.array([50, 100, 200])
        gamma = _evaluate_structure("linear", h, contribution=1.0, range_val=100)

        self.assertAlmostEqual(gamma[0], 0.5)
        self.assertAlmostEqual(gamma[1], 1.0)
        self.assertAlmostEqual(gamma[2], 2.0)


class TestGetVariogramData(unittest.TestCase):
    """Tests for variogram data extraction."""

    def test_from_dict(self):
        """Should extract data from dict."""
        variogram_dict = {
            "sill": 10.0,
            "nugget": 2.0,
            "structures": [{"variogram_type": "spherical", "contribution": 8.0}],
        }
        sill, nugget, structures, attribute = _get_variogram_data(variogram_dict)

        self.assertEqual(sill, 10.0)
        self.assertEqual(nugget, 2.0)
        self.assertEqual(len(structures), 1)
        self.assertIsNone(attribute)

    def test_from_dict_no_nugget(self):
        """Should default nugget to 0 if not present."""
        variogram_dict = {
            "sill": 10.0,
            "structures": [],
        }
        sill, nugget, structures, attribute = _get_variogram_data(variogram_dict)

        self.assertEqual(nugget, 0.0)

    def test_from_dict_with_attribute(self):
        """Should extract attribute if present."""
        variogram_dict = {
            "sill": 10.0,
            "structures": [],
            "attribute": "grade",
        }
        sill, nugget, structures, attribute = _get_variogram_data(variogram_dict)

        self.assertEqual(attribute, "grade")


class TestPlotFunctions(unittest.TestCase):
    """Tests for plot generation functions."""

    @classmethod
    def setUpClass(cls):
        """Create a sample variogram for testing."""
        cls.variogram = {
            "sill": 10.0,
            "nugget": 2.0,
            "structures": [
                {
                    "variogram_type": "spherical",
                    "contribution": 5.0,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 200, "semi_major": 150, "minor": 100},
                        "rotation": {"dip_azimuth": 45, "dip": 30, "pitch": 0},
                    },
                },
                {
                    "variogram_type": "exponential",
                    "contribution": 3.0,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 500, "semi_major": 400, "minor": 200},
                        "rotation": {"dip_azimuth": 0, "dip": 0, "pitch": 0},
                    },
                },
            ],
        }

    def test_plot_ellipsoids_returns_figure(self):
        """plot_variogram_ellipsoids should return a Plotly Figure."""
        import plotly.graph_objects as go

        fig = plot_variogram_ellipsoids(self.variogram)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_ellipsoids_has_traces(self):
        """plot_variogram_ellipsoids should have traces for each structure."""
        fig = plot_variogram_ellipsoids(self.variogram)
        # Should have at least 2 traces (one per structure, plus axes)
        self.assertGreaterEqual(len(fig.data), 2)

    def test_plot_ellipsoids_surface_mode(self):
        """plot_variogram_ellipsoids with surface=True should include surfaces."""
        fig = plot_variogram_ellipsoids(self.variogram, surface=True)
        # Should have surface traces
        surface_traces = [t for t in fig.data if hasattr(t, "colorscale")]
        self.assertGreater(len(surface_traces), 0)

    def test_plot_model_returns_figure(self):
        """plot_variogram_model should return a tuple of Plotly Figures."""
        import plotly.graph_objects as go

        result = plot_variogram_model(self.variogram)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        for fig in result:
            self.assertIsInstance(fig, go.Figure)

    def test_plot_model_has_traces(self):
        """plot_variogram_model should have traces for directional curves."""
        combined, minor, semi_major, major = plot_variogram_model(self.variogram)
        # Combined should have traces for 3 directions
        self.assertGreaterEqual(len(combined.data), 3)
        # Individual plots should have curve + sill line
        self.assertGreaterEqual(len(minor.data), 1)

    def test_plot_model_custom_max_lag(self):
        """plot_variogram_model should respect max_lag parameter."""
        combined, _, _, _ = plot_variogram_model(self.variogram, max_lag=1000)
        # Check that x range extends to max_lag
        self.assertEqual(combined.layout.xaxis.range[1], 1000)

    def test_plot_2d_returns_tuple(self):
        """plot_variogram_2d should return a tuple of 4 Plotly Figures."""
        import plotly.graph_objects as go

        result = plot_variogram_2d(self.variogram)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        for fig in result:
            self.assertIsInstance(fig, go.Figure)

    def test_plot_2d_has_directional_curves(self):
        """plot_variogram_2d should have curves for each direction in combined plot."""
        combined, minor, semi_major, major = plot_variogram_2d(self.variogram)
        # Combined plot should have Minor, Semi-major, Major curves
        curve_names = [t.name for t in combined.data if t.name]
        self.assertIn("Minor", curve_names)
        self.assertIn("Semi-major", curve_names)
        self.assertIn("Major", curve_names)

    def test_plot_2d_individual_plots(self):
        """plot_variogram_2d should return individual directional plots."""
        combined, minor, semi_major, major = plot_variogram_2d(self.variogram)
        # Check individual plot titles contain direction names
        self.assertIn("Minor", minor.layout.title.text)
        self.assertIn("Semi-major", semi_major.layout.title.text)
        self.assertIn("Major", major.layout.title.text)

    def test_plot_variogram_returns_tuple(self):
        """plot_variogram should return a tuple of 5 Plotly Figures."""
        import plotly.graph_objects as go

        result = plot_variogram(self.variogram)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 5)
        for fig in result:
            self.assertIsInstance(fig, go.Figure)

    def test_plot_variogram_has_both_views(self):
        """plot_variogram should return figures with traces."""
        combined, minor, semi_major, major, ellipsoids = plot_variogram(self.variogram)
        # Combined 2D should have directional curves
        self.assertGreaterEqual(len(combined.data), 3)
        # 3D should have ellipsoid traces
        self.assertGreaterEqual(len(ellipsoids.data), 2)

    def test_plot_variogram_with_title(self):
        """plot_variogram should accept custom title."""
        combined, _, _, _, _ = plot_variogram(self.variogram, title="My Custom Variogram")
        self.assertEqual(combined.layout.title.text, "My Custom Variogram")


class TestPlotWithTypedVariogram(unittest.TestCase):
    """Tests for plot functions with typed Variogram objects."""

    def test_plot_with_variogram_like_object(self):
        """Should work with objects that have sill, nugget, structures properties."""

        class MockVariogram:
            sill = 5.0
            nugget = 1.0
            structures = [
                {
                    "variogram_type": "spherical",
                    "contribution": 4.0,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 100, "semi_major": 80, "minor": 50},
                        "rotation": {"dip_azimuth": 0, "dip": 0, "pitch": 0},
                    },
                }
            ]

        fig = plot_variogram_ellipsoids(MockVariogram())
        self.assertGreater(len(fig.data), 0)


class TestSearchEllipsoidPlots(unittest.TestCase):
    """Tests for search ellipsoid plotting functions."""

    @classmethod
    def setUpClass(cls):
        """Create sample data for testing."""
        cls.variogram = {
            "sill": 10.0,
            "nugget": 2.0,
            "structures": [
                {
                    "variogram_type": "spherical",
                    "contribution": 8.0,
                    "anisotropy": {
                        "ellipsoid_ranges": {"major": 200, "semi_major": 150, "minor": 100},
                        "rotation": {"dip_azimuth": 45, "dip": 30, "pitch": 0},
                    },
                },
            ],
        }
        cls.search_ellipsoid_dict = {
            "ranges": {"major": 250, "semi_major": 200, "minor": 150},
            "rotation": {"dip_azimuth": 0, "dip": 0, "pitch": 0},
        }

    def test_plot_search_ellipsoid_from_dict(self):
        """plot_search_ellipsoid should work with dict input."""
        import plotly.graph_objects as go

        fig = plot_search_ellipsoid(self.search_ellipsoid_dict)
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)

    def test_plot_search_ellipsoid_has_wireframe(self):
        """plot_search_ellipsoid should have a wireframe trace."""
        fig = plot_search_ellipsoid(self.search_ellipsoid_dict)
        # Should have wireframe trace
        wireframe_traces = [t for t in fig.data if "Search Ellipsoid" in (t.name or "")]
        self.assertEqual(len(wireframe_traces), 1)

    def test_plot_ellipsoids_comparison_returns_figure(self):
        """plot_ellipsoids_comparison should return a Plotly Figure."""
        import plotly.graph_objects as go

        fig = plot_ellipsoids_comparison(self.variogram, self.search_ellipsoid_dict)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_ellipsoids_comparison_has_both(self):
        """plot_ellipsoids_comparison should have both variogram and search ellipsoids."""
        fig = plot_ellipsoids_comparison(self.variogram, self.search_ellipsoid_dict)
        trace_names = [t.name for t in fig.data if t.name]
        # Should have variogram structure traces and search ellipsoid trace
        has_variogram = any("spherical" in name.lower() for name in trace_names)
        has_search = any("search" in name.lower() for name in trace_names)
        self.assertTrue(has_variogram)
        self.assertTrue(has_search)

    def test_plot_search_ellipsoid_custom_color(self):
        """plot_search_ellipsoid should accept custom color."""
        fig = plot_search_ellipsoid(self.search_ellipsoid_dict, color="#FF0000")
        # Verify the trace uses the custom color
        wireframe_trace = [t for t in fig.data if "Search" in (t.name or "")][0]
        self.assertEqual(wireframe_trace.line.color, "#FF0000")


if __name__ == "__main__":
    unittest.main()
