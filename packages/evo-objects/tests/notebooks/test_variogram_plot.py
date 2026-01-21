#  Copyright © 2025 Bentley Systems, Incorporated
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Tests for ellipsoid and variogram data generation functions."""

import unittest

import numpy as np

from evo.objects.notebooks.ellipsoid import (
    EllipsoidData,
    VariogramCurveData,
    _evaluate_structure,
    _rotation_matrix,
    generate_ellipsoid_mesh,
    generate_ellipsoid_wireframe,
    generate_variogram_curves,
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
        result = rot @ np.array([1, 0, 0])
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

    def test_wireframe_returns_ellipsoid_data(self):
        """Wireframe should return EllipsoidData."""
        data = generate_ellipsoid_wireframe(ranges=(100, 50, 25))
        self.assertIsInstance(data, EllipsoidData)
        self.assertEqual(data.ranges, (100, 50, 25))

    def test_wireframe_shape(self):
        """Wireframe should return coordinate arrays of same length."""
        data = generate_ellipsoid_wireframe(ranges=(100, 50, 25), n_points=20)
        self.assertEqual(len(data.x), len(data.y))
        self.assertEqual(len(data.y), len(data.z))
        self.assertGreater(len(data.x), 0)

    def test_wireframe_bounds(self):
        """Wireframe points should be within ellipsoid bounds."""
        ranges = (100, 50, 25)
        data = generate_ellipsoid_wireframe(ranges=ranges, n_points=30)

        # Filter out NaN separators
        valid = ~np.isnan(data.x)
        self.assertTrue(np.all(np.abs(data.x[valid]) <= ranges[0] * 1.01))
        self.assertTrue(np.all(np.abs(data.y[valid]) <= ranges[1] * 1.01))
        self.assertTrue(np.all(np.abs(data.z[valid]) <= ranges[2] * 1.01))

    def test_wireframe_has_nan_separators(self):
        """Wireframe should have NaN values separating line segments."""
        data = generate_ellipsoid_wireframe(ranges=(100, 50, 25))
        self.assertTrue(np.any(np.isnan(data.x)))

    def test_wireframe_with_rotation(self):
        """Wireframe should apply rotation."""
        data_no_rot = generate_ellipsoid_wireframe(ranges=(100, 50, 25), rotation=(0, 0, 0))
        data_rot = generate_ellipsoid_wireframe(ranges=(100, 50, 25), rotation=(45, 30, 0))
        # Rotated should have different coordinates
        self.assertFalse(np.allclose(data_no_rot.x, data_rot.x, equal_nan=True))


class TestEllipsoidMesh(unittest.TestCase):
    """Tests for ellipsoid mesh generation."""

    def test_mesh_returns_ellipsoid_data(self):
        """Mesh should return EllipsoidData."""
        data = generate_ellipsoid_mesh(ranges=(100, 50, 25))
        self.assertIsInstance(data, EllipsoidData)
        self.assertEqual(data.ranges, (100, 50, 25))

    def test_mesh_is_2d_array(self):
        """Mesh should return 2D arrays for surface plotting."""
        data = generate_ellipsoid_mesh(ranges=(100, 50, 25), n_points=15)
        self.assertEqual(data.x.ndim, 2)
        self.assertEqual(data.y.ndim, 2)
        self.assertEqual(data.z.ndim, 2)
        self.assertEqual(data.x.shape, (15, 15))

    def test_mesh_bounds(self):
        """Mesh points should be within ellipsoid bounds."""
        ranges = (100, 50, 25)
        data = generate_ellipsoid_mesh(ranges=ranges)
        self.assertTrue(np.all(np.abs(data.x) <= ranges[0] * 1.01))
        self.assertTrue(np.all(np.abs(data.y) <= ranges[1] * 1.01))
        self.assertTrue(np.all(np.abs(data.z) <= ranges[2] * 1.01))


class TestEvaluateStructure(unittest.TestCase):
    """Tests for variogram structure evaluation."""

    def test_spherical_at_zero(self):
        """Spherical model should be 0 at origin."""
        h = np.array([0.0])
        gamma = _evaluate_structure("spherical", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [0])

    def test_spherical_at_range(self):
        """Spherical model should reach sill at range."""
        h = np.array([100.0])
        gamma = _evaluate_structure("spherical", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [1.0])

    def test_spherical_beyond_range(self):
        """Spherical model should stay at sill beyond range."""
        h = np.array([150.0, 200.0, 500.0])
        gamma = _evaluate_structure("spherical", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [1.0, 1.0, 1.0])

    def test_exponential_approaches_sill(self):
        """Exponential model should approach sill asymptotically."""
        h = np.array([0.0, 100.0, 300.0])
        gamma = _evaluate_structure("exponential", h, contribution=1.0, range_val=100)
        self.assertAlmostEqual(gamma[0], 0, places=5)
        self.assertGreater(gamma[1], 0.5)
        self.assertAlmostEqual(gamma[2], 1.0, places=2)

    def test_gaussian_smooth_at_origin(self):
        """Gaussian model should have smooth behavior near origin."""
        h = np.array([0.0, 1.0, 5.0, 10.0])
        gamma = _evaluate_structure("gaussian", h, contribution=1.0, range_val=100)
        self.assertAlmostEqual(gamma[0], 0, places=5)
        self.assertLess(gamma[1], 0.01)

    def test_cubic_reaches_sill(self):
        """Cubic model should reach sill at range."""
        h = np.array([100.0, 150.0])
        gamma = _evaluate_structure("cubic", h, contribution=1.0, range_val=100)
        np.testing.assert_array_almost_equal(gamma, [1.0, 1.0])

    def test_linear_unbounded(self):
        """Linear model should increase without bound."""
        h = np.array([50.0, 100.0, 200.0])
        gamma = _evaluate_structure("linear", h, contribution=1.0, range_val=100)
        self.assertAlmostEqual(gamma[0], 0.5)
        self.assertAlmostEqual(gamma[1], 1.0)
        self.assertAlmostEqual(gamma[2], 2.0)


class TestVariogramCurves(unittest.TestCase):
    """Tests for variogram curve generation."""

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

    def test_returns_three_curves(self):
        """Should return tuple of three VariogramCurveData."""
        major, semi_maj, minor = generate_variogram_curves(self.variogram)
        self.assertIsInstance(major, VariogramCurveData)
        self.assertIsInstance(semi_maj, VariogramCurveData)
        self.assertIsInstance(minor, VariogramCurveData)

    def test_curve_directions(self):
        """Each curve should have correct direction label."""
        major, semi_maj, minor = generate_variogram_curves(self.variogram)
        self.assertEqual(major.direction, "major")
        self.assertEqual(semi_maj.direction, "semi_major")
        self.assertEqual(minor.direction, "minor")

    def test_curve_sill(self):
        """Each curve should have the variogram sill."""
        major, semi_maj, minor = generate_variogram_curves(self.variogram)
        self.assertEqual(major.sill, 10.0)
        self.assertEqual(semi_maj.sill, 10.0)
        self.assertEqual(minor.sill, 10.0)

    def test_curve_starts_at_nugget(self):
        """Curves should start at nugget value."""
        major, semi_maj, minor = generate_variogram_curves(self.variogram)
        self.assertAlmostEqual(major.semivariance[0], 2.0)
        self.assertAlmostEqual(semi_maj.semivariance[0], 2.0)
        self.assertAlmostEqual(minor.semivariance[0], 2.0)

    def test_curve_approaches_sill(self):
        """Curves should approach sill at large distances."""
        major, semi_maj, minor = generate_variogram_curves(self.variogram)
        # Minor reaches sill fastest, major slowest
        self.assertGreater(minor.semivariance[-1], major.semivariance[-1] * 0.8)

    def test_custom_max_lag(self):
        """Should respect custom max_lag parameter."""
        major, _, _ = generate_variogram_curves(self.variogram, max_lag=1000)
        self.assertAlmostEqual(major.distance[-1], 1000.0)

    def test_custom_n_points(self):
        """Should respect custom n_points parameter."""
        major, _, _ = generate_variogram_curves(self.variogram, n_points=50)
        self.assertEqual(len(major.distance), 50)
        self.assertEqual(len(major.semivariance), 50)

    def test_range_values(self):
        """Should have correct range values for each direction."""
        major, semi_maj, minor = generate_variogram_curves(self.variogram)
        # Max range across structures: major=500, semi_major=400, minor=200
        self.assertEqual(major.range_value, 500.0)
        self.assertEqual(semi_maj.range_value, 400.0)
        self.assertEqual(minor.range_value, 200.0)


class TestVariogramLikeProtocol(unittest.TestCase):
    """Tests for variogram-like objects."""

    def test_works_with_variogram_like_object(self):
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

        major, semi_maj, minor = generate_variogram_curves(MockVariogram())
        self.assertEqual(major.sill, 5.0)
        self.assertAlmostEqual(major.semivariance[0], 1.0)


class TestEllipsoidClass(unittest.TestCase):
    """Tests for the Ellipsoid class."""

    def test_basic_creation(self):
        """Should create ellipsoid with ranges and rotation."""
        from evo.objects.notebooks import Ellipsoid

        ell = Ellipsoid(ranges=(100, 50, 25), rotation=(45, 30, 0))
        self.assertEqual(ell.ranges, (100, 50, 25))
        self.assertEqual(ell.rotation, (45, 30, 0))

    def test_surface_points(self):
        """Should generate surface points as 1D arrays."""
        from evo.objects.notebooks import Ellipsoid

        ell = Ellipsoid(ranges=(100, 50, 25))
        x, y, z = ell.surface_points(center=(0, 0, 0), n_points=10)

        self.assertEqual(len(x), 100)  # 10 x 10
        self.assertEqual(len(y), 100)
        self.assertEqual(len(z), 100)

    def test_surface_points_with_center(self):
        """Should offset surface points by center."""
        from evo.objects.notebooks import Ellipsoid

        ell = Ellipsoid(ranges=(100, 50, 25))
        x1, y1, z1 = ell.surface_points(center=(0, 0, 0))
        x2, y2, z2 = ell.surface_points(center=(100, 200, 50))

        # Second ellipsoid should be offset
        self.assertAlmostEqual(np.mean(x2) - np.mean(x1), 100, places=1)
        self.assertAlmostEqual(np.mean(y2) - np.mean(y1), 200, places=1)
        self.assertAlmostEqual(np.mean(z2) - np.mean(z1), 50, places=1)

    def test_wireframe_points(self):
        """Should generate wireframe points with NaN separators."""
        from evo.objects.notebooks import Ellipsoid

        ell = Ellipsoid(ranges=(100, 50, 25))
        x, y, z = ell.wireframe_points(center=(0, 0, 0))

        # Should have NaN separators
        self.assertTrue(np.any(np.isnan(x)))
        self.assertTrue(np.any(np.isnan(y)))
        self.assertTrue(np.any(np.isnan(z)))

    def test_from_variogram(self):
        """Should create ellipsoid from variogram structure."""
        from evo.objects.notebooks import Ellipsoid

        variogram = {
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
                }
            ],
        }

        ell = Ellipsoid.from_variogram(variogram)
        self.assertEqual(ell.ranges, (200, 150, 100))
        self.assertEqual(ell.rotation, (45, 30, 0))
        self.assertIn("spherical", ell.label)

    def test_from_search_ellipsoid_dict(self):
        """Should create ellipsoid from search ellipsoid dict."""
        from evo.objects.notebooks import Ellipsoid

        search_dict = {
            "ranges": {"major": 250, "semi_major": 200, "minor": 150},
            "rotation": {"dip_azimuth": 0, "dip": 0, "pitch": 0},
        }

        ell = Ellipsoid.from_search_ellipsoid(search_dict)
        self.assertEqual(ell.ranges, (250, 200, 150))
        self.assertEqual(ell.rotation, (0, 0, 0))
        self.assertEqual(ell.label, "Search Ellipsoid")


if __name__ == "__main__":
    unittest.main()
