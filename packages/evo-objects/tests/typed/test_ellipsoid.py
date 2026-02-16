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
from parameterized import parameterized

from evo.objects.typed.types import (
    Ellipsoid,
    EllipsoidRanges,
    Rotation,
)
from evo.objects.typed.variogram import (
    VariogramCurveData,
    _evaluate_structure,
)



class TestEllipsoidWireframe(unittest.TestCase):
    """Tests for ellipsoid wireframe generation."""

    def test_wireframe_shape(self):
        """Wireframe should return coordinate arrays of same length."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.wireframe_points(n_points=20)
        self.assertEqual(len(x), len(y))
        self.assertEqual(len(y), len(z))
        self.assertGreater(len(x), 0)

    def test_wireframe_bounds(self):
        """Wireframe points should be within ellipsoid bounds."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.wireframe_points(n_points=30)

        # Filter out NaN separators
        valid = ~np.isnan(x)
        self.assertTrue(np.all(np.abs(x[valid]) <= 100 * 1.01))
        self.assertTrue(np.all(np.abs(y[valid]) <= 50 * 1.01))
        self.assertTrue(np.all(np.abs(z[valid]) <= 25 * 1.01))

    def test_wireframe_has_nan_separators(self):
        """Wireframe should have NaN values separating line segments."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.wireframe_points()
        self.assertTrue(np.any(np.isnan(x)))

    def test_wireframe_with_rotation(self):
        """Wireframe should apply rotation."""
        ell_no_rot = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25), rotation=Rotation(0, 0, 0))
        ell_rot = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25), rotation=Rotation(45, 30, 0))
        x1, y1, z1 = ell_no_rot.wireframe_points()
        x2, y2, z2 = ell_rot.wireframe_points()
        # Rotated should have different coordinates
        self.assertFalse(np.allclose(x1, x2, equal_nan=True))


class TestEllipsoidSurface(unittest.TestCase):
    """Tests for ellipsoid surface mesh generation."""

    def test_surface_shape(self):
        """Surface should return flattened 1D arrays."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.surface_points(n_points=15)
        self.assertEqual(x.ndim, 1)
        self.assertEqual(y.ndim, 1)
        self.assertEqual(z.ndim, 1)
        self.assertEqual(len(x), 15 * 15)

    def test_surface_bounds(self):
        """Surface points should be within ellipsoid bounds."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.surface_points()
        self.assertTrue(np.all(np.abs(x) <= 100 * 1.01))
        self.assertTrue(np.all(np.abs(y) <= 50 * 1.01))
        self.assertTrue(np.all(np.abs(z) <= 25 * 1.01))


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


class TestVariogramMethods(unittest.TestCase):
    """Tests for Variogram class methods like get_ellipsoid, get_principal_directions, get_direction."""

    def _create_mock_variogram(self, structures):
        """Create a mock variogram object with given structure dicts."""
        from unittest.mock import MagicMock
        from evo.objects.typed.variogram import Variogram
        variogram = MagicMock()
        variogram.structures = structures
        variogram.nugget = 0.1
        variogram.sill = 1.0
        return variogram

    def test_get_ellipsoid_default_selects_largest_volume(self):
        """get_ellipsoid() with no args should select structure with largest volume."""
        from evo.objects.typed.variogram import Variogram

        # Create mock variogram with two structures
        # Structure 0: 100 * 50 * 25 = 125,000
        # Structure 1: 200 * 100 * 50 = 1,000,000 (larger)
        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.3,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 100.0, "semi_major": 50.0, "minor": 25.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
            {
                "variogram_type": "spherical",
                "contribution": 0.6,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 45.0, "dip": 30.0, "pitch": 15.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        # Bind the method from Variogram class
        ellipsoid = Variogram.get_ellipsoid(variogram)

        # Should select structure 1 (larger volume)
        self.assertEqual(ellipsoid.ranges.major, 200.0)
        self.assertEqual(ellipsoid.ranges.semi_major, 100.0)
        self.assertEqual(ellipsoid.ranges.minor, 50.0)
        self.assertEqual(ellipsoid.rotation.dip_azimuth, 45.0)

    def test_get_ellipsoid_explicit_index(self):
        """get_ellipsoid(structure_index=0) should select first structure."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.3,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 100.0, "semi_major": 50.0, "minor": 25.0},
                    "rotation": {"dip_azimuth": 10.0, "dip": 20.0, "pitch": 30.0},
                },
            },
            {
                "variogram_type": "spherical",
                "contribution": 0.6,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 45.0, "dip": 30.0, "pitch": 15.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        # Explicitly select structure 0
        ellipsoid = Variogram.get_ellipsoid(variogram, structure_index=0)

        self.assertEqual(ellipsoid.ranges.major, 100.0)
        self.assertEqual(ellipsoid.rotation.dip_azimuth, 10.0)

    def test_get_principal_directions_returns_three_curves(self):
        """get_principal_directions() should return major, semi_major, minor curves."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        major, semi_major, minor = Variogram.get_principal_directions(variogram, n_points=50)

        # Check that each is a VariogramCurveData
        self.assertIsInstance(major, VariogramCurveData)
        self.assertIsInstance(semi_major, VariogramCurveData)
        self.assertIsInstance(minor, VariogramCurveData)

        # Check direction labels
        self.assertEqual(major.direction, "major")
        self.assertEqual(semi_major.direction, "semi_major")
        self.assertEqual(minor.direction, "minor")

        # Check that arrays have expected length
        self.assertEqual(len(major.distance), 50)
        self.assertEqual(len(major.semivariance), 50)

    def test_get_principal_directions_respects_max_distance(self):
        """get_principal_directions(max_distance=...) should limit the distance range."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        major, _, _ = Variogram.get_principal_directions(variogram, max_distance=150.0)

        # Max distance should be 150
        self.assertAlmostEqual(major.distance[-1], 150.0)

    def test_get_direction_returns_arrays(self):
        """get_direction() should return (distance, semivariance) tuple of arrays."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        distance, semivariance = Variogram.get_direction(variogram, azimuth=45.0, dip=30.0, n_points=100)

        self.assertIsInstance(distance, np.ndarray)
        self.assertIsInstance(semivariance, np.ndarray)
        self.assertEqual(len(distance), 100)
        self.assertEqual(len(semivariance), 100)

    def test_get_direction_starts_at_nugget(self):
        """get_direction() should start at nugget value at distance 0."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        distance, semivariance = Variogram.get_direction(variogram, azimuth=0, dip=0)

        # First value should be at distance 0 with semivariance = nugget
        self.assertAlmostEqual(distance[0], 0.0)
        self.assertAlmostEqual(semivariance[0], 0.1)  # nugget = 0.1

    def test_get_direction_reaches_sill(self):
        """get_direction() should reach the sill (nugget + contributions) at/beyond range."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 100.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        # Get curve in any direction (isotropic, so all directions same)
        distance, semivariance = Variogram.get_direction(variogram, azimuth=0, dip=0, n_points=200)

        # Sill should be nugget (0.1) + contribution (0.9) = 1.0
        # At distances >= range, semivariance should be at the sill
        expected_sill = 0.1 + 0.9  # nugget + contribution
        self.assertAlmostEqual(semivariance[-1], expected_sill, places=2)

    def test_get_direction_anisotropic_different_ranges(self):
        """get_direction() should show different ranges in different directions for anisotropic variograms."""
        from evo.objects.typed.variogram import Variogram

        # Anisotropic variogram: major=200, semi_major=100, minor=50
        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        # With no rotation, the ellipsoid axes map to:
        # - major (200) -> X axis -> azimuth=90, dip=0
        # - semi_major (100) -> Y axis -> azimuth=0, dip=0
        # - minor (50) -> Z axis -> azimuth=0, dip=90

        # Direction along major axis (azimuth=90, dip=0 -> X direction, which is major)
        dist_major, sv_major = Variogram.get_direction(variogram, azimuth=90, dip=0, max_distance=300, n_points=100)

        # Direction along semi_major axis (azimuth=0, dip=0 -> Y direction, which is semi_major)
        dist_semi, sv_semi = Variogram.get_direction(variogram, azimuth=0, dip=0, max_distance=300, n_points=100)

        # Direction along minor axis (azimuth=0, dip=90 -> -Z direction, which is minor)
        dist_minor, sv_minor = Variogram.get_direction(variogram, azimuth=0, dip=90, max_distance=300, n_points=100)

        # At distance 75 (1.5x minor range, beyond minor but within semi_major and major)
        idx_75 = np.searchsorted(dist_major, 75)

        # Minor direction should be at sill (75 > 50)
        # For spherical: at h_norm=1.5, gamma = contribution = 0.9, plus nugget=0.1 = 1.0
        self.assertGreater(sv_minor[idx_75], 0.95)  # Should be at or near sill (1.0)

        # Semi-major direction should be part way (75 < 100)
        # For spherical at h_norm=0.75: gamma = 0.9 * (1.5*0.75 - 0.5*0.75^3) ≈ 0.82, plus nugget = 0.92
        # This is close to sill, so use a relative comparison
        self.assertLess(sv_semi[idx_75], sv_minor[idx_75])  # Should be less than minor (which is at sill)

        # Major direction should be even less (75 < 200)
        # For spherical at h_norm=0.375: gamma ≈ 0.47, plus nugget ≈ 0.57
        self.assertLess(sv_major[idx_75], sv_semi[idx_75])

    def test_get_direction_with_max_distance(self):
        """get_direction(max_distance=...) should respect the specified max distance."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        distance, _ = Variogram.get_direction(variogram, azimuth=45, dip=30, max_distance=500.0, n_points=50)

        self.assertAlmostEqual(distance[-1], 500.0)
        self.assertEqual(len(distance), 50)

    def test_get_direction_auto_max_distance(self):
        """get_direction() should auto-calculate max_distance as 1.3x effective range."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 100.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        # For isotropic variogram with range=100, auto max should be ~130
        distance, _ = Variogram.get_direction(variogram, azimuth=0, dip=0)

        # Should be approximately 1.3 * 100 = 130
        self.assertAlmostEqual(distance[-1], 130.0, places=0)

    def test_get_direction_with_rotation(self):
        """get_direction() should correctly handle rotated variogram structures."""
        from evo.objects.typed.variogram import Variogram

        # Rotated anisotropic variogram
        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 100.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 90.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        # With dip_azimuth rotation of 90°, the local coordinate system is rotated:
        # World Y (azimuth=0) maps to local X (major direction, range=200)
        # World X (azimuth=90) maps to local -Y (semi_major direction, range=100)
        dist_x, sv_x = Variogram.get_direction(variogram, azimuth=90, dip=0, max_distance=300, n_points=100)
        dist_y, sv_y = Variogram.get_direction(variogram, azimuth=0, dip=0, max_distance=300, n_points=100)

        # At distance 150 (between semi_major=100 and major=200)
        idx_150 = np.searchsorted(dist_x, 150)

        # X direction (semi_major in rotated system, range=100) should be at or near sill at 150
        self.assertGreater(sv_x[idx_150], 0.95)

        # Y direction (major in rotated system, range=200) should be less than sill at 150
        self.assertLess(sv_y[idx_150], 0.95)

    def test_get_direction_multiple_structures(self):
        """get_direction() should correctly sum contributions from multiple structures."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "spherical",
                "contribution": 0.4,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 50.0, "semi_major": 50.0, "minor": 50.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
            {
                "variogram_type": "spherical",
                "contribution": 0.5,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 200.0, "semi_major": 200.0, "minor": 200.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        distance, semivariance = Variogram.get_direction(variogram, azimuth=0, dip=0, max_distance=300, n_points=100)

        # At distance 75 (beyond first structure range of 50)
        idx_75 = np.searchsorted(distance, 75)
        # First structure contributes full 0.4, second structure partial
        # Nugget (0.1) + first (0.4) = 0.5, plus some from second
        self.assertGreater(semivariance[idx_75], 0.5)
        self.assertLess(semivariance[idx_75], 1.0)

        # Total sill should be nugget + all contributions = 0.1 + 0.4 + 0.5 = 1.0
        self.assertAlmostEqual(semivariance[-1], 1.0, places=2)

    def test_get_direction_exponential_structure(self):
        """get_direction() should work with exponential variogram structures."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "exponential",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 100.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        distance, semivariance = Variogram.get_direction(variogram, azimuth=0, dip=0, max_distance=300, n_points=100)

        # Exponential approaches sill asymptotically
        # At 3x range (300), should be very close to sill
        self.assertAlmostEqual(semivariance[-1], 1.0, places=1)

        # Should start at nugget
        self.assertAlmostEqual(semivariance[0], 0.1)

    def test_get_direction_gaussian_structure(self):
        """get_direction() should work with gaussian variogram structures."""
        from evo.objects.typed.variogram import Variogram

        structures = [
            {
                "variogram_type": "gaussian",
                "contribution": 0.9,
                "anisotropy": {
                    "ellipsoid_ranges": {"major": 100.0, "semi_major": 100.0, "minor": 100.0},
                    "rotation": {"dip_azimuth": 0.0, "dip": 0.0, "pitch": 0.0},
                },
            },
        ]
        variogram = self._create_mock_variogram(structures)

        distance, semivariance = Variogram.get_direction(variogram, azimuth=0, dip=0, max_distance=300, n_points=100)

        # Gaussian has parabolic behavior near origin (slow initial increase)
        # Check that early slope is small
        early_slope = (semivariance[5] - semivariance[0]) / (distance[5] - distance[0])
        later_slope = (semivariance[20] - semivariance[15]) / (distance[20] - distance[15])

        # Gaussian should have smaller slope near origin than later
        self.assertLess(early_slope, later_slope)


class TestEllipsoidClass(unittest.TestCase):
    """Tests for the Ellipsoid class."""

    def test_basic_creation(self):
        """Should create ellipsoid with ranges and rotation."""
        ell = Ellipsoid(
            ranges=EllipsoidRanges(100, 50, 25),
            rotation=Rotation(45, 30, 0),
        )
        self.assertEqual(ell.ranges.major, 100)
        self.assertEqual(ell.ranges.semi_major, 50)
        self.assertEqual(ell.ranges.minor, 25)
        self.assertEqual(ell.rotation.dip_azimuth, 45)

    def test_surface_points(self):
        """Should generate surface points as 1D arrays."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.surface_points(center=(0, 0, 0), n_points=10)

        self.assertEqual(len(x), 100)  # 10 x 10
        self.assertEqual(len(y), 100)
        self.assertEqual(len(z), 100)

    def test_surface_points_with_center(self):
        """Should offset surface points by center."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x1, y1, z1 = ell.surface_points(center=(0, 0, 0))
        x2, y2, z2 = ell.surface_points(center=(100, 200, 50))

        # Second ellipsoid should be offset
        self.assertAlmostEqual(np.mean(x2) - np.mean(x1), 100, places=1)
        self.assertAlmostEqual(np.mean(y2) - np.mean(y1), 200, places=1)
        self.assertAlmostEqual(np.mean(z2) - np.mean(z1), 50, places=1)

    def test_wireframe_points(self):
        """Should generate wireframe points with NaN separators."""
        ell = Ellipsoid(ranges=EllipsoidRanges(100, 50, 25))
        x, y, z = ell.wireframe_points(center=(0, 0, 0))

        # Should have NaN separators
        self.assertTrue(np.any(np.isnan(x)))
        self.assertTrue(np.any(np.isnan(y)))
        self.assertTrue(np.any(np.isnan(z)))

    def test_scaled(self):
        """Should create scaled ellipsoid."""
        ell = Ellipsoid(
            ranges=EllipsoidRanges(100, 50, 25),
            rotation=Rotation(45, 30, 0),
        )
        scaled = ell.scaled(2.0)

        self.assertEqual(scaled.ranges.major, 200)
        self.assertEqual(scaled.ranges.semi_major, 100)
        self.assertEqual(scaled.ranges.minor, 50)
        # Rotation should be preserved
        self.assertEqual(scaled.rotation.dip_azimuth, 45)

    def test_to_dict(self):
        """Should serialize to dictionary."""
        ell = Ellipsoid(
            ranges=EllipsoidRanges(100, 50, 25),
            rotation=Rotation(45, 30, 0),
        )
        d = ell.to_dict()

        self.assertEqual(d["ellipsoid_ranges"]["major"], 100)
        self.assertEqual(d["rotation"]["dip_azimuth"], 45)



class TestEllipsoidWireframeAxisAlignment(unittest.TestCase):
    """Parameterized tests for ellipsoid wireframe axis alignment after rotation.

    Uses a convenient 4:2:1 ratio for major:semi_major:minor to make axis identification clear.
    The unrotated ellipsoid has major along X, semi_major along Y, minor along Z.
    """

    def _get_axis_extent(self, wireframe_points: tuple, axis: str) -> float:
        """Get the extent (max - min) of wireframe points along a given axis."""
        x, y, z = wireframe_points
        axis_map = {'x': x, 'y': y, 'z': z}
        values = axis_map[axis]
        valid = ~np.isnan(values)
        return np.max(values[valid]) - np.min(values[valid])

    @parameterized.expand([
        # (dip, dip_azimuth, pitch, major_axis, semi_major_axis, minor_axis, description)
        (90, 0, 0, 'x', 'z', 'y', "Dip 90: major=X, semi_major=Z, minor=Y"),
        (0, 90, 0, 'y', 'x', 'z', "Dip azimuth 90: major=Y, semi_major=X, minor=Z"),
        (0, 0, 90, 'y', 'x', 'z', "Pitch 90: major=Y, semi_major=X, minor=Z"),
        (0, 90, 90, 'x', 'y', 'z', "Dip azimuth 90, pitch 90: major=X, semi_major=Y, minor=Z"),
        (90, 90, 90, 'z', 'y', 'x', "Dip 90, dip_azimuth 90, pitch 90: major=Z, semi_major=Y, minor=X"),
        (90, 90, 0, 'y', 'z', 'x', "Dip 90, dip_azimuth 90: major=Y, semi_major=Z, minor=X"),
        (90, 0, 90, 'z', 'x', 'y', "Dip 90, pitch 90: major=Z, semi_major=X, minor=Y"),
    ])
    # fmt: on
    def test_simple_axis_alignment(
        self, dip: float, dip_azimuth: float, pitch: float,
        major_axis: str, semi_major_axis: str, minor_axis: str, description: str
    ):
        """Test that wireframe axes align correctly after simple rotations."""
        # Use 4:2:1 ratio for clear axis identification
        major_range = 400
        semi_major_range = 200
        minor_range = 100

        ell = Ellipsoid(
            ranges=EllipsoidRanges(major_range, semi_major_range, minor_range),
            rotation=Rotation(dip_azimuth, dip, pitch),
        )
        wireframe = ell.wireframe_points()

        # Check major axis extent (should be 2*major_range = 800)
        major_extent = self._get_axis_extent(wireframe, major_axis)
        self.assertAlmostEqual(
            major_extent, 2 * major_range, delta=major_range * 0.1,
            msg=f"{description}: Major axis extent on {major_axis} should be ~{2*major_range}, got {major_extent}"
        )

        # Check semi_major axis extent (should be 2*semi_major_range = 400)
        semi_major_extent = self._get_axis_extent(wireframe, semi_major_axis)
        self.assertAlmostEqual(
            semi_major_extent, 2 * semi_major_range, delta=semi_major_range * 0.1,
            msg=f"{description}: Semi-major axis extent on {semi_major_axis} should be ~{2*semi_major_range}, got {semi_major_extent}"
        )

        # Check minor axis extent (should be 2*minor_range = 200)
        minor_extent = self._get_axis_extent(wireframe, minor_axis)
        self.assertAlmostEqual(
            minor_extent, 2 * minor_range, delta=minor_range * 0.1,
            msg=f"{description}: Minor axis extent on {minor_axis} should be ~{2*minor_range}, got {minor_extent}"
        )

    @parameterized.expand([
        # (major, semi_major, minor, dip_azimuth, dip, pitch, center, expected_major_p1, expected_semi_p1, expected_minor_p1, description)
        (134.0, 90.0, 40.0, 100.0, 65.0, 75.0, (500, 500, 100),
         (440, 475, 217), (495, 413, 79), (536, 494, 117),
         "Complex case 1: ranges=(134, 90, 40), rotation=(dip=65, dip_az=100, pitch=75)"),
        (100.0, 50.0, 10.0, 300.0, 13.0, 105.0, (500, 500, 100),
         (569, 431, 122), (535, 536, 103), (498, 501, 110),
         "Complex case 2: ranges=(100, 50, 10), rotation=(dip=13, dip_az=300, pitch=105)"),
    ])
    def test_complex_rotation_endpoints(
        self, major: float, semi_major: float, minor: float,
        dip_azimuth: float, dip: float, pitch: float,
        center: tuple[float, float, float],
        expected_major_p1: tuple[float, float, float],
        expected_semi_p1: tuple[float, float, float],
        expected_minor_p1: tuple[float, float, float] | None,
        description: str
    ):
        """Test ellipsoid axis endpoints against known reference values.

        This verifies that the rotation matrix produces correct axis directions
        by comparing computed endpoints to values from a reference tool.
        """

        # Get the rotation matrix
        R = Rotation(dip_azimuth, dip, pitch).as_rotation_matrix()

        # Compute axis directions and endpoints
        center_arr = np.array(center)
        major_dir = R @ np.array([1, 0, 0])
        semi_dir = R @ np.array([0, 1, 0])
        minor_dir = R @ np.array([0, 0, 1])

        major_p1 = center_arr + major * major_dir
        semi_p1 = center_arr + semi_major * semi_dir
        minor_p1 = center_arr + minor * minor_dir

        # Tolerance of ±1 for each coordinate
        tolerance = 1.0

        # Check major axis endpoint
        for i, (axis, computed, expected) in enumerate(zip(['X', 'Y', 'Z'], major_p1, expected_major_p1)):
            self.assertAlmostEqual(
                computed, expected, delta=tolerance,
                msg=f"{description}: Major P1 {axis} should be ~{expected}, got {computed:.1f}"
            )

        # Check semi-major axis endpoint
        for i, (axis, computed, expected) in enumerate(zip(['X', 'Y', 'Z'], semi_p1, expected_semi_p1)):
            self.assertAlmostEqual(
                computed, expected, delta=tolerance,
                msg=f"{description}: Semi P1 {axis} should be ~{expected}, got {computed:.1f}"
            )

        # Check minor axis endpoint (if expected values provided)
        if expected_minor_p1 is not None:
            for i, (axis, computed, expected) in enumerate(zip(['X', 'Y', 'Z'], minor_p1, expected_minor_p1)):
                self.assertAlmostEqual(
                    computed, expected, delta=tolerance,
                    msg=f"{description}: Minor P1 {axis} should be ~{expected}, got {computed:.1f}"
                )

    def test_identity_rotation_baseline(self):
        """Test that with no rotation, axes are in default positions (X=major, Y=semi_major, Z=minor)."""
        major_range = 400
        semi_major_range = 200
        minor_range = 100

        ell = Ellipsoid(
            ranges=EllipsoidRanges(major_range, semi_major_range, minor_range),
            rotation=Rotation(0, 0, 0),
        )
        wireframe = ell.wireframe_points()

        x_extent = self._get_axis_extent(wireframe, 'x')
        y_extent = self._get_axis_extent(wireframe, 'y')
        z_extent = self._get_axis_extent(wireframe, 'z')

        # X should have major extent (800)
        self.assertAlmostEqual(x_extent, 2 * major_range, delta=major_range * 0.1,
                               msg=f"X extent should be ~{2*major_range}, got {x_extent}")
        # Y should have semi_major extent (400)
        self.assertAlmostEqual(y_extent, 2 * semi_major_range, delta=semi_major_range * 0.1,
                               msg=f"Y extent should be ~{2*semi_major_range}, got {y_extent}")
        # Z should have minor extent (200)
        self.assertAlmostEqual(z_extent, 2 * minor_range, delta=minor_range * 0.1,
                               msg=f"Z extent should be ~{2*minor_range}, got {z_extent}")


if __name__ == "__main__":
    unittest.main()
