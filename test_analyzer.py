# test_analyzer.py

import unittest
import numpy as np

# Import the classes from your new structure
from structural_model import Node, Beam
from frame_analyzer import FrameAnalyzer

class TestFrameAnalyzer(unittest.TestCase):
    """Unit tests for the core calculation methods in FrameAnalyzer."""

    def setUp(self):
        """Set up a simple model to be used by all tests."""
        self.nodes = [Node(0, 0, 0), Node(1, 0, 0)]
        self.beams = [Beam(0, 1, E=1.0, A=1.0, Iy=1.0, Iz=1.0, J=1.0, cy=0.5, cz=0.5)]
        self.analyzer = FrameAnalyzer(self.nodes, self.beams, [])
        # Manually run compute_beam_properties to set length etc. for the test beam
        self.analyzer._compute_beam_properties(self.beams[0])

    def test_compute_beam_properties(self):
        """Test the calculation of beam length and rotation matrix."""
        nodes = [Node(0, 0, 0), Node(3, 4, 0)]
        beam = Beam(0, 1, E=1, A=1, Iy=1, Iz=1, J=1, cy=1, cz=1)
        analyzer = FrameAnalyzer(nodes, [beam], [])
        analyzer._compute_beam_properties(beam)

        self.assertAlmostEqual(beam.length, 5.0)
        # Check that the local x-axis is correct
        expected_local_x = np.array([0.6, 0.8, 0.0])
        np.testing.assert_allclose(beam.rotation_matrix[0, :], expected_local_x)

    def test_local_stiffness_matrix(self):
        """Test a few key values in the local stiffness matrix."""
        k_local = self.analyzer._local_stiffness_matrix(self.beams[0])

        # Test axial stiffness term k(0,0) = EA/L
        self.assertAlmostEqual(k_local[0, 0], 1.0)
        # Test bending stiffness term k(1,1) = 12EI_z/L^3
        self.assertAlmostEqual(k_local[1, 1], 12.0)
        # Test bending-rotation coupling term k(1,5) = 6EI_z/L^2
        self.assertAlmostEqual(k_local[1, 5], 6.0)

    def test_equivalent_nodal_loads(self):
        """Test the calculation of fixed-end forces from a distributed load."""
        q = -10.0  # Downward load
        f_local = self.analyzer._equivalent_nodal_loads(self.beams[0], q, 0)
        L = self.beams[0].length

        # Test shear force V = qL/2
        self.assertAlmostEqual(f_local[1], -q * L / 2)
        # Test moment M = qL^2/12
        self.assertAlmostEqual(f_local[11], q * L**2 / 12)
        self.assertAlmostEqual(f_local[5], -q * L**2 / 12)

if __name__ == '__main__':
    unittest.main()
