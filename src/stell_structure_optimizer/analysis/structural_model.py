# structural_model.py

import numpy as np
from typing import Optional, List

class Node:
    """Represents a node in 3D space.

    Attributes:
        x (float): The x-coordinate of the node.
        y (float): The y-coordinate of the node.
        z (float): The z-coordinate of the node.
        dof_indices (Optional[List[int]]): A list of global degree-of-freedom
            indices [u, v, w] associated with this node's translations.
    """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.dof_indices: Optional[List[int]] = None

class Beam:
    """Represents a 3D beam element defined by two nodes and material/section properties.

    Attributes:
        node1_idx (int): Index of the starting node in the main node list.
        node2_idx (int): Index of the ending node in the main node list.
        E (float): Young's modulus (in Pascals).
        nu (float): Poisson's ratio.
        A (float): Cross-sectional area (in m^2).
        Iy (float): Moment of inertia about the local y-axis (in m^4).
        Iz (float): Moment of inertia about the local z-axis (in m^4).
        J (float): Torsional constant (in m^4).
        cy (float): Max distance from neutral axis in local y-direction (for stress, in m).
        cz (float): Max distance from neutral axis in local z-direction (for stress, in m).
        ref_vector (np.ndarray): A vector used to define the orientation of the local y-axis.
        length (Optional[float]): The length of the beam (calculated).
        rotation_matrix (Optional[np.ndarray]): The 3x3 matrix to transform from global to local coordinates.
        dof_indices (Optional[List[int]]): The 12 global DOF indices for this beam.
    """
    def __init__(self, node1_idx: int, node2_idx: int, E: float, nu: float, rho: float, A: float, Iy: float, Iz: float, J: float, cy: float, cz: float, ref_vector: Optional[np.ndarray] = None):
        self.node1_idx = node1_idx
        self.node2_idx = node2_idx
        self.E = E
        self.nu = nu
        self.rho = rho
        self.A = A
        self.Iy = Iy
        self.Iz = Iz
        self.J = J
        self.cy = cy
        self.cz = cz
        self.ref_vector = ref_vector if ref_vector is not None else np.array([0, 0, 1])
        self.length: Optional[float] = None
        self.rotation_matrix: Optional[np.ndarray] = None
        self.dof_indices: Optional[List[int]] = None

class Connection:
    """Defines a connection between two beams at a shared node.

    Attributes:
        beam1_idx (int): Index of the first beam.
        beam2_idx (int): Index of the second beam.
        conn_type (str): The type of connection, either 'rigid' or 'spring'.
        k_spring (Optional[dict]): A dictionary of spring stiffnesses {'kx', 'ky', 'kz'}
            if the connection type is 'spring'.
    """
    def __init__(self, beam1_idx: int, beam2_idx: int, conn_type: str, k_spring: Optional[dict] = None):
        self.beam1_idx = beam1_idx
        self.beam2_idx = beam2_idx
        self.conn_type = conn_type
        self.k_spring = k_spring if conn_type == 'spring' else None
