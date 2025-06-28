# frame_analyzer.py

"""
This module defines the core FrameAnalyzer class for performing a 3D Finite
Element Analysis (FEA) on frame structures.

It encapsulates all the necessary logic for assembling the structural system,
solving for displacements, and post-processing the results to determine internal
forces, moments, and stresses.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Assumes the data model classes are in a separate file named `structural_model.py`
from structural_model import Node, Beam, Connection


class _UnionFind:
    """
    A private utility class implementing the Union-Find data structure.
    Used internally for efficiently grouping rotationally connected beam ends at a node
    to correctly assign shared degrees of freedom.
    """
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py: return
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1


class FrameAnalyzer:
    """
    Performs a 3D structural analysis of a frame structure.

    This class encapsulates the entire Finite Element Method (FEM) process for
    3D frame structures. It takes a structural definition (nodes, beams, etc.),
    assembles the global stiffness matrix, solves for displacements, and computes
    internal forces, moments, and stresses.

    Typical Usage:
        # 1. Define nodes, beams, connections
        nodes = [...]
        beams = [...]
        connections = [...]

        # 2. Instantiate the analyzer
        analyzer = FrameAnalyzer(nodes, beams, connections)

        # 3. Define loads and supports, then solve
        loads = [...]
        supports = [...]
        if analyzer.solve(loads, supports):
            # 4. Retrieve and use the results
            results = analyzer.get_results()
            print(results)
    """

    def __init__(self, nodes: List[Node], beams: List[Beam], connections: List[Connection]):
        """
        Initializes the FrameAnalyzer with the structural model.

        Args:
            nodes: A list of Node objects defining the geometry.
            beams: A list of Beam objects defining the elements.
            connections: A list of Connection objects defining how beams are joined.
        """
        self.nodes = nodes
        self.beams = beams
        self.connections = connections
        self._displacements: Optional[np.ndarray] = None
        self._results: Optional[List[Dict[str, Any]]] = None

    def solve(self, loads: List[Dict], boundary_conditions: List[Dict]) -> bool:
        """
        Assembles and solves the linear system of equations [K]{U} = {F}.

        This is the main public method to run a complete analysis.

        Args:
            loads: A list of dictionaries defining the applied loads.
                   Example: [{'node_idx': 2, 'Fy': -5000}, {'beam_idx': 0, 'qy': -1000}]
            boundary_conditions: A list of dictionaries defining the imposed displacements (supports).
                                 Example: [{'node_idx': 0, 'u':0,'v':0,'w':0}]

        Returns:
            True if the solution was successful, False otherwise.
        """
        try:
            K, F = self._assemble_system(loads, boundary_conditions)
            self._displacements = spsolve(K, F)
            if np.isnan(self._displacements).any():
                print("Warning: Solver returned NaN values. The structure may be unstable.")
                return False
            self._compute_results()
            return True
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            return False

    def get_results(self) -> Optional[List[Dict[str, Any]]]:
        """
        Returns the computed forces, moments, and stresses after a successful analysis.

        Returns:
            A list of dictionaries containing detailed results for each beam, or None if
            analysis has not been run or failed.
        """
        return self._results

    # --------------------------------------------------------------------------
    # Private Helper Methods - The core calculation engine
    # --------------------------------------------------------------------------

    def _compute_beam_properties(self, beam: Beam):
        """Calculates and caches the length and rotation matrix for a beam."""
        n1, n2 = self.nodes[beam.node1_idx], self.nodes[beam.node2_idx]
        vec = np.array([n2.x - n1.x, n2.y - n1.y, n2.z - n1.z])
        beam.length = np.linalg.norm(vec)

        if beam.length < 1e-9:
            raise ValueError(f"Beam {self.beams.index(beam)} has zero length.")

        local_x = vec / beam.length
        if np.allclose(np.cross(local_x, beam.ref_vector), 0):
            alt_ref_vector = np.array([0,1,0]) if not np.allclose(np.abs(local_x), [0,1,0]) else np.array([1,0,0])
            local_y = np.cross(alt_ref_vector, local_x)
        else:
            local_y = np.cross(beam.ref_vector, local_x)

        local_y /= np.linalg.norm(local_y)
        local_z = np.cross(local_x, local_y)
        beam.rotation_matrix = np.vstack([local_x, local_y, local_z])

    def _local_stiffness_matrix(self, beam: Beam) -> np.ndarray:
        """Generates the local 12x12 stiffness matrix for a 3D frame element."""
        E, L, A, J, Iy, Iz = beam.E, beam.length, beam.A, beam.J, beam.Iy, beam.Iz
        G = E / (2 * (1 + beam.nu))
        k = np.zeros((12, 12))

        k[0,0]=k[6,6]=E*A/L; k[0,6]=k[6,0]=-E*A/L
        k[3,3]=k[9,9]=G*J/L; k[3,9]=k[9,3]=-G*J/L
        c1=12*E*Iz/L**3; c2=6*E*Iz/L**2; c3=4*E*Iz/L; c4=2*E*Iz/L
        k[1,1]=c1; k[1,5]=c2; k[1,7]=-c1; k[1,11]=c2
        k[5,1]=c2; k[5,5]=c3; k[5,7]=-c2; k[5,11]=c4
        k[7,1]=-c1; k[7,5]=-c2; k[7,7]=c1; k[7,11]=-c2
        k[11,1]=c2; k[11,5]=c4; k[11,7]=-c2; k[11,11]=c3
        c5=12*E*Iy/L**3; c6=6*E*Iy/L**2; c7=4*E*Iy/L; c8=2*E*Iy/L
        k[2,2]=c5; k[2,4]=-c6; k[2,8]=-c5; k[2,10]=-c6
        k[4,2]=-c6; k[4,4]=c7; k[4,8]=c6; k[4,10]=c8
        k[8,2]=-c5; k[8,4]=c6; k[8,8]=c5; k[8,10]=c6
        k[10,2]=-c6;k[10,4]=c8; k[10,8]=c6; k[10,10]=c7

        return k

    def _transformation_matrix(self, beam: Beam) -> np.ndarray:
        """Creates the 12x12 transformation matrix for a beam."""
        R = beam.rotation_matrix
        T = np.zeros((12, 12))
        for i in range(4): T[i*3:(i+1)*3, i*3:(i+1)*3] = R
        return T

    def _equivalent_nodal_loads(self, beam: Beam, q_local_y: float, q_local_z: float) -> np.ndarray:
        """Calculates equivalent nodal loads (fixed-end actions) from distributed loads."""
        L = beam.length
        f_local = np.zeros(12)
        if q_local_y != 0:
            f_local[1]=f_local[7]=-q_local_y*L/2; f_local[5]=-q_local_y*L**2/12; f_local[11]=q_local_y*L**2/12
        if q_local_z != 0:
            f_local[2]=f_local[8]=-q_local_z*L/2; f_local[4]=q_local_z*L**2/12; f_local[10]=-q_local_z*L**2/12
        return f_local

    def _assign_dofs(self) -> Tuple[int, List[List[int]]]:
        """Assigns degrees of freedom to all nodes and beams in the structure."""
        beams_at_node = [[] for _ in self.nodes]
        for b_idx, beam in enumerate(self.beams):
            beams_at_node[beam.node1_idx].append(b_idx)
            beams_at_node[beam.node2_idx].append(b_idx)

        total_dofs = 0
        theta_dof_map = {}
        # First, assign translational DOFs to each node
        for n_idx, node in enumerate(self.nodes):
            node.dof_indices = [total_dofs, total_dofs + 1, total_dofs + 2]
            total_dofs += 3

        # Second, assign rotational DOFs based on connectivity
        for n_idx, node in enumerate(self.nodes):
            if not beams_at_node[n_idx]: continue

            beam_indices = beams_at_node[n_idx]
            uf = _UnionFind(len(self.beams))
            for conn in self.connections:
                if conn.conn_type == 'rigid' and (conn.beam1_idx in beam_indices and conn.beam2_idx in beam_indices):
                    uf.union(conn.beam1_idx, conn.beam2_idx)

            components = {}
            for b_idx in beam_indices:
                root = uf.find(b_idx)
                if root not in components:
                    components[root] = [total_dofs, total_dofs + 1, total_dofs + 2]
                    total_dofs += 3
                theta_dof_map[(n_idx, b_idx)] = components[root]

        # Finally, store the full 12 DOFs for each beam
        for b_idx, beam in enumerate(self.beams):
            dof1 = self.nodes[beam.node1_idx].dof_indices + theta_dof_map.get((beam.node1_idx, b_idx), [-1,-1,-1])
            dof2 = self.nodes[beam.node2_idx].dof_indices + theta_dof_map.get((beam.node2_idx, b_idx), [-1,-1,-1])
            beam.dof_indices = dof1 + dof2

        return total_dofs, beams_at_node

    def _assemble_system(self, loads: List[Dict], boundary_conditions: List[Dict]) -> Tuple[csr_matrix, np.ndarray]:
        """Assembles the global stiffness matrix and force vector."""
        total_dofs, beams_at_node = self._assign_dofs()
        K = lil_matrix((total_dofs, total_dofs))
        F = np.zeros(total_dofs)

        for beam in self.beams:
            self._compute_beam_properties(beam)
            k_local = self._local_stiffness_matrix(beam)
            T = self._transformation_matrix(beam)
            k_global = T.T @ k_local @ T
            for i in range(12):
                for j in range(12):
                    if beam.dof_indices[i] >= 0 and beam.dof_indices[j] >= 0:
                        K[beam.dof_indices[i], beam.dof_indices[j]] += k_global[i, j]

        for conn in self.connections:
            if conn.conn_type == 'spring' and conn.k_spring:
                b1, b2 = self.beams[conn.beam1_idx], self.beams[conn.beam2_idx]
                common_node_set = set([b1.node1_idx, b1.node2_idx]) & set([b2.node1_idx, b2.node2_idx])
                if common_node_set:
                    n_idx = common_node_set.pop()
                    th1 = b1.dof_indices[3:6] if b1.node1_idx == n_idx else b1.dof_indices[9:12]
                    th2 = b2.dof_indices[3:6] if b2.node1_idx == n_idx else b2.dof_indices[9:12]
                    axis_map = {'kx': 0, 'ky': 1, 'kz': 2}
                    for axis, k_val in conn.k_spring.items():
                        if axis in axis_map:
                            idx = axis_map[axis]
                            dof1, dof2 = th1[idx], th2[idx]
                            K[dof1,dof1]+=k_val; K[dof2,dof2]+=k_val
                            K[dof1,dof2]-=k_val; K[dof2,dof1]-=k_val

        for load in loads:
            if 'node_idx' in load:
                n_idx = load['node_idx']
                dofs = self.nodes[n_idx].dof_indices
                F[dofs[0]]+=load.get('Fx',0); F[dofs[1]]+=load.get('Fy',0); F[dofs[2]]+=load.get('Fz',0)
                if any(k in load for k in ['Mx','My','Mz']):
                    if beams_at_node[n_idx]:
                        b_idx=beams_at_node[n_idx][0]
                        th_dofs=self.beams[b_idx].dof_indices[3:6] if self.beams[b_idx].node1_idx==n_idx else self.beams[b_idx].dof_indices[9:12]
                        F[th_dofs[0]]+=load.get('Mx',0); F[th_dofs[1]]+=load.get('My',0); F[th_dofs[2]]+=load.get('Mz',0)
            elif 'beam_idx' in load:
                beam = self.beams[load['beam_idx']]
                f_local = self._equivalent_nodal_loads(beam, load.get('qy',0), load.get('qz',0))
                f_global = self._transformation_matrix(beam).T @ f_local
                for i in range(12): F[beam.dof_indices[i]] += f_global[i]

        for bc in boundary_conditions:
            n_idx = bc['node_idx']
            dof_map = {'u':0,'v':1,'w':2}
            for dof_type, value in bc.items():
                if dof_type == 'node_idx': continue
                dof = -1
                if dof_type in dof_map:
                    dof = self.nodes[n_idx].dof_indices[dof_map[dof_type]]
                elif dof_type in ['theta_x','theta_y','theta_z']:
                    if beams_at_node[n_idx]:
                        b_idx=beams_at_node[n_idx][0]
                        th_dofs=self.beams[b_idx].dof_indices[3:6] if self.beams[b_idx].node1_idx==n_idx else self.beams[b_idx].dof_indices[9:12]
                        th_map={'theta_x':0,'theta_y':1,'theta_z':2}
                        dof=th_dofs[th_map[dof_type]]
                if dof != -1:
                    K[dof,:], K[:,dof] = 0,0
                    K[dof,dof], F[dof] = 1,value

        return K.tocsr(), F

    def _compute_results(self):
        """
        Computes all end actions and stresses for each beam, applying a standard
        diagrammatic sign convention for moments.
        """
        results = []
        for i, beam in enumerate(self.beams):
            dofs = beam.dof_indices
            T = self._transformation_matrix(beam)
            u_local = T @ self._displacements[dofs]
            k_local = self._local_stiffness_matrix(beam)
            f_local = k_local @ u_local

            # --- Extract Internal Forces and Moments for Reporting ---
            # This convention produces results consistent with standard engineering diagrams
            # (e.g., NASTRAN), which is essential for verification and interpretation.

            # Axial (N): Positive is tension.
            N1, N2 = -f_local[0], f_local[6]

            # Shear (V): Internal shear on the cross-section.
            Vy1, Vy2 = f_local[1], -f_local[7]
            Vz1, Vz2 = f_local[2], -f_local[8]

            # Moment (M): For diagrams, the moment at End B is flipped.
            # This ensures that for a simple beam under gravity, both end
            # moments are negative (hogging) and the mid-span is positive (sagging).
            Mx1, My1, Mz1 = f_local[3], f_local[4], f_local[5]
            Mx2, My2, Mz2 = -f_local[9], -f_local[10], -f_local[11]

            # --- Calculate Stresses ---
            # NOTE: Stress calculation must use the original, unflipped local moments
            # from f_local to be physically correct.
            sigma_axial_1 = N1 / beam.A
            s1_pp = sigma_axial_1 + (f_local[5] * beam.cy / beam.Iz) - (f_local[4] * beam.cz / beam.Iy)
            s1_pn = sigma_axial_1 + (f_local[5] * beam.cy / beam.Iz) + (f_local[4] * beam.cz / beam.Iy)
            s1_np = sigma_axial_1 - (f_local[5] * beam.cy / beam.Iz) - (f_local[4] * beam.cz / beam.Iy)
            s1_nn = sigma_axial_1 - (f_local[5] * beam.cy / beam.Iz) + (f_local[4] * beam.cz / beam.Iy)

            sigma_axial_2 = N2 / beam.A
            s2_pp = sigma_axial_2 + (f_local[11] * beam.cy / beam.Iz) - (f_local[10] * beam.cz / beam.Iy)
            s2_pn = sigma_axial_2 + (f_local[11] * beam.cy / beam.Iz) + (f_local[10] * beam.cz / beam.Iy)
            s2_np = sigma_axial_2 - (f_local[11] * beam.cy / beam.Iz) - (f_local[10] * beam.cz / beam.Iy)
            s2_nn = sigma_axial_2 - (f_local[11] * beam.cy / beam.Iz) + (f_local[10] * beam.cz / beam.Iy)

            results.append({
                'beam_idx': i,
                'node1': {'N':N1, 'Vy':Vy1, 'Vz':Vz1, 'Mx':Mx1, 'My':My1, 'Mz':Mz1,
                          'sigma_axial': sigma_axial_1,
                          'stresses': {'y+z+': s1_pp, 'y+z-': s1_pn, 'y-z-': s1_nn, 'y-z+': s1_np}},
                'node2': {'N':N2, 'Vy':Vy2, 'Vz':Vz2, 'Mx':Mx2, 'My':My2, 'Mz':Mz2,
                          'sigma_axial': sigma_axial_2,
                          'stresses': {'y+z+': s2_pp, 'y+z-': s2_pn, 'y-z-': s2_nn, 'y-z+': s2_np}},
            })
        self._results = results
