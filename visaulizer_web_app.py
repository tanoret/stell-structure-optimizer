import numpy as np
from typing import List
import io  # Required for in-memory image buffer

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors

# This assumes frame_analyzer.py is in the same directory
from frame_analyzer import FrameAnalyzer

class Visualizer:
    """
    Handles the 3D visualization of the structural model and its results.
    MODIFIED: Generates BytesIO image objects instead of showing plots.
    """
    def __init__(self, analyzer: FrameAnalyzer):
        if analyzer.get_results() is None:
            raise ValueError("The provided analyzer has not been solved or the solution failed.")
        self.analyzer = analyzer
        self.nodes = analyzer.nodes
        self.beams = analyzer.beams
        self.results = analyzer.get_results()
        self.displacements = analyzer._displacements

    def plot_to_image(self, plot_type: str = 'moment_z', deformed: bool = False, amplification_factor: float = 50.0, title: str = None, line_width: int = 5):
        """
        Generates a 3D plot and returns it as an in-memory image BytesIO object.
        """
        if deformed:
            fig = plt.figure(figsize=(15, 11))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_deformed_shape_on_ax(ax, plot_type, amplification_factor, title, line_width)
        else:
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            self._plot_undeformed_on_ax(ax, plot_type, title, line_width)

        # Save the plot to an in-memory buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)  # Close the figure to free up memory
        buf.seek(0)
        return buf

    def _plot_undeformed_on_ax(self, ax, result_type, title, line_width):
        is_abs = 'abs' in result_type or 'mag' in result_type or 'stress' in result_type
        beam_end_vals, label, cmap, _ = self._get_result_data(result_type, for_beam_ends=True)
        if title is None:
            title = label

        all_values = [v for pair in beam_end_vals for v in pair]
        min_val, max_val = min(all_values) if all_values else 0, max(all_values) if all_values else 0

        if not is_abs and min_val < 0:
            abs_max = max(abs(min_val), abs(max_val)) if all_values else 1
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        else:
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

        for i, beam in enumerate(self.beams):
            n1, n2 = self.nodes[beam.node1_idx], self.nodes[beam.node2_idx]
            val1, val2 = beam_end_vals[i]
            points = np.array([[n1.x, n1.z, n1.y], [n2.x, n2.z, n2.y]])
            self._add_colored_line_segments(ax, points[0], points[1], val1, val2, cmap, norm, line_width)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
        sm.set_array([])
        fig = ax.get_figure()
        fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, label=label)

        ax.set_xlabel("Global X (m)")
        ax.set_ylabel("Global Z (m)")
        ax.set_zlabel("Global Y (m) - Elevation")
        ax.set_title(title, fontsize=16)
        self._set_plot_aspect_ratio(ax)

    def _plot_deformed_shape_on_ax(self, ax, result_type, amplification_factor, title, line_width):
        displaced_nodes_coords = [
            (n.x + self.displacements[n.dof_indices[0]] * amplification_factor,
             n.y + self.displacements[n.dof_indices[1]] * amplification_factor,
             n.z + self.displacements[n.dof_indices[2]] * amplification_factor)
            for n in self.nodes
        ]

        for beam in self.beams:
            n1_orig, n2_orig = self.nodes[beam.node1_idx], self.nodes[beam.node2_idx]
            ax.plot([n1_orig.x, n2_orig.x], [n1_orig.z, n2_orig.z], [n1_orig.y, n2_orig.y],
                    color='grey', linestyle='--', linewidth=2, alpha=0.4)

        nodal_values, label, cmap, is_abs = self._get_result_data(result_type)
        if title is None:
            title = label
        title = f"{title} on Deformed Shape ({amplification_factor}x)"

        all_values = [item for sublist in nodal_values for item in sublist]
        min_val, max_val = min(all_values) if all_values else 0, max(all_values) if all_values else 0

        if not is_abs and min_val < 0:
            abs_max = max(abs(min_val), abs(max_val)) if all_values else 1
            norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
        else:
            norm = mcolors.Normalize(vmin=min_val, vmax=max_val)

        for i, beam in enumerate(self.beams):
            n1_idx, n2_idx = beam.node1_idx, beam.node2_idx
            p1 = np.array([displaced_nodes_coords[n1_idx][0], displaced_nodes_coords[n1_idx][2], displaced_nodes_coords[n1_idx][1]])
            p2 = np.array([displaced_nodes_coords[n2_idx][0], displaced_nodes_coords[n2_idx][2], displaced_nodes_coords[n2_idx][1]])
            val1 = np.mean(nodal_values[n1_idx]) if nodal_values[n1_idx] else 0
            val2 = np.mean(nodal_values[n2_idx]) if nodal_values[n2_idx] else 0
            self._add_colored_line_segments(ax, p1, p2, val1, val2, cmap, norm, line_width)

        sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap), norm=norm)
        sm.set_array([])
        fig = ax.get_figure()
        fig.colorbar(sm, ax=ax, shrink=0.6, aspect=15, label=label)

        ax.set_xlabel("Global X (m)")
        ax.set_ylabel("Global Z (m)")
        ax.set_zlabel("Global Y (m) - Elevation")
        ax.set_title(title, fontsize=16)
        self._set_plot_aspect_ratio(ax)

    def _get_result_data(self, result_type, for_beam_ends=False):
        """Helper function to extract data and labels for plotting."""
        nodal_values = [[] for _ in self.nodes]
        beam_end_vals = []
        is_abs = 'abs' in result_type or 'mag' in result_type or 'stress' in result_type
        label, cmap = result_type.replace('_', ' ').title(), 'jet'

        if result_type in ['x', 'y', 'z']:
            cmap = 'viridis'
            disp_data = {'x': (0, "Displacement X (m)"), 'y': (1, "Displacement Y (m)"), 'z': (2, "Displacement Z (m)")}
            key, label = disp_data[result_type]
            for i, node in enumerate(self.nodes):
                val = self.displacements[node.dof_indices[key]]
                nodal_values[i].append(val)
        elif 'disp' in result_type:
            cmap = 'viridis'
            disp_data = {'disp_x': (0, "Displacement X (m)"), 'disp_y': (1, "Displacement Y (m)"),
                         'disp_z': (2, "Displacement Z (m)"), 'displacement_magnitude': (-1, "Displacement Magnitude (m)")}
            key, label = disp_data[result_type]
            for i, node in enumerate(self.nodes):
                dx, dy, dz = self.displacements[node.dof_indices]
                val = np.sqrt(dx**2 + dy**2 + dz**2) if key == -1 else self.displacements[node.dof_indices[key]]
                nodal_values[i].append(val)
        else:
            for r in self.results:
                b = self.beams[r['beam_idx']]
                v1, v2 = 0, 0
                if result_type == 'axial_force':
                    v1, v2, label = r['node1']['N'], r['node2']['N'], "Axial Force N (Tension +)"
                elif result_type == 'shear_x':
                    v1, v2, label = r['node1']['Vx'], r['node2']['Vx'], "Local Shear Vx (N)"
                elif result_type == 'shear_y':
                    v1, v2, label = r['node1']['Vy'], r['node2']['Vy'], "Local Shear Vy (N)"
                elif result_type == 'shear_z':
                    v1, v2, label = r['node1']['Vz'], r['node2']['Vz'], "Local Shear Vz (N)"
                elif result_type == 'stress':
                    v1, v2, label, cmap = max(abs(v) for v in r['node1']['stresses'].values()), max(abs(v) for v in r['node2']['stresses'].values()), "Max Corner Stress (Pa)", 'viridis'
                elif 'mag' in result_type:
                    label, cmap = "Magnitude", 'viridis'
                    if 'shear' in result_type:
                        v1, v2, label = np.sqrt(r['node1']['Vx']**2 + r['node1']['Vy']**2 + r['node1']['Vz']**2), np.sqrt(r['node2']['Vx']**2 + r['node2']['Vy']**2 + r['node2']['Vz']**2), "Shear Force Magnitude (N)"
                    if 'moment' in result_type:
                        v1, v2, label = np.sqrt(r['node1']['Mx']**2 + r['node1']['My']**2 + r['node1']['Mz']**2), np.sqrt(r['node2']['Mx']**2 + r['node2']['My']**2 + r['node2']['Mz']**2), "Bending Moment Magnitude (Nm)"
                elif 'moment' in result_type:
                    key, cmap = ('Mx' if 'x' in result_type else 'My' if 'y' in result_type else 'Mz'), 'viridis'
                    v1, v2, label = abs(r['node1'][key]), abs(r['node2'][key]), f"|Local Moment {key}| (Nm)"
                else:
                    raise ValueError(f"Unknown result_type: '{result_type}'")
                nodal_values[b.node1_idx].append(v1)
                nodal_values[b.node2_idx].append(v2)
                if for_beam_ends:
                    beam_end_vals.append((v1, v2))

        return (beam_end_vals if for_beam_ends else nodal_values), label, cmap, is_abs

    # _add_colored_line_segments and _set_plot_aspect_ratio remain unchanged
    def _add_colored_line_segments(self, ax, p1, p2, val1, val2, cmap, norm, line_width):
        """Helper to draw a single beam with interpolated colors."""
        points = np.vstack([p1, p2])
        num_segments = 10
        x, y, z = [np.linspace(points[0,i], points[1,i], num_segments+1) for i in range(3)]
        segments = np.array([[[x[j],y[j],z[j]], [x[j+1],y[j+1],z[j+1]]] for j in range(num_segments)])
        interp_vals = np.linspace(val1, val2, num_segments)
        colors = plt.get_cmap(cmap)(norm(interp_vals))
        ax.add_collection(Line3DCollection(segments, colors=colors, linewidths=line_width))

    def _set_plot_aspect_ratio(self, ax):
        """Sets the plot's aspect ratio to be equal based on node coordinates."""
        all_x = [n.x for n in self.nodes]
        all_y = [n.y for n in self.nodes]
        all_z = [n.z for n in self.nodes]

        if not all_x:  # Handle empty model case
            ax.set_box_aspect((1, 1, 1))
            return

        x_c, y_c, z_c = np.mean(all_x), np.mean(all_y), np.mean(all_z)
        x_r, y_r, z_r = np.ptp(all_x), np.ptp(all_y), np.ptp(all_z)

        max_range = max(x_r, y_r, z_r, 1.0)  # Use at least 1.0 to avoid zero-size plots

        ax.set_xlim(x_c - max_range / 2, x_c + max_range / 2)
        ax.set_ylim(z_c - max_range / 2, z_c + max_range / 2)  # Matplotlib's Y is our Z
        ax.set_zlim(y_c - max_range / 2, y_c + max_range / 2)  # Matplotlib's Z is our Y

        ax.set_box_aspect((1, 1, 1))
