import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative
import pickle
from itertools import combinations, product
from pathlib import Path
import time
import io
from importlib import resources

from stell_structure_optimizer.analysis.structural_model import Node, Beam, Connection
from stell_structure_optimizer.data import profile_manager
from stell_structure_optimizer.analysis.frame_analyzer import FrameAnalyzer
from stell_structure_optimizer.design.aisc360_verifier import AISC360Verifier
from stell_structure_optimizer.visualization.visaulizer_web_app import Visualizer

# --- Page Configuration ---
st.set_page_config(
    page_title="Professional 3D Frame Analyzer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if 'model_defined' not in st.session_state:
    st.session_state.nodes = []
    st.session_state.beams = []
    st.session_state.connections = []
    st.session_state.loads = []  # List of load packs, each a dict with 'type' and 'data'
    st.session_state.boundary_conditions = []
    st.session_state.model_defined = False
    st.session_state.results = None
    st.session_state.optimization_results = None
    st.session_state.log = []
    st.session_state.selected_col_profile = None
    st.session_state.selected_beam_profile = None
    st.session_state.gravity_vector = [0.0, -9.81, 0.0]  # Default gravity vector (m/s²)
    st.session_state.model_type = None  # Track Parametric or Custom model

# --- Data Loading ---
@st.cache_data
def load_profile_data(pickle_name: str):
    """Load profile data bundled with the package."""
    try:
        with resources.open_binary("stell_structure_optimizer.data", pickle_name) as f:
            db = pickle.load(f)
        profile_manager.populate_database(db)
        profile_names = sorted(list(profile_manager.PROFILES_DATABASE.keys()))
        return db, profile_names
    except FileNotFoundError:
        return None, []

# --- Backend Logic ---
def build_parametric_model(n_stories, h_story, bays_x_str, bays_z_str, col_profile, beam_profile, E, nu, rho, use_custom_props=False, custom_props=None):
    try:
        bays_x = [float(x.strip()) for x in bays_x_str.split(',')]
        bays_z = [float(z.strip()) for z in bays_z_str.split(',')]
        n_stories, h_story = int(n_stories), float(h_story)
    except ValueError:
        st.error("Invalid parametric input. Please use numbers or comma-separated numbers.")
        return [], [], [], []

    # Validate profile names
    if not use_custom_props and (col_profile not in profile_manager.PROFILES_DATABASE or beam_profile not in profile_manager.PROFILES_DATABASE):
        st.error(f"Profile '{col_profile}' or '{beam_profile}' not found in profiles.pkl.")
        return [], [], [], []

    nx, nz = len(bays_x) + 1, len(bays_z) + 1
    x_pts, z_pts = [0.0] + list(np.cumsum(bays_x)), [0.0] + list(np.cumsum(bays_z))

    # Create nodes and assign node_idx
    nodes = [Node(x, s * h_story, z) for s in range(n_stories + 1) for z in z_pts for x in x_pts]
    for idx, node in enumerate(nodes):
        node.node_idx = idx

    if use_custom_props and custom_props:
        col_p = {'E': E, 'nu': nu, 'rho': rho, **custom_props['columns']}
        beam_p = {'E': E, 'nu': nu, 'rho': rho, **custom_props['beams']}
        col_profile_name = "Custom_Column"
        beam_profile_name = "Custom_Beam"
    else:
        col_p = profile_manager.get_beam_properties(col_profile, E, nu, rho)
        beam_p = profile_manager.get_beam_properties(beam_profile, E, nu, rho)
        col_profile_name = col_profile
        beam_profile_name = beam_profile

    beams, attach = [], [[] for _ in nodes]
    bid = 0

    def nid(s, ix, iz): return s * (nx * nz) + iz * nx + ix

    # Columns (vertical, ref_vector along global X)
    for s in range(n_stories):
        for iz in range(nz):
            for ix in range(nx):
                n1, n2 = nid(s, ix, iz), nid(s + 1, ix, iz)
                bm = Beam(n1, n2, **col_p, ref_vector=np.array([1, 0, 0]))
                bm.profile_name = col_profile_name
                # Compute length
                n1_pos = np.array([nodes[n1].x, nodes[n1].y, nodes[n1].z])
                n2_pos = np.array([nodes[n2].x, nodes[n2].y, nodes[n2].z])
                bm.length = np.linalg.norm(n2_pos - n1_pos)
                beams.append(bm)
                attach[n1].append(bid)
                attach[n2].append(bid)
                bid += 1

    # Beams (horizontal, ref_vector along global Y)
    for s in range(1, n_stories + 1):
        for iz in range(nz):  # X-direction
            for ix in range(nx - 1):
                n1, n2 = nid(s, ix, iz), nid(s, ix + 1, iz)
                bm = Beam(n1, n2, **beam_p, ref_vector=np.array([0, 1, 0]))
                bm.profile_name = beam_profile_name
                # Compute length
                n1_pos = np.array([nodes[n1].x, nodes[n1].y, nodes[n1].z])
                n2_pos = np.array([nodes[n2].x, nodes[n2].y, nodes[n2].z])
                bm.length = np.linalg.norm(n2_pos - n1_pos)
                beams.append(bm)
                attach[n1].append(bid)
                attach[n2].append(bid)
                bid += 1
        for ix in range(nx):  # Z-direction
            for iz in range(nz - 1):
                n1, n2 = nid(s, ix, iz), nid(s, ix, iz + 1)
                bm = Beam(n1, n2, **beam_p, ref_vector=np.array([0, 1, 0]))
                bm.profile_name = beam_profile_name
                # Compute length
                n1_pos = np.array([nodes[n1].x, nodes[n1].y, nodes[n1].z])
                n2_pos = np.array([nodes[n2].x, nodes[n2].y, nodes[n2].z])
                bm.length = np.linalg.norm(n2_pos - n1_pos)
                beams.append(bm)
                attach[n1].append(bid)
                attach[n2].append(bid)
                bid += 1

    conns = [Connection(b1, b2, 'rigid') for lst in attach if len(lst) > 1 for b1, b2 in combinations(lst, 2)]
    bcs = [{"node_idx": i, 'u': 0, 'v': 0, 'w': 0, 'theta_x': 0, 'theta_y': 0, 'theta_z': 0} for i in range(nx * nz)]

    return nodes, beams, conns, bcs

def parse_nodes_from_file(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, header=None, comment='#', skipinitialspace=True)
        nodes = [Node(row[1], row[2], row[3]) for _, row in df.iterrows()]
        for idx, node in enumerate(nodes):
            node.node_idx = idx
        return nodes
    except Exception as e:
        st.error(f"Error parsing node file: {e}")
        return []

# --- UI Rendering ---
def draw_sidebar():
    st.sidebar.title("🛠️ Structural Analysis Controls")
    st.sidebar.markdown("Configure your 3D frame model below.")

    with st.sidebar:
        # Geometry Section
        with st.container():
            st.header("1. Geometry")
            geom_mode = st.radio(
                "Model Type",
                ["Parametric", "Custom"],
                key="geom_mode",
                help="Choose Parametric for grid-based structures or Custom for manual input."
            )

            if geom_mode == "Parametric":
                with st.expander("Parametric Inputs", expanded=True):
                    st.markdown("**Define Building Geometry**")
                    n_stories = st.text_input(
                        "Number of Stories",
                        "3",
                        help="Enter the number of building stories."
                    )
                    h_story = st.number_input(
                        "Story Height (m)",
                        min_value=1.0, value=4.0, step=0.1,
                        format="%.2f",
                        help="Height of each story in meters."
                    )
                    bays_x = st.text_input(
                        "Bay Spacing X (m, comma-sep)",
                        "6, 6",
                        help="Comma-separated bay lengths along X-axis (e.g., 6,6)."
                    )
                    bays_z = st.text_input(
                        "Bay Spacing Z (m, comma-sep)",
                        "8",
                        help="Comma-separated bay lengths along Z-axis (e.g., 8)."
                    )
                    col_profile = st.selectbox(
                        "Column Profile",
                        profile_list,
                        key="param_col_profile",
                        help="Select the profile for columns."
                    )
                    beam_profile = st.selectbox(
                        "Beam Profile",
                        profile_list,
                        key="param_beam_profile",
                        help="Select the profile for beams."
                    )
                    use_custom_props = st.checkbox(
                        "Use Custom Properties",
                        False,
                        help="Override profile properties with custom values."
                    )

                    if use_custom_props:
                        st.markdown("**Custom Column Properties**")
                        col_A = st.number_input("Column Area (m²)", min_value=0.0, value=0.0123, step=0.0001)
                        col_Iz = st.number_input("Column Iz (m⁴)", min_value=0.0, value=222e-6, step=1e-6)
                        col_Iy = st.number_input("Column Iy (m⁴)", min_value=0.0, value=72.4e-6, step=1e-6)
                        col_J = st.number_input("Column J (m⁴)", min_value=0.0, value=800e-9, step=1e-9)
                        col_cy = st.number_input("Column cy (m)", min_value=0.0, value=0.155, step=0.001)
                        col_cz = st.number_input("Column cz (m)", min_value=0.0, value=0.1525, step=0.001)
                        st.markdown("**Custom Beam Properties**")
                        beam_A = st.number_input("Beam Area (m²)", min_value=0.0, value=0.0104, step=0.0001)
                        beam_Iz = st.number_input("Beam Iz (m⁴)", min_value=0.0, value=370e-6, step=1e-6)
                        beam_Iy = st.number_input("Beam Iy (m⁴)", min_value=0.0, value=19.9e-6, step=1e-6)
                        beam_J = st.number_input("Beam J (m⁴)", min_value=0.0, value=650e-9, step=1e-9)
                        beam_cy = st.number_input("Beam cy (m)", min_value=0.0, value=0.2285, step=0.001)
                        beam_cz = st.number_input("Beam cz (m)", min_value=0.0, value=0.095, step=0.001)
                        custom_props = {
                            'columns': {'A': col_A, 'Iz': col_Iz, 'Iy': col_Iy, 'J': col_J, 'cy': col_cy, 'cz': col_cz},
                            'beams': {'A': beam_A, 'Iz': beam_Iz, 'Iy': beam_Iy, 'J': beam_J, 'cy': beam_cy, 'cz': beam_cz}
                        }
                    else:
                        custom_props = None

                    if st.button("Generate Parametric Model", type="primary", use_container_width=True):
                        nodes, beams, conns, bcs = build_parametric_model(
                            n_stories, h_story, bays_x, bays_z, col_profile, beam_profile, 2e11, 0.3, 7850,
                            use_custom_props, custom_props
                        )
                        st.session_state.nodes = nodes
                        st.session_state.beams = beams
                        st.session_state.connections = conns
                        st.session_state.boundary_conditions = bcs
                        st.session_state.model_defined = True
                        st.session_state.results = None
                        st.session_state.optimization_results = None
                        # Store parameters for optimization
                        st.session_state.n_stories = n_stories
                        st.session_state.h_story = h_story
                        st.session_state.bays_x = bays_x
                        st.session_state.bays_z = bays_z
                        st.session_state.use_custom_props = use_custom_props
                        st.session_state.custom_props = custom_props
                        st.session_state.selected_col_profile = col_profile
                        st.session_state.selected_beam_profile = beam_profile
                        st.session_state.model_type = "Parametric"
                        st.success("Parametric model generated successfully!")

            else:  # Custom Mode
                with st.expander("Custom Inputs", expanded=True):
                    st.markdown("**Add Nodes Manually**")
                    col_n1, col_n2, col_n3 = st.columns(3)
                    node_x = col_n1.number_input("Node X (m)", format="%.2f", key="node_x")
                    node_y = col_n2.number_input("Node Y (m)", format="%.2f", key="node_y")
                    node_z = col_n3.number_input("Node Z (m)", format="%.2f", key="node_z")
                    if st.button("Add Node", use_container_width=True):
                        new_node = Node(node_x, node_y, node_z)
                        new_node.node_idx = len(st.session_state.nodes)
                        st.session_state.nodes.append(new_node)
                        st.session_state.model_defined = True
                        st.session_state.results = None
                        st.session_state.optimization_results = None
                        st.session_state.model_type = "Custom"
                        st.success(f"Node {new_node.node_idx} added at ({node_x}, {node_y}, {node_z})")

                    if st.session_state.nodes:
                        node_df = pd.DataFrame([
                            {'ID': n.node_idx, 'X (m)': n.x, 'Y (m)': n.y, 'Z (m)': n.z}
                            for n in st.session_state.nodes
                        ])
                        st.subheader("Defined Nodes")
                        st.dataframe(node_df, use_container_width=True)
                        node_to_remove = st.selectbox("Select Node to Remove", node_df['ID'], key="remove_node")
                        if st.button("Remove Node", type="secondary", use_container_width=True):
                            st.session_state.nodes = [n for n in st.session_state.nodes if n.node_idx != node_to_remove]
                            # Reassign node_idx
                            for idx, node in enumerate(st.session_state.nodes):
                                node.node_idx = idx
                            st.session_state.beams = [b for b in st.session_state.beams if b.node1_idx != node_to_remove and b.node2_idx != node_to_remove]
                            st.session_state.boundary_conditions = [bc for bc in st.session_state.boundary_conditions if bc['node_idx'] != node_to_remove]
                            st.session_state.loads = [pack for pack in st.session_state.loads if all(load.get('node_idx', -1) != node_to_remove for load in pack['data'])]
                            st.session_state.model_defined = bool(st.session_state.nodes)
                            st.success(f"Node {node_to_remove} removed.")

                    st.markdown("---")
                    st.markdown("**Add Beams**")
                    col_b1, col_b2 = st.columns(2)
                    node1 = col_b1.number_input("Node 1 ID", min_value=0, step=1, key="beam_n1")
                    node2 = col_b2.number_input("Node 2 ID", min_value=0, step=1, key="beam_n2")
                    beam_profile = st.selectbox("Beam Profile", profile_list, key="custom_beam_profile")
                    ref_vector = st.selectbox("Reference Vector", ["[1,0,0]", "[0,1,0]", "[0,0,1]"], index=2)
                    ref_vector_array = np.array(eval(ref_vector))
                    if st.button("Add Beam", use_container_width=True):
                        if st.session_state.model_defined and node1 < len(st.session_state.nodes) and node2 < len(st.session_state.nodes):
                            beam_props = profile_manager.get_beam_properties(beam_profile, 2e11, 0.3, 7850)
                            new_beam = Beam(node1, node2, **beam_props, ref_vector=ref_vector_array)
                            new_beam.profile_name = beam_profile
                            n1_pos = np.array([st.session_state.nodes[node1].x, st.session_state.nodes[node1].y, st.session_state.nodes[node1].z])
                            n2_pos = np.array([st.session_state.nodes[node2].x, st.session_state.nodes[node2].y, st.session_state.nodes[node2].z])
                            new_beam.length = np.linalg.norm(n2_pos - n1_pos)
                            st.session_state.beams.append(new_beam)
                            st.success(f"Beam {len(st.session_state.beams)-1} added: {node1} to {node2}")
                        else:
                            st.warning("Invalid node IDs or no nodes defined.")

                    if st.session_state.beams:
                        beam_df = pd.DataFrame([
                            {'Beam ID': idx, 'Node 1': b.node1_idx, 'Node 2': b.node2_idx, 'Profile': b.profile_name}
                            for idx, b in enumerate(st.session_state.beams)
                        ])
                        st.subheader("Defined Beams")
                        st.dataframe(beam_df, use_container_width=True)
                        beam_to_remove = st.selectbox("Select Beam to Remove", beam_df['Beam ID'], key="remove_beam")
                        if st.button("Remove Beam", type="secondary", use_container_width=True):
                            st.session_state.beams.pop(beam_to_remove)
                            # Rebuild connections
                            attach = [[] for _ in st.session_state.nodes]
                            for bid, beam in enumerate(st.session_state.beams):
                                attach[beam.node1_idx].append(bid)
                                attach[beam.node2_idx].append(bid)
                            st.session_state.connections = [Connection(b1, b2, 'rigid') for lst in attach if len(lst) > 1 for b1, b2 in combinations(lst, 2)]
                            st.success(f"Beam {beam_to_remove} removed.")

                    st.markdown("---")
                    st.markdown("**Add Supports**")
                    support_node = st.number_input("Support Node ID", min_value=0, step=1, key="support_node_id")
                    if st.button("Add Fixed Support", use_container_width=True):
                        if support_node < len(st.session_state.nodes):
                            st.session_state.boundary_conditions.append(
                                {'node_idx': support_node, 'u': 0, 'v': 0, 'w': 0, 'theta_x': 0, 'theta_y': 0, 'theta_z': 0}
                            )
                            st.session_state.model_type = "Custom"
                            st.success(f"Fixed support added at node {support_node}")
                        else:
                            st.warning("Invalid support node ID.")

                    if st.session_state.boundary_conditions:
                        bc_df = pd.DataFrame([
                            {'Node ID': bc['node_idx'], 'Type': 'Fixed (All DOFs)'}
                            for bc in st.session_state.boundary_conditions
                        ])
                        st.subheader("Defined Supports")
                        st.dataframe(bc_df, use_container_width=True)
                        bc_to_remove = st.selectbox("Select Support to Remove", bc_df.index, key="remove_bc")
                        if st.button("Remove Support", type="secondary", use_container_width=True):
                            st.session_state.boundary_conditions.pop(bc_to_remove)
                            st.session_state.model_type = "Custom"
                            st.success(f"Support {bc_to_remove} removed.")

                    st.markdown("---")
                    st.markdown("**Upload Nodes (Optional)**")
                    node_file = st.file_uploader("Upload Nodes (ID, X, Y, Z)", type=['csv', 'txt'])
                    if node_file:
                        new_nodes = parse_nodes_from_file(node_file)
                        if new_nodes:
                            max_idx = max([n.node_idx for n in st.session_state.nodes], default=-1) + 1
                            for idx, node in enumerate(new_nodes):
                                node.node_idx = max_idx + idx
                            st.session_state.nodes.extend(new_nodes)
                            st.session_state.model_defined = True
                            st.session_state.results = None
                            st.session_state.optimization_results = None
                            st.session_state.model_type = "Custom"
                            st.success(f"{len(new_nodes)} nodes uploaded successfully.")

        # Materials & Profiles
        with st.container():
            st.header("2. Materials & Profiles")
            with st.expander("Material Properties", expanded=False):
                col_m1, col_m2 = st.columns(2)
                fy = col_m1.text_input("Yield Strength Fy (Pa)", "355e6", key="fy")
                fu = col_m2.text_input("Ultimate Strength Fu (Pa)", "490e6", key="fu")
                st.multiselect(
                    "Column Profiles for Optimization",
                    profile_list,
                    key="col_profiles",
                    help="Select profiles to test for columns during optimization."
                )
                st.multiselect(
                    "Beam Profiles for Optimization",
                    profile_list,
                    key="beam_profiles",
                    help="Select profiles to test for beams during optimization."
                )

        # Loads
        with st.container():
            st.header("3. Loads")
            with st.expander("Load Definition", expanded=False):
                st.markdown("**Gravity Vector for Self-Weight**")
                col_g1, col_g2, col_g3 = st.columns(3)
                gx = col_g1.number_input("Gravity X (m/s²)", value=0.0, step=0.1, format="%.2f", key="gravity_x")
                gy = col_g2.number_input("Gravity Y (m/s²)", value=-9.81, step=0.1, format="%.2f", key="gravity_y")
                gz = col_g3.number_input("Gravity Z (m/s²)", value=0.0, step=0.1, format="%.2f", key="gravity_z")
                st.session_state.gravity_vector = [gx, gy, gz]
                st.markdown("**User-Defined Loads**")
                load_pack_name = st.text_input("Load Pack Name", "Load Pack 1")
                load_type = st.radio(
                    "Load Type",
                    ["Nodal", "Distributed"],
                    key="load_type",
                    help="Choose Nodal for point loads or Distributed for loads along beams."
                )
                if load_type == "Nodal":
                    node_ids_str = st.text_input(
                        "Node IDs (comma-sep)",
                        key="load_nodes",
                        help="Enter node IDs to apply the load (e.g., 1,2,3)."
                    )
                    col_l1, col_l2, col_l3 = st.columns(3)
                    fx = col_l1.number_input("Fx (N)", format="%.2f", key="load_fx")
                    fy = col_l2.number_input("Fy (N)", format="%.2f", key="load_fy")
                    fz = col_l3.number_input("Fz (N)", format="%.2f", key="load_fz")
                    if st.button("Add Nodal Load Pack", use_container_width=True):
                        try:
                            node_ids = [int(n.strip()) for n in node_ids_str.split(',')]
                            if all(nid < len(st.session_state.nodes) for nid in node_ids):
                                load_data = [{'node_idx': nid, 'Fx': fx, 'Fy': fy, 'Fz': fz} for nid in node_ids]
                                st.session_state.loads.append({'name': load_pack_name, 'type': 'Nodal', 'data': load_data})
                                st.session_state.model_type = "Custom"
                                st.success(f"Nodal load pack '{load_pack_name}' added.")
                            else:
                                st.warning("Invalid node IDs.")
                        except:
                            st.warning("Invalid Node ID format.")
                else:
                    beam_id = st.number_input("Beam ID", min_value=0, step=1, key="dist_beam_id")
                    qy = st.number_input("qy (N/m)", format="%.2f", key="load_qy")
                    qz = st.number_input("qz (N/m)", format="%.2f", key="load_qz")
                    if st.button("Add Distributed Load Pack", use_container_width=True):
                        if beam_id < len(st.session_state.beams):
                            st.session_state.loads.append({'name': load_pack_name, 'type': 'Distributed', 'data': [{'beam_idx': beam_id, 'qy': qy, 'qz': qz}]})
                            st.session_state.model_type = "Custom"
                            st.success(f"Distributed load pack '{load_pack_name}' added.")
                        else:
                            st.warning("Invalid beam ID.")

                if st.session_state.loads:
                    load_df = pd.DataFrame([
                        {'Pack ID': idx, 'Name': pack['name'], 'Type': pack['type'], 'Details': f"{len(pack['data'])} loads"}
                        for idx, pack in enumerate(st.session_state.loads)
                    ])
                    st.subheader("Defined Load Packs")
                    st.dataframe(load_df, use_container_width=True)
                    load_pack_to_remove = st.selectbox("Select Load Pack to Remove", load_df['Pack ID'], key="remove_load")
                    if st.button("Remove Load Pack", type="secondary", use_container_width=True):
                        removed_pack = st.session_state.loads.pop(load_pack_to_remove)
                        st.session_state.model_type = "Custom"
                        st.success(f"Load pack '{removed_pack['name']}' removed.")

                if st.button("Clear All Loads", type="secondary", use_container_width=True):
                    st.session_state.loads = []
                    st.session_state.model_type = "Custom"
                    st.success("All user-defined load packs cleared.")

        # Analysis
        with st.container():
            st.header("4. Analysis")
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                if st.button("Run Single Analysis", type="primary", use_container_width=True):
                    run_single_analysis()
            with col_a2:
                if st.button("Find Lightest Design", type="primary", use_container_width=True):
                    run_optimization()

def draw_main_content():
    st.title("🏗️ 3D Frame Analysis Dashboard")
    st.markdown("Visualize and analyze your structural model with interactive tools.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📏 3D Model View",
        "📊 Results Summary",
        "🔍 Detailed Results",
        "✅ Verification Details",
        "📜 Optimization Log"
    ])

    with tab1:
        st.subheader("Interactive 3D Model")

        show_annotations = st.checkbox("Show Details & Annotations", value=True)

        if not st.session_state.model_defined:
            st.info("Define a model using the sidebar controls to visualize it here.")

        else:
            # ------------------------------------------------------------------
            # figure container --------------------------------------------------
            # ------------------------------------------------------------------
            fig = go.Figure()

            # ------------------------------------------------------------------
            # scale-dependent arrow size (use 5 % of the longest beam) ----------
            # ------------------------------------------------------------------
            base_arrow_size = 1.0
            if st.session_state.beams:
                beam_lengths = [b.length for b in st.session_state.beams
                                if hasattr(b, "length") and b.length is not None]
                if beam_lengths:
                    max_beam_length = max(beam_lengths)
                    base_arrow_size = 1.0 if max_beam_length == 0 else max_beam_length / 10.0

            # ------------------------------------------------------------------
            # attempt to compute local axes for each beam -----------------------
            # ------------------------------------------------------------------
            advanced_visuals_possible = False
            if st.session_state.beams:
                try:
                    tmp = FrameAnalyzer(st.session_state.nodes,
                                        st.session_state.beams, [])
                    for b in st.session_state.beams:
                        tmp._compute_beam_properties(b)
                    advanced_visuals_possible = True
                except Exception:
                    st.warning("Could not render advanced visuals "
                            "(orientation/loads). Model may be incomplete.")

            # ------------------------------------------------------------------
            # node coordinates as (X, Y, Z) ------------------------------------
            # ------------------------------------------------------------------
            node_xyz = np.array([[n.x, n.y, n.z] for n in st.session_state.nodes])

            # ------------------------------------------------------------------
            # plot nodes -------------------------------------------------------
            # ------------------------------------------------------------------
            node_mode = "markers+text" if show_annotations else "markers"
            fig.add_trace(
                go.Scatter3d(
                    x=node_xyz[:, 0],          # X
                    y=node_xyz[:, 2],          # Z → plotly-Y
                    z=node_xyz[:, 1],          # Y → plotly-Z (elevation)
                    mode=node_mode,
                    text=[f"{i}" for i in range(len(st.session_state.nodes))],
                    marker=dict(size=5, color="blue"),
                    textfont=dict(size=10, color="darkblue"),
                    name="Nodes",
                    hoverinfo="text",
                    showlegend=False,
                )
            )

            # ------------------------------------------------------------------
            # plot beams -------------------------------------------------------
            # ------------------------------------------------------------------
            beam_mid_xyz, beam_labels = [], []
            for idx, beam in enumerate(st.session_state.beams):
                n1 = st.session_state.nodes[beam.node1_idx]
                n2 = st.session_state.nodes[beam.node2_idx]

                # beam centre line
                fig.add_trace(
                    go.Scatter3d(
                        x=[n1.x, n2.x],
                        y=[n1.z, n2.z],
                        z=[n1.y, n2.y],
                        mode="lines",
                        line=dict(color="black", width=4),
                        hoverinfo="name",
                        name=f"Beam {beam.node1_idx}-{beam.node2_idx}",
                        showlegend=False,
                    )
                )

                # keep mid-points for labels
                if show_annotations:
                    mid = (node_xyz[beam.node1_idx] + node_xyz[beam.node2_idx]) / 2
                    beam_mid_xyz.append(mid)
                    beam_labels.append(f"B{idx}")

            # beam labels
            if show_annotations and beam_mid_xyz:
                mid_arr = np.array(beam_mid_xyz)
                fig.add_trace(
                    go.Scatter3d(
                        x=mid_arr[:, 0],
                        y=mid_arr[:, 2],
                        z=mid_arr[:, 1],
                        mode="text",
                        text=beam_labels,
                        textfont=dict(color="purple", size=9),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            # ------------------------------------------------------------------
            # fixed supports ---------------------------------------------------
            # ------------------------------------------------------------------
            if show_annotations and st.session_state.boundary_conditions:
                support_nodes = {bc["node_idx"]
                                for bc in st.session_state.boundary_conditions}
                V = base_arrow_size * 1.5
                H = base_arrow_size * 0.75
                legend_added = False

                for idx in support_nodes:
                    x, y, z = node_xyz[idx]
                    base = (x, y - V, z)
                    x_arm = [(x - H, base[1], z), (x + H, base[1], z)]
                    z_arm = [(x, base[1], z - H), (x, base[1], z + H)]

                    for pts in [[(x, y, z), base], x_arm, z_arm]:
                        p0, p1 = pts
                        fig.add_trace(
                            go.Scatter3d(
                                x=[p0[0], p1[0]],
                                y=[p0[2], p1[2]],
                                z=[p0[1], p1[1]],
                                mode="lines",
                                line=dict(color="red", width=8),
                                name="Fixed Supports",
                                legendgroup="Fixed Supports",
                                showlegend=not legend_added,
                                hoverinfo="name",
                            )
                        )
                    legend_added = True

            # ------------------------------------------------------------------
            # section major-axis arrows ---------------------------------------
            # ------------------------------------------------------------------
            if show_annotations and advanced_visuals_possible:
                L = base_arrow_size
                ez_local = np.array([0, 0, 1])   # local +Z
                for beam in st.session_state.beams:
                    if not hasattr(beam, "rotation_matrix"):
                        continue
                    mid = (node_xyz[beam.node1_idx] + node_xyz[beam.node2_idx]) / 2
                    ez_global = beam.rotation_matrix.T @ ez_local
                    u, v, w = ez_global[0], ez_global[2], ez_global[1]
                    fig.add_trace(
                        go.Cone(
                            x=[mid[0]], y=[mid[2]], z=[mid[1]],
                            u=[u * L], v=[v * L], w=[w * L],
                            showscale=False,
                            colorscale=[[0, "grey"], [1, "grey"]],
                            name="Major Axis",
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )

            # ------------------------------------------------------------------
            # loads (if any) ---------------------------------------------------
            # ------------------------------------------------------------------
            if show_annotations and advanced_visuals_possible and st.session_state.loads:

                colors = qualitative.Plotly
                L_load = base_arrow_size * 2.0
                shown = set()

                sep_ratio = -0.4

                for i, pack in enumerate(st.session_state.loads):
                    colour = colors[i % len(colors)]
                    pname = pack.get("name", f"Load Pack {i + 1}")
                    legend = pname not in shown

                    # ---------------------------------------------------------------
                    # NODAL LOADS (tail moved by +sep along F̂)
                    # ---------------------------------------------------------------
                    if pack["type"] == "Nodal":
                        for load in pack["data"]:
                            pos = node_xyz[load["node_idx"]]
                            F   = np.array([load.get("Fx", 0),
                                            load.get("Fy", 0),
                                            load.get("Fz", 0)])
                            mag = np.linalg.norm(F)
                            if mag < 1e-6:
                                continue
                            dir_ = F / mag

                            sep  = sep_ratio * L_load
                            tail = pos + dir_ * sep

                            u = dir_[0] * L_load
                            v = dir_[2] * L_load
                            w = dir_[1] * L_load
                            fig.add_trace(
                                go.Cone(
                                    x=[tail[0]], y=[tail[2]], z=[tail[1]],   # shifted tail
                                    u=[u], v=[v], w=[w],
                                    showscale=False,
                                    colorscale=[[0, colour], [1, colour]],
                                    name=pname, legendgroup=pname,
                                    showlegend=legend,
                                    hovertext=(f"<b>{pname}</b><br>"
                                            f"Node {load['node_idx']}<br>"
                                            f"Force: [{F[0]:.1f}, {F[1]:.1f}, {F[2]:.1f}] N"),
                                    hoverinfo="text",
                                )
                            )
                            shown.add(pname)

                    # ---------------------------------------------------------------
                    # DISTRIBUTED LOADS (tail moved by +sep along q̂)
                    # ---------------------------------------------------------------
                    elif pack["type"] == "Distributed":
                        for load in pack["data"]:
                            beam = st.session_state.beams[load["beam_idx"]]
                            if not hasattr(beam, "rotation_matrix"):
                                continue

                            n1 = node_xyz[beam.node1_idx]
                            n2 = node_xyz[beam.node2_idx]

                            q_local = np.array([0,
                                                load.get("qy", 0),
                                                load.get("qz", 0)])
                            if np.linalg.norm(q_local) < 1e-6:
                                continue
                            q_glob = beam.rotation_matrix.T @ q_local
                            q_dir  = q_glob / np.linalg.norm(q_glob)

                            u = q_dir[0] * L_load
                            v = q_dir[2] * L_load
                            w = q_dir[1] * L_load

                            sep = sep_ratio * L_load   # same 15 % offset

                            for frac in (0.25, 0.5, 0.75):
                                base  = n1 + frac * (n2 - n1)   # original start
                                tail  = base + q_dir * sep      # shifted tail
                                fig.add_trace(
                                    go.Cone(
                                        x=[tail[0]], y=[tail[2]], z=[tail[1]],
                                        u=[u], v=[v], w=[w],
                                        showscale=False,
                                        colorscale=[[0, colour], [1, colour]],
                                        name=pname, legendgroup=pname,
                                        showlegend=legend and frac == 0.25,
                                        hovertext=(f"<b>{pname}</b><br>"
                                                f"Beam {load['beam_idx']}<br>"
                                                f"Dist. Load: "
                                                f"[qy={load.get('qy',0):.1f}, "
                                                f"qz={load.get('qz',0):.1f}] N/m"),
                                        hoverinfo="text",
                                    )
                                )
                            shown.add(pname)

            # ------------------------------------------------------------------
            # enforce a perfect 1 : 1 : 1 cube ---------------------------------
            # ------------------------------------------------------------------
            xs, ys, zs = [], [], []
            for tr in fig.data:
                if hasattr(tr, "x") and tr.x is not None:
                    xs.extend(tr.x)
                if hasattr(tr, "y") and tr.y is not None:
                    ys.extend(tr.y)
                if hasattr(tr, "z") and tr.z is not None:
                    zs.extend(tr.z)

                # for cones, also include the arrow tips (tail + vector)
                if tr.type == "cone":
                    if (tr.u is not None and tr.v is not None and tr.w is not None
                            and tr.x is not None and tr.y is not None and tr.z is not None):
                        xs.extend(np.array(tr.x) + np.array(tr.u))
                        ys.extend(np.array(tr.y) + np.array(tr.v))
                        zs.extend(np.array(tr.z) + np.array(tr.w))

            xs, ys, zs = np.array(xs), np.array(ys), np.array(zs)

            # fallback for an empty figure
            if xs.size == 0:
                xs = ys = zs = np.array([0.0])

            span_x, span_y, span_z = (np.ptp(xs), np.ptp(ys), np.ptp(zs))
            cube = max(span_x, span_y, span_z) or 1.0e-6          # avoid /0
            half = cube / 2.0 * 1.1
            mid_x, mid_y, mid_z = xs.mean(), ys.mean(), zs.mean()

            x_rng = [mid_x - half, mid_x + half]
            y_rng = [mid_y - half, mid_y + half]
            z_rng = [mid_z - half, mid_z + half]

            # ------------------------------------------------------------------
            # final layout -----------------------------------------------------
            # ------------------------------------------------------------------
            fig.update_layout(
                scene=dict(
                    aspectmode="manual",
                    aspectratio=dict(x=1, y=1, z=1),
                    xaxis=dict(range=x_rng, title="Global X (m)"),
                    yaxis=dict(range=y_rng, title="Global Z (m)"),
                    zaxis=dict(range=z_rng, title="Global Y (m) – Elevation"),
                ),
                legend=dict(orientation="h",
                            yanchor="bottom", y=1.02,
                            xanchor="right", x=1),
                showlegend=True,
                margin=dict(l=0, r=0, b=0, t=40),
                height=700,
            )

            # ------------------------------------------------------------------
            # render in Streamlit ---------------------------------------------
            # ------------------------------------------------------------------
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Analysis Results")
        if st.session_state.results is None:
            st.info("Run an analysis to see results here.")
        else:
            res = st.session_state.results
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric("Total Mass (kg)", f"{res['mass']:,.2f}")
            with col_r2:
                st.metric("Max Unity Check", f"{res['max_uc']:.3f}", delta=f"{res['max_uc']-1.0:.3f} from limit")
            st.subheader("Result Plots")
            # Dynamically get available components from analyzer results and map to supported plot types
            analyzer = res['analyzer']
            available_components = set()
            for r in analyzer.get_results():
                available_components.update(r['node1'].keys())
                if 'node2' in r:
                    available_components.update(r['node2'].keys())
            # Define supported plot types and their corresponding labels
            supported_mappings = {
                'N': 'axial_force',
                'Vx': 'shear_x',
                'Vy': 'shear_y',
                'Vz': 'shear_z',
                'Mx': 'moment_x',
                'My': 'moment_y',
                'Mz': 'moment_z',
                'stresses': 'stress',
                'x': 'x',
                'y': 'y',
                'z': 'z'
            }
            # Filter available components to those supported by Visualizer
            valid_components = [comp for comp in available_components if comp in supported_mappings]
            selected_component = st.selectbox("Select Component to Color", valid_components, index=valid_components.index('Mz') if 'Mz' in valid_components else 0)
            plot_type = supported_mappings[selected_component]
            visualizer = Visualizer(res['analyzer'])
            plot_result = visualizer.plot_to_image(plot_type=plot_type)
            st.image(plot_result, caption=f"Structure Colored by {selected_component}", use_container_width=True)

            st.subheader("Deformed Shape")
            # Linear slider for amplification factor with minimum and default of 1.0
            amplification_factor = st.slider("Amplification Factor", min_value=1.0, max_value=500.0, value=1.0, step=1.0)
            # Number input for manual entry
            manual_amplification = st.number_input("Amplification Factor", min_value=1.0, value=amplification_factor, step=1.0, format="%.1f")
            amplification_factor = manual_amplification if manual_amplification >= 1.0 else amplification_factor
            plot_deformed = visualizer.plot_to_image(plot_type='disp_y', deformed=True, amplification_factor=amplification_factor)
            st.image(plot_deformed, caption="Deformed Shape", use_container_width=True)

    with tab3:
        st.subheader("Detailed Results")
        if st.session_state.results and 'results_table' in st.session_state.results:
            st.dataframe(st.session_state.results['results_table'], use_container_width=True)
        else:
            st.info("No detailed results available. Run an analysis to populate this tab.")

    with tab4:
        st.subheader("Verification Details (AISC 360)")
        if st.session_state.results and 'uc_table' in st.session_state.results:
            st.write("**Proposed Design Verification**")
            st.dataframe(st.session_state.results['uc_table'], use_container_width=True)
            csv = st.session_state.results['uc_table'].to_csv(index=False)
            st.download_button(
                label="Download Proposed Verification as CSV",
                data=csv,
                file_name="proposed_verification.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Run a single analysis to see proposed design verification details.")

        if st.session_state.optimization_results and 'uc_table' in st.session_state.optimization_results:
            st.write("**Optimized Design Verification**")
            st.dataframe(st.session_state.optimization_results['uc_table'], use_container_width=True)
            csv = st.session_state.optimization_results['uc_table'].to_csv(index=False)
            st.download_button(
                label="Download Optimized Verification as CSV",
                data=csv,
                file_name="optimized_verification.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Run optimization to see optimized design verification details.")

    with tab5:
        st.subheader("Optimization Log")
        log_text = "\n".join(st.session_state.log)
        st.code(log_text, language=None)

def compute_self_weight_loads(nodes, beams):
    g = 9.81  # Gravity acceleration (m/s²)
    global_load_direction = np.array(st.session_state.gravity_vector) / g  # Normalize to unit vector: [0, -1, 0]
    self_weight_loads = []
    temp_analyzer = FrameAnalyzer(nodes, beams, [])  # Temporary analyzer to compute beam properties
    for idx, beam in enumerate(beams):
        if hasattr(beam, 'length') and beam.length is not None:
            temp_analyzer._compute_beam_properties(beam)  # Ensure rotation_matrix is computed
            # Compute distributed load magnitude
            q_magnitude = -beam.rho * g * beam.A  # N/m (negative for downward)
            # Transform gravity direction to local coordinates
            q_local_dir = beam.rotation_matrix @ global_load_direction
            # Local distributed loads
            qy_local = q_magnitude * q_local_dir[1]  # Local y-axis (along beam cross-section)
            qz_local = q_magnitude * q_local_dir[2]  # Local z-axis (perpendicular to beam)
            self_weight_loads.append({'beam_idx': idx, 'qy': qy_local, 'qz': qz_local})
            # Log for debugging
            n1 = nodes[beam.node1_idx]
            n2 = nodes[beam.node2_idx]
            is_column = abs(n1.y - n2.y) > abs(n1.x - n2.x) and abs(n1.y - n2.y) > abs(n1.z - n2.z)
            st.session_state.log.append(
                f"Beam {idx} (Nodes {beam.node1_idx}-{beam.node2_idx}, {'Column' if is_column else 'Beam'}): "
                f"q_magnitude={q_magnitude:.2f} N/m, q_local_dir={q_local_dir.tolist()}, "
                f"qy={qy_local:.2f} N/m, qz={qz_local:.2f} N/m, "
                f"rotation_matrix={beam.rotation_matrix.tolist()}"
            )
            # Validate column loads
            if is_column and abs(qy_local + beam.rho * g * beam.A) > 1e-6:
                st.session_state.log.append(
                    f"Warning: Column {idx} has unexpected qy={qy_local:.2f} N/m, expected ~{-beam.rho * g * beam.A:.2f} N/m"
                )
    return self_weight_loads

def run_single_analysis():
    if not st.session_state.model_defined:
        st.warning("Please define a model before running an analysis.")
        return

    with st.spinner("Preparing analysis..."):
        # Rebuild connections to ensure consistency
        attach = [[] for _ in st.session_state.nodes]
        for bid, beam in enumerate(st.session_state.beams):
            attach[beam.node1_idx].append(bid)
            attach[beam.node2_idx].append(bid)
        st.session_state.connections = [Connection(b1, b2, 'rigid') for lst in attach if len(lst) > 1 for b1, b2 in combinations(lst, 2)]

        # Compute self-weight and superimposed loads
        self_weight_loads = compute_self_weight_loads(st.session_state.nodes, st.session_state.beams)
        st.session_state.log.append(f"Applied self-weight and superimposed loads to {len(self_weight_loads)} beams with gravity direction {st.session_state.gravity_vector}.")

        # Combine all loads
        all_loads = self_weight_loads[:]
        for pack in st.session_state.loads:
            all_loads.extend(pack['data'])
        if not all_loads and len(st.session_state.nodes) > 2:
            all_loads.append({'node_idx': 0, 'Fx': 0, 'Fy': 0, 'Fz': 0})
            st.session_state.log.append("No user-defined loads. Applied default 0 N load on node 0.")

        analyzer = FrameAnalyzer(st.session_state.nodes, st.session_state.beams, st.session_state.connections)
        success = analyzer.solve(all_loads, st.session_state.boundary_conditions)
        if not success:
            st.error("Analysis failed. The structure may be unstable or improperly defined.")
            st.session_state.log.append("Analysis failed: Check node connectivity, supports, or profile properties.")
            return

    with st.spinner("Verifying design (AISC 360)..."):
        fy = float(st.session_state.fy)
        fu = float(st.session_state.fu)
        verifier = AISC360Verifier(analyzer, profile_db, fy, fu)
        try:
            uc_table = verifier.run()
            max_uc = uc_table["governing_uc"].max()
        except KeyError as e:
            st.error(f"Verification failed: Profile {e} not found in profiles.pkl.")
            st.session_state.log.append(f"Verification failed: {e}")
            return

    with st.spinner("Generating result plots..."):
        visualizer = Visualizer(analyzer)
        plot_moment = visualizer.plot_to_image(plot_type='moment_z')
        plot_axial = visualizer.plot_to_image(plot_type='axial_force')
        plot_deformed = visualizer.plot_to_image(plot_type='disp_y', deformed=True, amplification_factor=100)

    results_table = pd.DataFrame([
        {
            'Beam ID': r['beam_idx'],
            'Node 1': st.session_state.beams[r['beam_idx']].node1_idx,
            'Node 2': st.session_state.beams[r['beam_idx']].node2_idx,
            'Axial (N)': r['node1'].get('N', 0),
            'Moment Mx (Nm)': np.max([np.abs(r['node1'].get('Mx', 0)), np.abs(r['node2'].get('Mx', 0)) if 'node2' in r else 0]),
            'Moment My (Nm)': np.max([np.abs(r['node1'].get('My', 0)), np.abs(r['node2'].get('My', 0)) if 'node2' in r else 0]),
            'Moment Mz (Nm)': np.max([np.abs(r['node1'].get('Mz', 0)), np.abs(r['node2'].get('Mz', 0)) if 'node2' in r else 0]),
            'Shear Vx (N)': np.max([np.abs(r['node1'].get('Vx', 0)), np.abs(r['node2'].get('Vx', 0)) if 'node2' in r else 0]),
            'Shear Vy (N)': np.max([np.abs(r['node1'].get('Vy', 0)), np.abs(r['node2'].get('Vy', 0)) if 'node2' in r else 0]),
            'Shear Vz (N)': np.max([np.abs(r['node1'].get('Vz', 0)), np.abs(r['node2'].get('Vz', 0)) if 'node2' in r else 0]),
            'Max Stress (MPa)': max(
                max([abs(v) for v in r['node1']['stresses'].values()]),
                max([abs(v) for v in r['node2']['stresses'].values()]) if 'node2' in r else 0
            ) / 1e6
        } for r in analyzer.get_results()
    ])

    # Debug log for moment and stress values
    for r in analyzer.get_results():
        st.session_state.log.append(f"Beam {r['beam_idx']} raw moments: Mx={r['node1'].get('Mx', 0)}, My={r['node1'].get('My', 0)}, Mz={r['node1'].get('Mz', 0)}")
        st.session_state.log.append(f"Beam {r['beam_idx']} raw stresses: {r['node1']['stresses']}")

    mass = sum(bm.A * bm.rho * bm.length for bm in st.session_state.beams if hasattr(bm, 'length') and bm.length is not None)
    st.session_state.results = {
        "mass": mass,
        "max_uc": max_uc,
        "uc_table": uc_table,
        "plot_moment": plot_moment,
        "plot_axial": plot_axial,
        "plot_deformed": plot_deformed,
        "results_table": results_table,
        "analyzer": analyzer  # Add this line to store the solved analyzer instance
    }
    st.session_state.log.append(f"Single analysis completed: Mass={mass:,.2f} kg, Max UC={max_uc:.3f}")
    st.success(f"Analysis and verification complete! Total mass: {mass:,.2f} kg")
    time.sleep(1)

def run_optimization():
    if not st.session_state.col_profiles or not st.session_state.beam_profiles:
        st.warning("Please select one or more column and beam profiles to run optimization.")
        return

    st.session_state.log = []
    log_placeholder = st.empty()

    best_mass, best_pair, best_uc_table = float('inf'), None, None
    E, nu, rho = 2e11, 0.3, 7850
    fy = float(st.session_state.fy)
    fu = float(st.session_state.fu)

    if st.session_state.model_type == "Parametric":
        # Parametric model optimization
        n_stories = st.session_state.get("n_stories", "3")
        h_story = st.session_state.get("h_story", "4.0")
        bays_x = st.session_state.get("bays_x", "6,6")
        bays_z = st.session_state.get("bays_z", "8")
        use_custom_props = st.session_state.get("use_custom_props", False)
        custom_props = st.session_state.get("custom_props", None)

        combinations_to_test = list(product(st.session_state.col_profiles, st.session_state.beam_profiles))
        progress_bar = st.progress(0)

        for i, (col_sh, beam_sh) in enumerate(combinations_to_test):
            log_line = f"Testing C: {col_sh}, B: {beam_sh}..."
            st.session_state.log.append(log_line)
            log_placeholder.code("\n".join(st.session_state.log), language=None)

            nodes, beams, conns, bcs = build_parametric_model(n_stories, h_story, bays_x, bays_z, col_sh, beam_sh, E, nu, rho, use_custom_props, custom_props)
            # Compute self-weight loads
            self_weight_loads = compute_self_weight_loads(nodes, beams)
            st.session_state.log.append(f"  Applied self-weight loads to {len(self_weight_loads)} beams.")

            # Combine all loads
            all_loads = self_weight_loads[:]
            for pack in st.session_state.loads:
                all_loads.extend(pack['data'])
                st.session_state.log.append(f"  Applied user-defined load pack '{pack['name']}' with {len(pack['data'])} loads.")
            if not all_loads and len(nodes) > 0:
                top_nodes = [idx for idx, n in enumerate(nodes) if n.y == float(h_story) * int(n_stories)]
                if top_nodes:
                    all_loads.append({'node_idx': 0, 'Fx': 0, 'Fy': 0, 'Fz': 0})
                    st.session_state.log.append("  No user-defined loads. Applied default 0 N load on node 0.")

            # Log model details for debugging
            st.session_state.log.append(f"  Model: {len(nodes)} nodes, {len(beams)} beams, {len(all_loads)} loads")

            analyzer = FrameAnalyzer(nodes, beams, conns)
            if not analyzer.solve(all_loads, bcs):
                st.session_state.log.append("  ✗ Analysis FAILED.")
                continue

            verifier = AISC360Verifier(analyzer, profile_db, fy, fu)
            try:
                uc_table = verifier.run()
                uc_max = uc_table["governing_uc"].max()
            except KeyError as e:
                st.session_state.log.append(f"  ✗ Verification FAILED: Profile {e} not found.")
                continue

            mass = sum(bm.A * bm.rho * bm.length for bm in beams if hasattr(bm, 'length') and bm.length is not None)
            st.session_state.log.append(f"  Mass: {mass:,.2f} kg")

            if uc_max <= 1.0:
                st.session_state.log.append(f"  ✓ PASS (UC={uc_max:.2f}, Mass={mass:,.2f} kg)")
                if mass < best_mass:
                    best_mass = mass
                    best_pair = (col_sh, beam_sh)
                    best_uc_table = uc_table
                    st.session_state.log.append(f"    ★ NEW BEST! ★")
            else:
                st.session_state.log.append(f"  ✗ FAIL (UC={uc_max:.2f})")

            # Log axial forces for columns
            results_table = pd.DataFrame([
                {
                    'Beam ID': int(r['beam_idx']),
                    'Node 1': int(beams[r['beam_idx']].node1_idx),
                    'Node 2': int(beams[r['beam_idx']].node2_idx),
                    'Axial (N)': r['node1']['N']
                } for r in analyzer.get_results()
            ])
            for idx, row in results_table.iterrows():
                # Verify node indices are integers
                if not isinstance(row['Node 1'], (int, np.integer)) or not isinstance(row['Node 2'], (int, np.integer)):
                    st.session_state.log.append(
                        f"Warning: Non-integer node indices in optimization results_table: "
                        f"Node 1={row['Node 1']} (type={type(row['Node 1'])}), "
                        f"Node 2={row['Node 2']} (type={type(row['Node 2'])})"
                    )
                n1 = nodes[int(row['Node 1'])]
                n2 = nodes[int(row['Node 2'])]
                if abs(n1.y - n2.y) > abs(n1.x - n2.x) and abs(n1.y - n2.y) > abs(n1.z - n2.z):
                    axial = row['Axial (N)']
                    st.session_state.log.append(
                        f"  Column {int(row['Beam ID'])} (Nodes {int(row['Node 1'])}-{int(row['Node 2'])}): "
                        f"Axial force = {axial:.2f} N {'(tension)' if axial > 0 else '(compression)'}"
                    )

            progress_bar.progress((i + 1) / len(combinations_to_test))
    else:
        # Custom model optimization
        combinations_to_test = list(product(st.session_state.col_profiles, st.session_state.beam_profiles))
        progress_bar = st.progress(0)

        for i, (col_sh, beam_sh) in enumerate(combinations_to_test):
            log_line = f"Testing C: {col_sh}, B: {beam_sh}..."
            st.session_state.log.append(log_line)
            log_placeholder.code("\n".join(st.session_state.log), language=None)

            # Clone nodes and boundary conditions
            nodes = [Node(n.x, n.y, n.z) for n in st.session_state.nodes]
            for idx, node in enumerate(nodes):
                node.node_idx = idx
            bcs = st.session_state.boundary_conditions[:]

            # Rebuild beams with new profiles
            beams = []
            attach = [[] for _ in nodes]
            for bid, orig_beam in enumerate(st.session_state.beams):
                # Determine if beam is a column or beam based on orientation
                n1 = nodes[orig_beam.node1_idx]
                n2 = nodes[orig_beam.node2_idx]
                is_column = abs(n1.y - n2.y) > abs(n1.x - n2.x) and abs(n1.y - n2.y) > abs(n1.z - n2.z)
                profile = col_sh if is_column else beam_sh
                beam_props = profile_manager.get_beam_properties(profile, E, nu, rho)
                new_beam = Beam(
                    orig_beam.node1_idx,
                    orig_beam.node2_idx,
                    **beam_props,
                    ref_vector=orig_beam.ref_vector
                )
                new_beam.profile_name = profile
                n1_pos = np.array([nodes[new_beam.node1_idx].x, nodes[new_beam.node1_idx].y, nodes[new_beam.node1_idx].z])
                n2_pos = np.array([nodes[new_beam.node2_idx].x, nodes[new_beam.node2_idx].y, nodes[new_beam.node2_idx].z])
                new_beam.length = np.linalg.norm(n2_pos - n1_pos)
                beams.append(new_beam)
                attach[new_beam.node1_idx].append(bid)
                attach[new_beam.node2_idx].append(bid)

            # Rebuild connections
            conns = [Connection(b1, b2, 'rigid') for lst in attach if len(lst) > 1 for b1, b2 in combinations(lst, 2)]
            st.session_state.log.append(f"  Generated {len(conns)} connections.")

            # Compute self-weight loads
            self_weight_loads = compute_self_weight_loads(nodes, beams)
            st.session_state.log.append(f"  Applied self-weight loads to {len(self_weight_loads)} beams.")

            # Combine all loads
            all_loads = self_weight_loads[:]
            for pack in st.session_state.loads:
                all_loads.extend(pack['data'])
                st.session_state.log.append(f"  Applied user-defined load pack '{pack['name']}' with {len(pack['data'])} loads.")
            if not all_loads and len(nodes) > 0:
                all_loads.append({'node_idx': min(2, len(nodes)-1), 'Fx': 0, 'Fy': -1, 'Fz': 0})
                st.session_state.log.append("  No user-defined loads. Applied default -1 N load on node.")

            # Log model details for debugging
            st.session_state.log.append(f"  Model: {len(nodes)} nodes, {len(beams)} beams, {len(all_loads)} loads")

            analyzer = FrameAnalyzer(nodes, beams, conns)
            if not analyzer.solve(all_loads, bcs):
                st.session_state.log.append("  ✗ Analysis FAILED.")
                continue

            verifier = AISC360Verifier(analyzer, profile_db, fy, fu)
            try:
                uc_table = verifier.run()
                uc_max = uc_table["governing_uc"].max()
            except KeyError as e:
                st.session_state.log.append(f"  ✗ Verification FAILED: Profile {e} not found.")
                continue

            mass = sum(bm.A * bm.rho * bm.length for bm in beams if hasattr(bm, 'length') and bm.length is not None)
            st.session_state.log.append(f"  Mass: {mass:,.2f} kg")

            if uc_max <= 1.0:
                st.session_state.log.append(f"  ✓ PASS (UC={uc_max:.2f}, Mass={mass:,.2f} kg)")
                if mass < best_mass:
                    best_mass = mass
                    best_pair = (col_sh, beam_sh)
                    best_uc_table = uc_table
                    st.session_state.log.append(f"    ★ NEW BEST! ★")
            else:
                st.session_state.log.append(f"  ✗ FAIL (UC={uc_max:.2f})")

            # Log axial forces for columns
            results_table = pd.DataFrame([
                {
                    'Beam ID': int(r['beam_idx']),
                    'Node 1': int(beams[r['beam_idx']].node1_idx),
                    'Node 2': int(beams[r['beam_idx']].node2_idx),
                    'Axial (N)': r['node1']['N']
                } for r in analyzer.get_results()
            ])
            for idx, row in results_table.iterrows():
                # Verify node indices are integers
                if not isinstance(row['Node 1'], (int, np.integer)) or not isinstance(row['Node 2'], (int, np.integer)):
                    st.session_state.log.append(
                        f"Warning: Non-integer node indices in optimization results_table: "
                        f"Node 1={row['Node 1']} (type={type(row['Node 1'])}), "
                        f"Node 2={row['Node 2']} (type={type(row['Node 2'])})"
                    )
                n1 = nodes[int(row['Node 1'])]
                n2 = nodes[int(row['Node 2'])]
                if abs(n1.y - n2.y) > abs(n1.x - n2.x) and abs(n1.y - n2.y) > abs(n1.z - n2.z):
                    axial = row['Axial (N)']
                    st.session_state.log.append(
                        f"  Column {int(row['Beam ID'])} (Nodes {int(row['Node 1'])}-{int(row['Node 2'])}): "
                        f"Axial force = {axial:.2f} N {'(tension)' if axial > 0 else '(compression)'}"
                    )

            progress_bar.progress((i + 1) / len(combinations_to_test))

    if best_pair:
        st.session_state.log.append("\n--- OPTIMIZATION COMPLETE ---")
        st.session_state.log.append(f"Lightest Passing Design: C: {best_pair[0]}, B: {best_pair[1]}")
        st.session_state.log.append(f"Mass: {best_mass:,.2f} kg")
        st.session_state.optimization_results = {
            "mass": best_mass,
            "profiles": best_pair,
            "uc_table": best_uc_table
        }
        st.success(f"Optimization complete! Lightest design found: {best_pair[0]}/{best_pair[1]}, Mass: {best_mass:,.2f} kg")
    else:
        st.session_state.log.append("\n--- OPTIMIZATION COMPLETE ---")
        st.session_state.log.append("No passing design found in the selected profiles.")
        st.session_state.optimization_results = None
        st.warning("Optimization complete. No passing design was found.")

    log_placeholder.code("\n".join(st.session_state.log), language=None)

# --- Main Execution ---
profile_db, profile_list = load_profile_data("profiles.pkl")

if not profile_list:
    st.error("Could not load `profiles.pkl`. Please reinstall the package or check the installation.")
else:
    draw_sidebar()
    draw_main_content()
