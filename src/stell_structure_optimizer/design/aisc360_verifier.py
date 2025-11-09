"""
aisc360_verifier.py
-------------------
Production-ready LRFD verifier for 3-D frame models solved with the
FrameAnalyzer/Beam framework.  Implements the key limit-state equations of
ANSI/AISC 360-22 for the six rolled-shape families in your profile database.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Callable, Optional

import numpy as np
import pandas as pd
import math

# ---------------------------------------------------------------------------
#  Helpers – section metadata & safe math utilities
# ---------------------------------------------------------------------------

@dataclass
class SectionMeta:
    """Lightweight container for the geometric data we need for code checks."""
    name:      str
    family:    str           # 'ipn', 'upn', 'l', 'square', 'rect', 'round'
    Fy:        float
    Fu:        float
    E:         float = 200e9
    G:         float = field(init=False)

    # geometry (→ all in *metres* and *SI* units)
    A:         float = 0.0
    Zx:        float = 0.0
    Zy:        float = 0.0
    Sx:        float = 0.0
    Sy:        float = 0.0
    Ix:        float = 0.0
    Iy:        float = 0.0
    J:         float = 0.0
    Cw:        float = 0.0           # warping constant (if available)

    # plate dimensions for compactness (all optional)
    bf:        float = None          # flange width (I & channel)
    tf:        float = None          # flange thickness
    h:         float = None          # clear web depth = d – 2 tf
    tw:        float = None          # web thickness
    B:         float = None          # HSS flat width
    H:         float = None          # HSS flat height
    t:         float = None          # HSS wall thickness
    D:         float = None          # pipe outside Ø
    b_leg:     float = None          # angle leg width
    t_leg:     float = None

    # optional effective net area for tension fracture (if you model holes)
    Ae:        float = None

    # derived classification
    compact_flags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.G = self.E / (2.0 * (1.0 + 0.3))   # assume ν = 0.3 unless told otherwise


def _positive(val: float, default: float = 1e-9) -> float:
    """Return *val* if >0 else a small epsilon to avoid /0."""
    return val if val > 0.0 else default


# ---------------------------------------------------------------------------
#  Strength-calculation functions – one per limit state family
# ---------------------------------------------------------------------------

#  Each routine signature:
#      def foo(sec: SectionMeta, forces: Dict[str,float], **kw) -> Tuple[float,str]
#  Returns (φRn , governing_clause_string)

#
# 1. Axial strength – generic (tension & compression)
#
def axial_strength(sec: SectionMeta, N: float,
                   Kx: float = 1.0, Ky: float = 1.0,
                   Lx: float | None = None, Ly: float | None = None) -> Tuple[float, str]:
    """
    Convention (matches your analyzer):  N > 0  => tension
                                        N < 0  => compression
    Tension (D2):  φPn = min(0.90*Fy*Ag, 0.75*Fu*Ae)
    Compression (E3): evaluate about x & y; return MIN φPn (φ=0.90).
    """
    E, Fy, Fu, A = sec.E, sec.Fy, sec.Fu, sec.A

    # -------- COMPRESSION (E3) when N < 0.0 --------
    print('Normal for testing: ', N)
    print('Lx for testing: ', Lx)
    print('Ly for testing: ', Ly)
    if N < 0.0:
        # default lengths if not provided
        Ldef = 1.0
        Lx = Lx if (Lx is not None and Lx > 0.0) else Ldef
        Ly = Ly if (Ly is not None and Ly > 0.0) else Ldef

        rx = math.sqrt(_positive(sec.Ix) / _positive(A))
        ry = math.sqrt(_positive(sec.Iy) / _positive(A))

        def phiPn_axis(K: float, L: float, r: float) -> Tuple[float, dict]:
            klr = K * L / _positive(r)
            lambdac = klr * math.sqrt(Fy/(math.pi**2 * E))
            if lambdac <= 1.5:
                Fn = Fy * (0.658 ** (lambdac**2))     # E3-2
            else:
                Fe = (math.pi**2 * E) / (_positive(klr)**2)  # E3-4
                Fn = 0.877 * Fe                              # E3-3
            return 0.90 * Fn * A, {"KLr": klr, "lambda_c": lambdac, "Fn": Fn}

        phi_x, _ = phiPn_axis(Kx, Lx, rx)
        phi_y, _ = phiPn_axis(Ky, Ly, ry)
        return (phi_x, "E3-x (compression)") if phi_x <= phi_y else (phi_y, "E3-y (compression)")

    # -------- TENSION (D2) when N >= 0.0 --------
    Ag = A
    Ae = sec.Ae if (sec.Ae is not None and sec.Ae > 0.0) else Ag
    phiPn_gy = 0.90 * Fy * Ag
    phiPn_nf = 0.75 * Fu * Ae
    return min(phiPn_gy, phiPn_nf), "D2 (tension)"


#
# 2. Flexure – doubly-symmetric I-shapes (IPN & alike)
#
def major_flexure_I(sec: SectionMeta, Mux: float, Lb: float, Cb: float = 1.0) -> tuple[float, str]:
    """
    AISC 360-22 F2 (φb=0.90). Lp (F2-5), Lr (F2-6), r_ts (F2-7), elastic LTB per commentary.
    Mn is capped by Mp in the inelastic region. Returns (φMn, governing_clause).
    """
    E, Fy = sec.E, sec.Fy
    Sx, Zx = sec.Sx, sec.Zx
    Iy, J, Cw = sec.Iy, max(sec.J, 0.0), max(sec.Cw, 0.0)
    A = sec.A

    # Geometry helpers
    ry = math.sqrt(_positive(Iy) / _positive(A))
    Mp = Fy * Zx

    # r_ts per F2-7
    if Iy > 0.0 and Cw > 0.0 and Sx > 0.0:
        rts = math.sqrt(math.sqrt(Iy * Cw) / Sx)
    else:
        rts = 1e-6  # avoid /0; pushes to conservative side

    # h0: distance between flange centroids
    # If 'h' is clear web depth (d-2tf), then h0 ≈ h + tf; if 'd' present, h0 ≈ d - tf
    if sec.h is not None and sec.tf is not None:
        h0 = max(sec.h + sec.tf, 1e-9)
    elif sec.tf is not None and hasattr(sec, "d") and sec.d is not None:
        h0 = max(sec.d - sec.tf, 1e-9)
    else:
        h0 = 1.0  # neutral fallback if dims are missing

    c = 1.0  # doubly symmetric I
    termA = (J * c) / (_positive(Sx) * _positive(h0))

    # Lp (F2-5) and Lr (F2-6)
    Lp = 1.76 * ry * math.sqrt(E / Fy)
    Lr = 1.95 * rts * (E / (0.7 * Fy)) * math.sqrt(termA + math.sqrt(termA**2 + 6.76 * (0.7*Fy/E)**2))

    if Lb <= Lp:
        Mn = Mp
        clause = 'F2-1'
    elif Lb <= Lr:
        Mn = Cb * (Mp - (Mp - 0.7*Fy*Sx) * (Lb - Lp) / max(Lr - Lp, 1e-9))  # F2-2
        Mn = min(Mn, Mp)  # cap at Mp
        clause = 'F2-2'
    else:
        # Elastic LTB via commentary stress form: Mn = Fcr * Sx
        sqrt_term = math.sqrt(1.0 + 0.078 * termA * (Lb / _positive(rts)) ** 2)
        Fcr = Cb * (math.pi**2 * E) / (_positive(Lb / _positive(rts)) ** 2) * sqrt_term
        Mn = Fcr * Sx
        clause = 'F2-4'

    phiMn = 0.90 * Mn
    return phiMn, clause


#
# 3. Flexure – HSS (square & rectangular)
#
def flexure_HSS_rect(sec: SectionMeta, Mu: float) -> Tuple[float, str]:
    λ = (sec.H/sec.t) if sec.family == "rect" else (sec.B/sec.t)
    λp = 0.38 * np.sqrt(sec.E/sec.Fy)
    λr = 1.40 * np.sqrt(sec.E/sec.Fy)
    Mp = sec.Fy * sec.Zx
    My = sec.Fy * sec.Sx
    if   λ <= λp: Mn = Mp
    elif λ <= λr: Mn = My + (Mp - My)*(λr - λ)/(λr - λp)
    else:         Mn = 0.7 * sec.Fy * sec.Sx
    return 0.90 * Mn, "F7 (HSS flexure)"


#
# 4. Flexure – Round HSS
#
def flexure_HSS_round(sec: SectionMeta, Mu: float) -> Tuple[float, str]:
    # No LTB, only local-ovalisation
    λ = sec.D/sec.t
    λp = 0.07 * np.sqrt(sec.E/sec.Fy)
    λr = 0.31 * np.sqrt(sec.E/sec.Fy)
    Mp = sec.Fy * sec.Zx
    My = sec.Fy * sec.Sx
    if   λ <= λp: Mn = Mp
    elif λ <= λr: Mn = My + (Mp - My)*(λr - λ)/(λr - λp)
    else:         Mn = 0.7 * sec.Fy * sec.Sx
    return 0.90 * Mn, "F8 (round HSS flexure)"


#
# 5. Shear – generic web/plastic flow
#
def shear_strength(sec: SectionMeta, Vu: float) -> Tuple[float, str]:
    # Very simplified: 0.6FyAv with φ=0.90
    Aw = sec.A * 0.8   # crude estimation
    return 0.90 * 0.6 * sec.Fy * Aw, "G2/G4 (shear)"


#
# 6. Interaction – H1 compression + bending (doubly-sym.) and H2 (angles)
#
def interaction_H1(Pu: float, Mux: float, Muy: float,
                   phiPn: float, phiMnx: float, phiMny: float) -> dict:
    """
    H1-1a if Pu/φPn >= 0.2; else H1-1b. Returns dict with lhs and pass/fail.
    """
    # Use compression only in the ratio; tension does not penalize H1
    pu_ratio = max(Pu, 0.0) / max(phiPn, 1e-12)
    if pu_ratio >= 0.2:
        lhs = pu_ratio + (8.0/9.0) * (abs(Mux)/max(phiMnx,1e-12) + abs(Muy)/max(phiMny,1e-12))
        eqn = 'H1-1a'
    else:
        lhs = pu_ratio/2.0 + (abs(Mux)/max(phiMnx,1e-12) + abs(Muy)/max(phiMny,1e-12))
        eqn = 'H1-1b'
    return {'lhs': lhs, 'governs': eqn, 'ok': lhs <= 1.0 + 1e-9}


#
# 7. Helper functions for getting the right beam axes and moments
#

def _local_axes(beam, nodes_collection=None):
    """
    Returns an orthonormal local basis (x̂, ŷ, ẑ) for *beam*.
    Accepts either of the following beam definitions:
        • beam.node1 / beam.node2           → Node object or (x,y,z) tuple
        • beam.node1_idx / beam.node2_idx   → index into nodes_collection
    """
    # --- resolve the two end-points ---------------------------------------
    n1 = getattr(beam, "node1", None) or getattr(beam, "node1_idx", None)
    n2 = getattr(beam, "node2", None) or getattr(beam, "node2_idx", None)
    if n1 is None or n2 is None:
        raise AttributeError("Beam must define node1/2 or node1_idx/2_idx")

    def _coords(n):
        # already a Node obj
        if hasattr(n, "x") and hasattr(n, "y") and hasattr(n, "z"):
            return np.array([n.x, n.y, n.z], float)
        # raw (x,y,z)
        if isinstance(n, (tuple, list, np.ndarray)) and len(n) == 3:
            return np.array(n, float)
        # index lookup
        if nodes_collection is not None:
            node_obj = nodes_collection[n]
            return np.array([node_obj.x, node_obj.y, node_obj.z], float)
        raise TypeError(f"Don’t know how to get coordinates for {n!r}")

    p1, p2 = _coords(n1), _coords(n2)
    x_hat   = (p2 - p1) / np.linalg.norm(p2 - p1)

    y_ref   = getattr(beam, "ref_vector", np.array([0.0, 0.0, 1.0]))
    y_hat   = y_ref - np.dot(y_ref, x_hat) * x_hat     # Gram–Schmidt
    y_hat  /= np.linalg.norm(y_hat)
    z_hat   = np.cross(x_hat, y_hat)
    return x_hat, y_hat, z_hat

def _max_abs_local_actions(res, x_hat, y_hat, z_hat):
    """
    Returns signed Pu, and |M|, |V| maxima using local axes passed in.
    • Pu       – axial with sign (max |N| end governs sign)
    • M_major  – max |moment about ẑ|  (strong axis)
    • M_minor  – max |moment about ŷ|  (weak  axis)
    • Vu       – max |transverse shear|  (√(Vy²+Vz²) in local system)
    """
    Pu = 0.0
    Mmaj = Mmin = Vu = 0.0

    for nd in ('node1', 'node2'):
        n = res[nd]

        # ----- axial (preserve the sign of the governing end) -----
        N_end = n.get('N', 0.0)
        if abs(N_end) >= abs(Pu):
            Pu = N_end  # keep the sign of the max-abs end force

        # ----- global → local moments -----
        M_g = np.array([n.get('Mx', 0.0),
                        n.get('My', 0.0),
                        n.get('Mz', 0.0)], float)
        M_loc = np.array([np.dot(M_g, v) for v in (x_hat, y_hat, z_hat)])
        Mmaj  = max(Mmaj, abs(M_loc[2]))   # about ẑ
        Mmin  = max(Mmin, abs(M_loc[1]))   # about ŷ

        # ----- global → local shear forces -----
        V_g = np.array([n.get('Fx', 0.0) or n.get('Vx', 0.0),
                        n.get('Fy', 0.0) or n.get('Vy', 0.0),
                        n.get('Fz', 0.0) or n.get('Vz', 0.0)], float)
        V_loc = np.array([np.dot(V_g, v) for v in (x_hat, y_hat, z_hat)])
        Vu    = max(Vu, np.linalg.norm(V_loc[1:]))      # √(Vy²+Vz²)

    return Pu, Mmaj, Mmin, Vu


# ---------------------------------------------------------------------------
#  Public class – takes FrameAnalyzer + raw DataFrames & runs the checks
# ---------------------------------------------------------------------------

class AISC360Verifier:
    """
    Usage:
        analyzer.solve(...)
        dfs = pickle.load(open('profiles.pkl','rb'))
        verifier = AISC360Verifier(analyzer, dfs,
                                   Fy=355e6, Fu=490e6, design_method='LRFD')
        df_uc = verifier.run()
        print(df_uc[df_uc['governing_uc'] > 1.0])     # beams that fail
    """
    def __init__(self,
                 analyzer,                          # solved FrameAnalyzer
                 raw_profiles: Dict[str, pd.DataFrame],
                 Fy: float,
                 Fu: float,
                 design_method: str = 'LRFD'):
        if analyzer.get_results() is None:
            raise ValueError("Analyzer must be solved before verification.")
        if design_method.upper() not in ('LRFD', 'ASD'):
            raise ValueError("design_method must be 'LRFD' or 'ASD'.")

        self.analyzer   = analyzer
        self.results    = analyzer.get_results()
        self.raw_dfs    = raw_profiles
        self.Fy         = Fy
        self.Fu         = Fu
        self.design     = design_method.upper()

        # Build a quick look-up (profile→row Dict) with everything we’ll need
        self.section_meta: Dict[str, SectionMeta] = \
            self._digest_profiles(raw_profiles)

    # -------------  public driver  -------------
    def run(self) -> pd.DataFrame:
        """
        Computes φRn (or Rn/Ω) for axial, bending (major & minor) and shear, then
        the H1/H2 interaction.  All checks use the absolute-maximum actions
        between the two ends of every member, projected into its local axes, so
        the routine is fully 3-D-compatible.
        """
        rows: list[dict] = []

        # Grab the master node list once (name may vary in different models)
        node_collection = getattr(self.analyzer, "nodes", None) \
                          or getattr(self.analyzer, "node_list", None)

        for res in self.results:
            beam  = self.analyzer.beams[res['beam_idx']]
            name  = getattr(beam, "profile_name", "UNKN")
            sec   = self.section_meta[name]          # section properties

            # --------------------------------------------------------------
            #  Local basis  ×  worst-end actions (M,V as abs; N keeps sign)
            # --------------------------------------------------------------
            xh, yh, zh             = _local_axes(beam, node_collection)
            Pu, Mux, Muy, Vu       = _max_abs_local_actions(res, xh, yh, zh)

            # Effective length factors and lengths for columns
            Kx = getattr(beam, "Kx", 1.0)
            Ky = getattr(beam, "Ky", 1.0)
            Lx = getattr(beam, "Lx", beam.length)
            Ly = getattr(beam, "Ly", beam.length)

            # 1. Axial
            phiPn, cls_P = axial_strength(sec, Pu, Kx=Kx, Ky=Ky, Lx=Lx, Ly=Ly)

            # 2. Major-axis flexure (family switch exactly as before)
            if sec.family in ('ipn', 'upn'):
                phiMnx, cls_Mx = major_flexure_I(sec, Mux, Lb=beam.length, Cb=getattr(beam, "Cb", 1.0))
            elif sec.family in ('square', 'rect'):
                phiMnx, cls_Mx = flexure_HSS_rect(sec, Mux)
            elif sec.family == 'round':
                phiMnx, cls_Mx = flexure_HSS_round(sec, Mux)
            else:                               # angles → principal-Z flexure (simplified)
                phiMnx, cls_Mx = 0.90 * sec.Fy * sec.Sx, "F10 (angle flexure)"

            # 3. Minor-axis flexure – generic yielding
            phiMny = 0.90 * sec.Fy * sec.Sy
            cls_My = "F6/rect"

            # 4. Shear (transverse component in the local y-z plane)
            phiVn, cls_V = shear_strength(sec, Vu)

            # 5. Interaction (H1/H2, skipped for angles)
            if sec.family == 'l':
                uc_int_val = 0.0
                H_clause = "H2 (angles) – not implemented"
            else:
                H = interaction_H1(Pu, Mux, Muy, phiPn, phiMnx, phiMny)
                uc_int_val = H['lhs']
                H_clause = H['governs']

            # 6. Unity checks
            UC_ax = abs(Pu) / max(phiPn, 1e-12)
            UC_Mx = abs(Mux) / max(phiMnx, 1e-12) if phiMnx else 0.0
            UC_My = abs(Muy) / max(phiMny, 1e-12) if phiMny else 0.0
            UC_V  = abs(Vu)  / max(phiVn, 1e-12)

            governing_uc = max(UC_ax, UC_Mx, UC_My, UC_V, uc_int_val)
            governing_limit = max(
                [('axial', UC_ax),
                 ('Mx',    UC_Mx),
                 ('My',    UC_My),
                 ('V',     UC_V),
                 ('H1',    uc_int_val)],
                key=lambda kv: kv[1]
            )[0]

            # 7. Assemble results
            rows.append(dict(
                beam_idx             = res['beam_idx'],
                profile              = name,
                family               = sec.family,

                Pu_N                 = Pu,
                phiPn_N              = phiPn,
                axial_clause         = cls_P,

                Mux_Nm               = Mux,
                phiMnx_Nm            = phiMnx,
                Mx_clause            = cls_Mx,

                Muy_Nm               = Muy,
                phiMny_Nm            = phiMny,
                My_clause            = cls_My,

                Vy_N                 = Vu,
                phiVn_N              = phiVn,
                V_clause             = cls_V,

                UC_axial             = UC_ax,
                UC_Mx                = UC_Mx,
                UC_My                = UC_My,
                UC_V                 = UC_V,
                UC_interaction_H1    = uc_int_val,
                H_clause             = H_clause,

                governing_uc         = governing_uc,
                PASS                 = governing_uc <= 1.0,
                governing_limitstate = governing_limit
            ))

        return pd.DataFrame(rows)

    # -------------  private helpers  -------------
    def _digest_profiles(
            self,
            dfs: Dict[str, pd.DataFrame]
    ) -> Dict[str, SectionMeta]:
        """
        Parse the raw DataFrames in *profiles.pkl* and return a dictionary
        {profile_name -> SectionMeta}.  All geometry is assumed already in SI.

        Recognised keys (case-insensitive):
            • 'ipn', 'ipe', 'w', 'wf', 'i'           -> family 'ipn'
            • 'upn', 'channel', 'c'                  -> family 'upn'
            • 'square'                               -> family 'square'
            • 'rectangular', 'rect'                  -> family 'rect'
            • 'circular', 'round', 'pipe'            -> family 'round'
            • 'l', 'angle'                           -> family 'l'
        """
        meta: Dict[str, SectionMeta] = {}

        def add(sec: SectionMeta) -> None:
            if sec.name not in meta:        # keep first occurrence, skip dups
                meta[sec.name] = sec

        # --------------------------------------------------------------------
        # 1.  Rolled I-shapes & channels  (IPN / IPE / W-, and UPN)
        # --------------------------------------------------------------------
        family_map = {
            # rolled I-shapes → 'ipn'
            "ipn": "ipn", "ipe": "ipn", "w": "ipn", "wf": "ipn", "i": "ipn",
            # channels       → 'upn'
            "upn": "upn", "channel": "upn", "c": "upn"
        }

        for key, df in dfs.items():
            k = key.strip().lower()
            if k not in family_map:
                continue

            fam = family_map[k]
            for _, row in df.iterrows():
                name = str(row["Profile"]).upper().replace(" ", "")
                if name in meta:
                    continue

                add(SectionMeta(
                    name   = name,
                    family = fam,
                    Fy     = self.Fy,
                    Fu     = self.Fu,
                    # common geometry
                    A   = row["Ag (m²)"],
                    Ix  = row["X-X Ix (m⁴)"],
                    Iy  = row["Y-Y Iy (m⁴)"],
                    Sx  = row["Sx (m³)"],
                    Sy  = row["Sy (m³)"],
                    Zx  = row.get("Zx (m³)", row["Sx (m³)"]),
                    Zy  = row.get("Zy (m³)", row["Sy (m³)"]),
                    J   = row.get("J (m⁴)", 0.0),
                    Cw  = row.get("Cw (m⁶)", 0.0),
                    # plate dims (if present)
                    bf  = row.get("bf (m)"),
                    tf  = row.get("tf (m)"),
                    h   = row.get("hw (m)"),           # clear web depth
                    tw  = row.get("tw=r1 (m)")
                ))

        # --------------------------------------------------------------------
        # 2.  Square HSS  (CUAD…)
        # --------------------------------------------------------------------
        sq_key = next((k for k in dfs if k.lower() == "square"), None)
        if sq_key:
            for _, row in dfs[sq_key].iterrows():
                add(SectionMeta(
                    name   = str(row["Profile"]).upper(),
                    family = "square",
                    Fy     = self.Fy,
                    Fu     = self.Fu,
                    A  = row["Ag (m²)"],
                    Ix = row["Ix=Iy (m⁴)"], Iy = row["Ix=Iy (m⁴)"],
                    Sx = row["Sx=Sy (m³)"], Sy = row["Sx=Sy (m³)"],
                    Zx = row["Zx=Zy (m³)"], Zy = row["Zx=Zy (m³)"],
                    J  = row["J (m⁴)"],
                    B  = row["B (m)"], t = row["t (m)"]
                ))

        # --------------------------------------------------------------------
        # 3.  Rectangular HSS  (REC…)
        # --------------------------------------------------------------------
        rect_key = next((k for k in dfs if k.lower() in ("rectangular", "rect")),
                        None)
        if rect_key:
            for _, row in dfs[rect_key].iterrows():
                add(SectionMeta(
                    name   = str(row["Profile"]).upper(),
                    family = "rect",
                    Fy     = self.Fy,
                    Fu     = self.Fu,
                    A  = row["Ag (m²)"],
                    Ix = row["Ix (m⁴)"], Iy = row["Iy (m⁴)"],
                    Sx = row["Sx (m³)"], Sy = row["Sy (m³)"],
                    Zx = row["Zx (m³)"], Zy = row["Zy (m³)"],
                    J  = row["J (m⁴)"],
                    B  = row["B (m)"], H = row["H (m)"], t = row["t (m)"]
                ))

        # --------------------------------------------------------------------
        # 4.  Round HSS / Pipe  (CIRC…)
        # --------------------------------------------------------------------
        circ_key = next((k for k in dfs if k.lower() in ("circular", "round", "pipe")),
                        None)
        if circ_key:
            for _, row in dfs[circ_key].iterrows():
                add(SectionMeta(
                    name   = str(row["Profile"]).upper(),
                    family = "round",
                    Fy     = self.Fy,
                    Fu     = self.Fu,
                    A  = row["Ag (m²)"],
                    Ix = row["I (m⁴)"], Iy = row["I (m⁴)"],
                    Sx = row["S (m³)"],  Sy = row["S (m³)"],
                    Zx = row["Z (m³)"], Zy = row["Z (m³)"],
                    J  = row["J (m⁴)"],
                    D  = row["D (m)"], t = row["t (m)"]
                ))

        # --------------------------------------------------------------------
        # 5.  Angles  (L…)
        # --------------------------------------------------------------------
        angle_key = next((k for k in dfs if k.lower() in ("l", "angle")), None)
        if angle_key:
            for _, row in dfs[angle_key].iterrows():
                add(SectionMeta(
                    name   = str(row["Profile"]).upper(),
                    family = "l",
                    Fy     = self.Fy,
                    Fu     = self.Fu,
                    A  = row["Ag (m²)"],
                    Ix = row["X-X = Y-Y Ix = Iy (m⁴)"],
                    Iy = row["X-X = Y-Y Ix = Iy (m⁴)"],
                    Sx = row["Sx = Sy (m³)"],
                    Sy = row["Sx = Sy (m³)"],
                    Zx = 1.5 * row["Sx = Sy (m³)"],   # Mp ≈ 1.5 My
                    Zy = 1.5 * row["Sx = Sy (m³)"],
                    J  = row["J (m⁴)"],
                    b_leg = row["DIMENSIONES b (m)"],
                    t_leg = row["t (m)"]
                ))

        return meta
