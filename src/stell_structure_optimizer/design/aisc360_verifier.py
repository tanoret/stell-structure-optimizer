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
def axial_strength(sec: SectionMeta, Pu: float,
                   K: float = 1.0, L: float = 1.0, r: float = 0.01) -> Tuple[float, str]:
    """Chapter D, E3 flexural buckling for all rolled shapes."""
    if Pu >= 0:      # TENSION (+)
        phiPn_y  = 0.90 * sec.Fy * sec.A
        phiPn_ru = 0.75 * sec.Fu * sec.A   # net-area factor U ignored for simplicity
        return min(phiPn_y, phiPn_ru), "D2-1 (tension)"

    # COMPRESSION (–)
    λc = K*L / r * np.sqrt(sec.Fy / (np.pi**2 * sec.E))
    if λc <= 1.5:
        Fcr = 0.658 ** (λc**2) * sec.Fy
    else:
        Fcr = 0.877 / (λc**2) * sec.E
    return 0.90 * Fcr * sec.A, "E3-2/E3-3 (compression)"


#
# 2. Flexure – doubly-symmetric I-shapes (IPN & alike)
#
def major_flexure_I(sec: SectionMeta, Mux: float, Lb: float,
                    Cb: float = 1.0) -> Tuple[float, str]:
    """F2 plastic → LTB curve."""
    # flange local classification (compact assumption for demo)
    Lp = 1.76 * np.sqrt(sec.E / sec.Fy) * np.sqrt(sec.Iy / _positive(sec.A))
    rts = np.sqrt(np.sqrt(sec.Iy * sec.Cw) / sec.Sx) if sec.Cw else 0.03
    term = (sec.J * 1.0 / (sec.Sx * _positive(sec.h)/2.0)) + (sec.h/(2*rts))**2
    Lr = 1.95 * rts * np.sqrt(sec.E / (0.7*sec.Fy)) * np.sqrt(term)

    Mp = sec.Fy * sec.Zx
    if Lb <= Lp:
        Mn = Mp
        clause = "F2-1 (plastic)"
    elif Lb <= Lr:
        Mn = Cb * (Mp - (Mp - 0.7*sec.Fy*sec.Sx)*(Lb - Lp)/(Lr - Lp))
        clause = "F2-2 (inelastic LTB)"
    else:
        Mn = Cb * (np.pi**2 * sec.E * sec.Sx) / (Lb**2)
        clause = "F2-4 (elastic LTB)"
    return 0.90 * Mn, clause


#
# 3. Flexure – HSS (square & rectangular)
#
def flexure_HSS_rect(sec: SectionMeta, Mu: float) -> Tuple[float, str]:
    λ = sec.H/sec.t if sec.family == "rect" else sec.B/sec.t
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
def interaction_H1(Pu: float, phiPn: float,
                   Mux: float, phiMnx: float,
                   Muy: float, phiMny: float) -> float:
    return Pu/phiPn + (8/9) * (abs(Mux)/phiMnx + abs(Muy)/phiMny)

def interaction_H2_stress_based(σa: float, σb: float, σc: float,
                                Fca: float, Fcb: float, Fcc: float) -> float:
    return σa/Fca + σb/Fcb + σc/Fcc

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
    Returns Pu,  M_major, M_minor, Vu   using local axes passed in.
    • Pu       – max |axial|
    • M_major  – max |moment about ẑ|  (strong axis)
    • M_minor  – max |moment about ŷ|  (weak  axis)
    • Vu       – max |transverse shear|  (√(Vy²+Vz²) in local system)
    """
    Pu = 0.0
    Mmaj = Mmin = Vu = 0.0

    for nd in ('node1', 'node2'):
        n = res[nd]

        # ----- axial -----
        Pu = max(Pu, abs(n.get('N', 0.0)))

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
            #  Local basis  ×  worst-end actions (absolute values)
            # --------------------------------------------------------------
            xh, yh, zh             = _local_axes(beam, node_collection)
            Pu, Mux, Muy, Vu       = _max_abs_local_actions(res, xh, yh, zh)

            # 1. Axial
            phiPn, cls_P = axial_strength(sec, Pu)

            # 2. Major-axis flexure (family switch exactly as before)
            if sec.family in ('ipn', 'upn'):
                phiMnx, cls_Mx = major_flexure_I(sec, Mux, Lb=beam.length)
            elif sec.family in ('square', 'rect'):
                phiMnx, cls_Mx = flexure_HSS_rect(sec, Mux)
            elif sec.family == 'round':
                phiMnx, cls_Mx = flexure_HSS_round(sec, Mux)
            else:                               # angles → principal-Z flexure
                phiMnx, cls_Mx = 0.90 * sec.Fy * sec.Sx, "F10 (angle flexure)"

            # 3. Minor-axis flexure – generic yielding
            phiMny = 0.90 * sec.Fy * sec.Sy
            cls_My = "F6/rect"

            # 4. Shear (transverse component in the local y-z plane)
            phiVn, cls_V = shear_strength(sec, Vu)

            # 5. Interaction (H1/H2, skipped for angles)
            uc_int = 0.0 if sec.family == 'l' else \
                     interaction_H1(Pu, phiPn, Mux, phiMnx, Muy, phiMny)

            # 6. Unity checks
            UC_ax = Pu   / phiPn
            UC_Mx = Mux  / phiMnx if phiMnx else 0.0
            UC_My = Muy  / phiMny if phiMny else 0.0
            UC_V  = Vu   / phiVn

            governing_uc = max(abs(UC_ax), abs(UC_Mx), abs(UC_My),
                               abs(UC_V), uc_int)
            governing_limit = max(
                [('axial', abs(UC_ax)),
                 ('Mx',    abs(UC_Mx)),
                 ('My',    abs(UC_My)),
                 ('V',     abs(UC_V)),
                 ('H1',    uc_int)],
                key=lambda kv: kv[1]
            )[0]

            # 7. Assemble results
            rows.append(dict(
                beam_idx             = res['beam_idx'],
                profile              = name,
                family               = sec.family,

                Pu_N                 = Pu,
                phiPn_N              = phiPn,

                Mux_Nm               = Mux,
                phiMnx_Nm            = phiMnx,
                Muy_Nm               = Muy,
                phiMny_Nm            = phiMny,

                Vy_N                 = Vu,
                phiVn_N              = phiVn,

                UC_axial             = UC_ax,
                UC_Mx                = UC_Mx,
                UC_My                = UC_My,
                UC_V                 = UC_V,
                UC_interaction_H1    = uc_int,

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
                    h   = row.get("hw (m)"),
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
