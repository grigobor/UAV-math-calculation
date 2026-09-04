import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import neuralfoil as nf
import aerosandbox as asb
import aerosandbox.numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import aerosandbox.tools.pretty_plots as p

import inputs

def create_airfoil():
    """
    Runs both independent optimizations, builds and validates the morph
    path between them, and returns results in a shape main.py can use.

    Returns
    -------
    tuple:
        cruise_airfoil, hover_airfoil,
        cruise_alpha, cruise_CL, cruise_CD, cruise_CM,
        hover_alpha, hover_CL, hover_CD, hover_CM,
        morph_fn  -- call morph_fn(t) to get the airfoil at any morph
                     state 0..1 (this is what buildinguav.py should call
                     for the wing cross-sections at a given flight phase,
                     instead of hardcoding the two endpoints).
    """
    cruise_airfoil, cruise_alpha, cruise_CL, cruise_CD, cruise_CM, cruise_airfoil_export = optimize_cruise_airfoil()
    hover_airfoil, hover_alpha, hover_CL, hover_CD, hover_CM, hover_airfoil_export = optimize_hover_airfoil()

    plot_comparison(cruise_airfoil, hover_airfoil)
    airfoil_characteristics(cruise_airfoil, hover_airfoil)

    validate_morph_path(hover_airfoil, cruise_airfoil, n=11)
    plot_morph_sequence(hover_airfoil, cruise_airfoil, n=7)

    def morph_fn(t):
        return morph_airfoil(hover_airfoil, cruise_airfoil, t)

    return (
        cruise_airfoil, hover_airfoil,
        cruise_alpha, cruise_CL, cruise_CD, cruise_CM,
        hover_alpha, hover_CL, hover_CD, hover_CM, 
        cruise_airfoil_export, hover_airfoil_export,
        morph_fn
    )

# ---------------------------------------------------------------------------
# HORIZONTAL / CRUISE AIRFOIL -- asymmetric, free trailing edge
# ---------------------------------------------------------------------------
def optimize_cruise_airfoil():
    """
    Optimize a cambered airfoil for cruise. Only the leading edge is
    constrained (upper = -lower at the first two CST stations, to keep a
    round, well-posed nose). The trailing-edge CST weight is left free on
    both surfaces, so the aft camber -- where camber has the most leverage
    on CL and CM -- can actually develop, instead of being forced back to
    a symmetric shape.
    """
    opti = asb.Opti()

    CL_multipoint_targets = np.array([0.5, 0.6, 0.7, 0.9, 1.1])
    CL_multipoint_weights = np.array([4, 5, 6, 7, 10])

    initial_airfoil = asb.KulfanAirfoil("naca0012")

    upper_weights = opti.variable(
        init_guess=initial_airfoil.upper_weights,
        #lower_bound=-0.1,
        #upper_bound=0.5
        lower_bound=-0.5,
        upper_bound=0.25
    )
    lower_weights = opti.variable(
        init_guess=initial_airfoil.lower_weights,
        #lower_bound=-0.3,
        #upper_bound=0.3
        lower_bound=-0.25,
        upper_bound=0.5
    )

    # Leading edge only -- keeps the nose round and well-posed.
    opti.subject_to(upper_weights[0] == -lower_weights[0])
    # (No trailing-edge equality constraint here -- this is the whole point:
    # the aft surface is now free to be asymmetric.)

    cruise_airfoil = asb.KulfanAirfoil(
        name="Asymmetric Cruise Airfoil",
        upper_weights=upper_weights,
        lower_weights=lower_weights,
        leading_edge_weight=opti.variable(init_guess=0, lower_bound=-0.05, upper_bound=0.05),
        TE_thickness=0,
    )

    alpha = opti.variable(
        # The formula is based on the fundamental relationship for the potential flow around a thin airfoil.
        # CL = 2*pi*alpha_(rad)
        init_guess=np.degrees(CL_multipoint_targets / (2 * np.pi)), 
        lower_bound=-5,
        upper_bound=15,
    )

    aero = cruise_airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=inputs.DesignConstants.Re_cruise,
        mach=inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound,
        model_size="large"
    )

    opti.subject_to(
        [
            aero["CL"] >= CL_multipoint_targets,
            aero["CM"] >= -0.02,
            cruise_airfoil.local_thickness(x_over_c=0.005) >= 0.01,
            cruise_airfoil.local_thickness(x_over_c=0.33) >= 0.05,
            cruise_airfoil.local_thickness(x_over_c=0.60) >= 0.035,
            cruise_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
            cruise_airfoil.local_thickness() <= 0.12,
            cruise_airfoil.local_thickness() > 0,
        ]
    )

    get_wiggliness = lambda af: sum(
        [
            np.sum(np.diff(np.diff(array)) ** 2)
            for array in [af.lower_weights, af.upper_weights]
        ]
    )
    opti.subject_to(
        get_wiggliness(cruise_airfoil) < 2 * get_wiggliness(initial_airfoil),
    )

    # Small alpha penalty: CL targets are still hard constraints, so this
    # doesn't stop the airfoil from meeting them -- it just removes the
    # solver's incentive to satisfy CL cheaply via angle-of-attack instead
    # of camber once camber is no longer penalized by the reflex
    # requirement above. Weight is small on purpose: large enough to break
    # the "just pitch up" tie, small enough not to fight the CD objective.
    alpha_penalty_weight = 0.002
    opti.minimize(
        np.mean(aero["CD"] * CL_multipoint_weights)
        + alpha_penalty_weight * np.mean(alpha ** 2)
    )

    sol = opti.solve()

    cruise_airfoil_to_export = asb.KulfanAirfoil(
        name="Asymmetric Cruise Airfoil",
        upper_weights=sol.value(upper_weights),
        lower_weights=sol.value(lower_weights),
        leading_edge_weight=sol.value(cruise_airfoil.leading_edge_weight),
        TE_thickness=0,
    )

    return (
        sol.value(cruise_airfoil),
        sol.value(alpha),
        sol.value(aero["CL"]),
        sol.value(aero["CD"]),
        sol.value(aero["CM"]),
        cruise_airfoil_to_export
    )


# ---------------------------------------------------------------------------
# VERTICAL / HOVER AIRFOIL -- fully symmetric, independently optimized
# ---------------------------------------------------------------------------
def optimize_hover_airfoil():

    opti = asb.Opti()

    initial_airfoil = asb.KulfanAirfoil("naca0012")

    weights = opti.variable(
        init_guess=initial_airfoil.upper_weights,
        lower_bound=-0.5,
        upper_bound=0.5,
    )

    hover_airfoil = asb.KulfanAirfoil(
        name="Symmetric Hover Airfoil",
        upper_weights=weights,
        lower_weights=-weights,
        leading_edge_weight=0,
        TE_thickness=0,
    )

    alphas_hover = np.array([0, 3, 6, 10, 15])

    aero = hover_airfoil.get_aero_from_neuralfoil(
        alpha=alphas_hover,
        Re=inputs.DesignConstants.Re_cruise,
        mach=inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound,
        model_size="large"
    )

    opti.subject_to(
        [
            hover_airfoil.local_thickness(x_over_c=0.005) >= 0.01,
            hover_airfoil.local_thickness(x_over_c=0.33) >= 0.05,
            hover_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
            hover_airfoil.local_thickness() <= 0.12,
            hover_airfoil.local_thickness() > 0,
        ]
    )

    get_wiggliness = lambda af: np.sum(np.diff(np.diff(af.upper_weights)) ** 2)
    opti.subject_to(
        get_wiggliness(hover_airfoil) < 2 * get_wiggliness(initial_airfoil),
    )

    opti.minimize(np.mean(aero["CD"]))

    sol = opti.solve()

    hover_airfoil_to_export = asb.KulfanAirfoil(
        name="Symmetric Hover Airfoil",
        upper_weights=sol.value(hover_airfoil.upper_weights),
        lower_weights=sol.value(hover_airfoil.lower_weights),
        leading_edge_weight=sol.value(hover_airfoil.leading_edge_weight),
        TE_thickness=sol.value(hover_airfoil.TE_thickness),
    )

    return (
        sol.value(hover_airfoil),
        alphas_hover,
        sol.value(aero["CL"]),
        sol.value(aero["CD"]),
        sol.value(aero["CM"]),
        hover_airfoil_to_export
    )



# ---------------------------------------------------------------------------
# THE ADAPTIVE AIRFOIL: one physical shape, morphing between the two optima
# ---------------------------------------------------------------------------
def morph_airfoil(hover_af, cruise_af, t, name=None):
    """
    Return the airfoil at morph state `t`, by linearly interpolating the
    Kulfan (CST) weights between the hover shape (t=0) and the cruise
    shape (t=1).

    This works because both airfoils are expressed on the *same* CST
    basis (same number of weights, same Bernstein order) -- they were
    both initialized from the same naca0012 KulfanAirfoil, so the weight
    vectors line up one-to-one. Interpolating coefficients on a shared
    polynomial basis is what actually gives you a continuous, physically
    realizable family of shapes -- this is the standard way morphing/CST
    airfoils are represented in the literature (e.g. compliant/morphing
    skin structures whose surface follows a blended CST shape as an
    actuator moves it from one designed state to another).

    Parameters
    ----------
    hover_af, cruise_af : asb.KulfanAirfoil
        The two independently-optimized endpoint shapes.
    t : float or array-like, 0 to 1
        Morph state. 0 = hover (symmetric), 1 = cruise (asymmetric).
        Values outside [0, 1] are allowed if you want to check
        overshoot/extrapolation, but aren't physically meaningful.

    Returns
    -------
    asb.KulfanAirfoil
    """
    assert len(hover_af.upper_weights) == len(cruise_af.upper_weights), (
        "Hover and cruise airfoils must share the same CST order to morph "
        "between them -- make sure both are still initialized from the "
        "same base KulfanAirfoil."
    )

    upper = (1 - t) * hover_af.upper_weights + t * cruise_af.upper_weights
    lower = (1 - t) * hover_af.lower_weights + t * cruise_af.lower_weights
    le_weight = (1 - t) * hover_af.leading_edge_weight + t * cruise_af.leading_edge_weight
    te_thickness = (1 - t) * hover_af.TE_thickness + t * cruise_af.TE_thickness

    return asb.KulfanAirfoil(
        name=name or f"Adaptive Airfoil (t={t:.2f})",
        upper_weights=upper,
        lower_weights=lower,
        leading_edge_weight=le_weight,
        TE_thickness=te_thickness,
    )


def sample_morph_sequence(hover_af, cruise_af, n=11):
    """Convenience: list of KulfanAirfoils evenly spaced from t=0 to t=1."""
    ts = np.linspace(0, 1, n)
    return [morph_airfoil(hover_af, cruise_af, t) for t in ts], ts


def validate_morph_path(hover_af, cruise_af, n=11,
                         Re=None, mach=None, alpha_check=0.0,
                         min_thickness=0.005):
    """
    Sanity-check the morph path before you commit to building it: since
    the two endpoints were optimized *independently*, nothing guarantees
    the shapes in between stay well-behaved (thin sections pinching to
    zero thickness, self-intersecting surfaces, or a CD/CL that spikes
    partway through the morph instead of varying smoothly). This sweeps
    t and reports thickness and aero at each state so you can catch a bad
    intermediate shape before it's a physical part.

    Returns a pandas DataFrame with one row per t.
    """
    if Re is None:
        Re = inputs.DesignConstants.Re_cruise
    if mach is None:
        mach = inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound

    airfoils, ts = sample_morph_sequence(hover_af, cruise_af, n=n)
    rows = []
    for t, af in zip(ts, airfoils):
        min_t = af.local_thickness()  # array of thickness values along chord
        aero = af.get_aero_from_neuralfoil(alpha=alpha_check, Re=Re, mach=mach, model_size="large")
        rows.append({
            "t": round(float(t), 3),
            "min_local_thickness": float(np.min(min_t)),
            "thickness_ok": bool(np.min(min_t) >= min_thickness),
            "CL": float(aero["CL"]),
            "CD": float(aero["CD"]),
            "CM": float(aero["CM"]),
        })

    df = pd.DataFrame(rows)
    if not df["thickness_ok"].all():
        print("⚠ Morph path has intermediate shapes below the minimum thickness "
              "-- inspect these t values before building the structure:")
        print(df[~df["thickness_ok"]])
    else:
        print(f"✓ Morph path OK: all {n} intermediate shapes meet the "
              f"{min_thickness:.3f} min thickness requirement.")
    return df


def export_morph_sequence(hover_af, cruise_af, n=11, n_points_per_side=100,
                           out_dir="."):
    """
    Write a .dat coordinate file (Selig format, same style as
    my_custom_airfoil.dat) for each morph state -- one per manufactured
    /interpolated cross-section, for CAD or CFD use.
    """
    import os
    airfoils, ts = sample_morph_sequence(hover_af, cruise_af, n=n)
    paths = []
    for t, af in zip(ts, airfoils):
        discretized = af.to_airfoil(n_points_per_side=n_points_per_side)
        fname = os.path.join(out_dir, f"morph_t{t:.2f}.dat")
        with open(fname, "w") as f:
            f.write(f"Adaptive Airfoil t={t:.2f}\n")
            for x, y in discretized.coordinates:
                f.write(f"{x:.6f} {y:.6f}\n")
        paths.append(fname)
    print(f"✓ Wrote {len(paths)} morph-state .dat files to {out_dir}")
    return paths


def plot_morph_sequence(hover_af, cruise_af, n=7):
    """Overlay the morph sequence to visually confirm a smooth, non-self-
    intersecting transformation from hover to cruise shape."""
    airfoils, ts = sample_morph_sequence(hover_af, cruise_af, n=n)

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("coolwarm")
    for t, af in zip(ts, airfoils):
        color = cmap(t)
        ax.plot(af.x(), af.y(), color=color, linewidth=2,
                label=f"t={t:.2f}" if t in (0.0, 1.0) else None)

    ax.set_title("Adaptive Airfoil: Morph Path from Hover (t=0) to Cruise (t=1)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("$x/c$")
    ax.set_ylabel("$y/c$")
    ax.axis("equal")
    ax.legend(fontsize=10)
    plt.tight_layout()
    p.show_plot(title=None, legend=False)


def plot_comparison(af_asym, af_sym):
    airfoils_and_colors = {
        "Asymmetric (Horizontal/Cruise) Airfoil": (af_asym, 'blue'),
        "Symmetric (Vertical/Hover) Airfoil": (af_sym, 'red'),
        "NACA0012 (Baseline)": (asb.KulfanAirfoil("naca0012"), 'green'),
        "MH-60 (Reference)": (asb.KulfanAirfoil("mh60"), 'orange')
    }

    Re_plot = inputs.DesignConstants.Re_cruise
    mach_plot = inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    for i, (name, (af, color)) in enumerate(airfoils_and_colors.items()):
        ax[0].fill(
            af.x(),
            af.y(),
            facecolor=(*p.adjust_lightness(color, 1.5), 0.1),
            edgecolor=color,
            linewidth=2,
            label=name,
            linestyle='-' if "Cruise" in name else '--',
            zorder=4 if "Cruise" in name else 3,
        )
        alphas = np.linspace(0, 12, 50)
        aero = af.get_aero_from_neuralfoil(
            alpha=alphas,
            Re=Re_plot,
            mach=mach_plot,
        )
        ax[1].plot(
            aero["CD"],
            aero["CL"],
            color=color,
            label=name,
            alpha=0.8
        )

    ax[0].legend(fontsize=11, loc="lower center", ncol=len(airfoils_and_colors) // 2)
    ax[0].set_title("I. Airfoil Profile Geometry", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("$x/c$")
    ax[0].set_ylabel("$y/c$")
    ax[0].axis("equal")

    ax[1].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[1].set_title(f"II. Aerodynamic Polar ($Re={Re_plot/1e3:.0f}k$)", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Drag Coefficient $C_D$")
    ax[1].set_ylabel("Lift Coefficient $C_L$")
    ax[1].set_xlim(0, 0.035)
    ax[1].set_ylim(-0.2, 1.6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p.show_plot(title=None, legend=False)


def airfoil_characteristics(af_asym, af_sym):
    airfoils_and_colors = {
        "Asymmetric (Horizontal/Cruise) Airfoil": (af_asym, 'blue'),
        "Symmetric (Vertical/Hover) Airfoil": (af_sym, 'red')
    }

    Re_plot = inputs.DesignConstants.Re_cruise
    mach_plot = inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound

    fig, ax = plt.subplots(3, 1, figsize=(8, 12))

    for i, (name, (af, color)) in enumerate(airfoils_and_colors.items()):
        alphas = np.linspace(-20, 20, 100)
        aero = af.get_aero_from_neuralfoil(
            alpha=alphas,
            Re=Re_plot,
            mach=mach_plot,
        )
        ax[0].plot(alphas, aero["CL"], color=color, label=name, alpha=0.8)
        ax[1].plot(alphas, aero["CD"], color=color, label=name, alpha=0.8, linewidth=2)
        ax[2].plot(alphas, aero["CM"], color=color, label=name, alpha=0.8)

    ax[0].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[0].set_title(f"I. Lift Coefficient vs. Alpha ($Re={Re_plot/1e3:.0f}k$)", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("Angle of Attack $\\alpha$ (degrees)")
    ax[0].set_ylabel("Lift Coefficient $C_L$")
    ax[0].grid(True, alpha=0.3)
    ax[0].set_ylim(-1.8, 1.8)

    ax[1].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[1].set_title("II. Drag Coefficient vs. Alpha", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Angle of Attack $\\alpha$ (degrees)")
    ax[1].set_ylabel("Drag Coefficient $C_D$")
    ax[1].grid(True, alpha=0.3)
    ax[1].set_ylim(0, 0.15)

    ax[2].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[2].set_title("III. Pitching Moment vs. Alpha", fontsize=12, fontweight='bold')
    ax[2].set_xlabel("Angle of Attack $\\alpha$ (degrees)")
    ax[2].set_ylabel("Pitching Moment Coefficient $C_m$")
    ax[2].grid(True, alpha=0.3)
    ax[2].set_ylim(-0.25, 0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    p.show_plot(title=None, legend=False)

def extend_trailing_edge(base_chord, extension_fraction):
    """
    Physical chord after a Fowler-style telescoping TE extension.

    The Kulfan/CST weights (normalized x/c shape) don't change -- a
    telescoping TE slides the aft structure aft without reshaping the
    section. Only the physical chord grows. Everything downstream
    (chord, wing area, Re) must be scaled by the same chord_multiplier
    to stay geometrically consistent between the two states.
    """
    chord_multiplier = 1.0 + extension_fraction
    return {
        "chord_multiplier": chord_multiplier,
        "extended_chord": base_chord * chord_multiplier,
    }


def find_extension_for_target(base_chord, target_chord):
    """Inverse of extend_trailing_edge: solve extension_fraction/chord_multiplier
    needed to reach a target chord (e.g. from a required cruise CL or a
    structural constraint)."""
    chord_multiplier = target_chord / base_chord
    return {
        "chord_multiplier": chord_multiplier,
        "extension_fraction": chord_multiplier - 1.0,
    }


def compare_hover_cruise_geometry(wing_geo, const, chord_multiplier):
    """
    Side-by-side table of hover (t=0, retracted) vs cruise (t=1, extended)
    wing geometry, propagating chord_multiplier into chord, area, AR, and
    Re together -- this is the piece that was missing: extending chord
    alone without also updating S_wing/AR/Re gives you two aero states
    that don't correspond to the same physical wing.

    Span is held fixed -- a telescoping TE extends chord, not span, so
    the area increase comes entirely from chord growth and AR drops
    accordingly.
    """
    def reynolds(chord):
        return (const.air_density_cruise * const.cruise_speed_ms * chord
                / const.dynamic_viscosity_cruise)

    hover_S = wing_geo["S_wing"]
    hover_MAC = wing_geo["MAC"]
    cruise_MAC = hover_MAC * chord_multiplier
    cruise_S = hover_S * chord_multiplier  # span fixed -> area scales with chord

    rows = [
        {
            "state": "hover (t=0, retracted)",
            "MAC_m": hover_MAC,
            "root_chord_m": wing_geo["root_chord"],
            "tip_chord_m": wing_geo["tip_chord"],
            "S_wing_m2": hover_S,
            "AR": wing_geo["wingspan"] ** 2 / hover_S,
            "Re": reynolds(hover_MAC),
        },
        {
            "state": "cruise (t=1, extended)",
            "MAC_m": cruise_MAC,
            "root_chord_m": wing_geo["root_chord"] * chord_multiplier,
            "tip_chord_m": wing_geo["tip_chord"] * chord_multiplier,
            "S_wing_m2": cruise_S,
            "AR": wing_geo["wingspan"] ** 2 / cruise_S,
            "Re": reynolds(cruise_MAC),
        },
    ]
    df = pd.DataFrame(rows).set_index("state")
    pct_change = (df.loc["cruise (t=1, extended)"] / df.loc["hover (t=0, retracted)"] - 1) * 100
    df.loc["Δ (%)"] = pct_change
    return df

def export_endpoint_airfoils(hover_airfoil, cruise_airfoil, 
                               n_points_per_side=100, 
                               out_dir=".", 
                               formats=["dat", "csv"]):
    """
    Export the hover and cruise endpoint airfoils to files compatible with 
    ANSYS Discovery, ANSYS Fluent, and other CFD tools.
    
    Supports multiple formats:
    - 'dat': Selig format (.dat), traditional airfoil format, good for ANSYS
    - 'csv': Comma-separated values, easy to import into spreadsheets/ANSYS
    
    Parameters
    ----------
    hover_airfoil : asb.KulfanAirfoil
        The symmetric hover (VTOL) airfoil.
    cruise_airfoil : asb.KulfanAirfoil
        The asymmetric cruise (fixed-wing) airfoil.
    n_points_per_side : int, optional
        Number of points to discretize on upper and lower surfaces (default 100).
        Higher values (150-200) are recommended for CFD mesh quality.
    out_dir : str, optional
        Output directory path (default current directory).
    formats : list of str, optional
        Export formats: ["dat", "csv"] or subset (default both).
    
    Returns
    -------
    dict
        Dictionary mapping airfoil display names to nested dicts of file paths.
        Example: {'Symmetric Hover Airfoil': {'dat': '...', 'csv': '...'}, ...}
        
    Examples
    --------
    >>> exported = export_endpoint_airfoils(hover_af, cruise_af, 
    ...                                       out_dir="./airfoils",
    ...                                       formats=["dat", "csv"])
    >>> print(exported)
    {'Symmetric Hover Airfoil': 
        {'dat': './airfoils/hover_airfoil.dat', 
         'csv': './airfoils/hover_airfoil.csv'},
     'Asymmetric Cruise Airfoil': 
        {'dat': './airfoils/cruise_airfoil.dat', 
         'csv': './airfoils/cruise_airfoil.csv'}}
    """
    import os
    
    airfoils = {
        "hover": (hover_airfoil, "Symmetric Hover Airfoil"),
        "cruise": (cruise_airfoil, "Asymmetric Cruise Airfoil"),
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    exported_files = {}
    
    for key, (airfoil, display_name) in airfoils.items():
        # Discretize airfoil to x, y coordinates
        discretized = airfoil.to_airfoil(n_coordinates_per_side=n_points_per_side)
        coords = np.array(discretized.coordinates)  # shape: (n_points, 2)
        x = coords[:, 0]
        y = coords[:, 1]
        
        file_paths = {}
        
        # Export as Selig .dat format
        """if "dat" in formats:
            dat_filename = os.path.join(out_dir, f"{key}_airfoil.dat")
            with open(dat_filename, "w") as f:
                # Selig format: first line is title, then x y pairs
                f.write(f"{display_name}\n")
                for xi, yi in zip(x, y):
                    f.write(f"{xi:.6f} {yi:.6f}\n")
            file_paths["dat"] = os.path.abspath(dat_filename)
            print(f"✓ Exported {display_name} (Selig format)")
            print(f"  → {os.path.abspath(dat_filename)}")
        
        # Export as CSV format (ANSYS-friendly)
        if "csv" in formats:
            csv_filename = os.path.join(out_dir, f"{key}_airfoil.csv")
            # Create a DataFrame for easy CSV export
            df_coords = pd.DataFrame({
                "x": x,
                "y": y,
            })
            df_coords.to_csv(csv_filename, index=False, float_format="%.6f")
            file_paths["csv"] = os.path.abspath(csv_filename)
            print(f"✓ Exported {display_name} (CSV format)")
            print(f"  → {os.path.abspath(csv_filename)}")"""

        if "txt" in formats:
            txt_filename = os.path.join(out_dir, f"{key}_airfoil.txt")
            with open(txt_filename, "w") as f:
                #f.write("3d=true\n")
                f.write("polyline=false\n")
                for x, y in zip(x, y):
                    f.write(f"1\t{x:.6f}\t{y:.6f}\n")
            file_paths["txt"] = os.path.abspath(txt_filename)
            print(f"✓ Exported {display_name} (TXT format)")
            print(f"  → {os.path.abspath(txt_filename)}")

        
        exported_files[display_name] = file_paths
    
    print(f"\n{'='*70}")
    print(f"AIRFOIL EXPORT SUMMARY")
    print(f"{'='*70}")
    print(f"Airfoils exported: {len(exported_files)}")
    print(f"Formats: {', '.join(formats)}")
    print(f"Points per side: {n_points_per_side}")
    print(f"Output directory: {os.path.abspath(out_dir)}")
    print(f"{'='*70}\n")
    
    return exported_files
 
 
def export_airfoils_for_ansys(hover_airfoil, cruise_airfoil, out_dir="."):
    """
    Convenience wrapper: export endpoint airfoils in ANSYS-optimized formats.
    
    Exports both .dat (Selig) and .csv formats for maximum compatibility
    with ANSYS Discovery, Fluent, CFX, and other analysis tools.
    
    Parameters
    ----------
    hover_airfoil : asb.KulfanAirfoil
        Symmetric hover (VTOL) airfoil.
    cruise_airfoil : asb.KulfanAirfoil
        Asymmetric cruise (fixed-wing) airfoil.
    out_dir : str, optional
        Output directory (default: current directory).
    
    Returns
    -------
    dict
        Exported file paths for both airfoils in both formats.
    """
    return export_endpoint_airfoils(
        hover_airfoil, 
        cruise_airfoil,
        n_points_per_side=150,  # Higher resolution for CFD mesh quality
        out_dir=out_dir,
        formats=["txt"]
    )