import os
import numpy as np
import copy
import matplotlib.pyplot as plt
import aerosandbox as asb


def _run(airplane, atmosphere, velocity, alpha, beta):
    """One vectorized AeroBuildup call. alpha/beta in degrees, any broadcastable shapes."""
    alpha, beta = np.broadcast_arrays(np.asarray(alpha, dtype=float),
                                      np.asarray(beta, dtype=float))
    op = asb.OperatingPoint(
        atmosphere=atmosphere,
        velocity=velocity,
        alpha=alpha.copy(),
        beta=beta.copy(),
    )
    return asb.AeroBuildup(airplane=airplane, op_point=op).run()


def _prepare_airplane(airplane, xyz_cg=None):
    #I don't even know why we copy aircraft first:)
    airplane = copy.deepcopy(airplane)
    if xyz_cg is not None:
        airplane.xyz_ref = list(xyz_cg)

    #Checking wing span in wing_geo and span specified in aircraft creating (buidinguav.py)
    span = float(airplane.wings[0].span())
    if abs(airplane.b_ref - span) / span > 0.05:
        print(f"WARNING: b_ref = {airplane.b_ref:.3f} m but the main wing span is {span:.3f} m. "
              f"Cl and Cn are normalised by b_ref, so they will be wrong. "
              f"Set b_ref=wing_geo['wingspan'] in build_airplane_model().")
    return airplane


def _flight_condition(airplane, Re, altitude, velocity):
    """ISA atmosphere and speed. Re (based on c_ref) sets V unless velocity is given."""
    atmosphere = asb.Atmosphere(altitude=altitude)
    rho = float(atmosphere.density())
    mu = float(atmosphere.dynamic_viscosity())
    c_ref = float(airplane.c_ref)
    if velocity is None:
        velocity = Re * mu / (rho * c_ref)
    else:
        Re = rho * velocity * c_ref / mu
    q = 0.5 * rho * velocity ** 2
    return atmosphere, rho, velocity, Re, q


def _zero_crossing(x, y, want_negative_slope=True):
    """x where y crosses zero (linear interpolation). NaN if there is no crossing."""
    idx = np.where(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    if want_negative_slope:
        stable = [i for i in idx if y[i + 1] < y[i]]
        idx = np.array(stable) if len(stable) else idx
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return float(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]))


def _slope_per_rad(x_deg, y, mask):
    """Linear-fit slope of y vs x (x in degrees) over mask, returned per radian."""
    return float(np.polyfit(x_deg[mask], y[mask], 1)[0] * 180.0 / np.pi)


def run_aerobuildup_analysis(
    airplane,
    Re=450_000,
    altitude=1000.0,
    velocity=None,
    alpha_range=(-10, 20, 61),
    beta_range=(-15, 15, 31),
    alphas_for_beta=(0, 4, 8),
    mass_kg=None,
    xyz_cg=None,
):
    """
    AeroBuildup sweeps of a whole aircraft: alpha sweep (beta = 0), beta sweeps
    at several alphas, and the stability derivatives derived from them.

    Parameters
    ----------
    airplane : asb.Airplane          from build_airplane_model()
    Re : float                       Reynolds number based on airplane.c_ref (MAC).
                                     Sets the flight speed at the given altitude.
    altitude : float                 m, ISA. (rho = 1.1117 kg/m3 in inputs.py is ~1000 m.)
    velocity : float or None         m/s. If given, overrides Re and Re is recomputed.
    alpha_range, beta_range : (start, stop, n_points) in degrees
    alphas_for_beta : alphas (deg) at which the beta sweeps are run
    mass_kg : float or None          if given, adds the weight line and required CL
    xyz_cg : [x, y, z] or None       moment reference point. None keeps airplane.xyz_ref.
                                     All moments (Cm, Cl, Cn) are about this point.
    """
    airplane = _prepare_airplane(airplane, xyz_cg)
    atmosphere, rho, velocity, Re, q = _flight_condition(airplane, Re, altitude, velocity)
    c_ref = float(airplane.c_ref)
    s_ref = float(airplane.s_ref)

    # --- alpha sweep, beta = 0 ---
    alpha = np.linspace(*alpha_range)
    r = _run(airplane, atmosphere, velocity, alpha, np.zeros_like(alpha))
    CL, CD, Cm = (np.asarray(r[k], dtype=float) for k in ("CL", "CD", "Cm"))
    LD = CL / CD
    lift_N = CL * q * s_ref

    alpha_trim = _zero_crossing(alpha, Cm)
    CL_trim = float(np.interp(alpha_trim, alpha, CL)) if np.isfinite(alpha_trim) else np.nan
    CD_trim = float(np.interp(alpha_trim, alpha, CD)) if np.isfinite(alpha_trim) else np.nan

    # Longitudinal derivatives from a linear fit +-3 deg around trim (or around 0 if no trim)
    a0 = alpha_trim if np.isfinite(alpha_trim) else 0.0
    mask = np.abs(alpha - a0) <= 3.0
    CLa = _slope_per_rad(alpha, CL, mask)
    Cma = _slope_per_rad(alpha, Cm, mask)
    static_margin = -Cma / CLa                      # fraction of c_ref, > 0 is stable
    x_np = airplane.xyz_ref[0] + static_margin * c_ref   # x is positive aft

    i_best = int(np.nanargmax(LD))

    # --- beta sweeps ---
    beta = np.linspace(*beta_range)
    beta_data = {}
    small = np.abs(beta) <= 5.0
    for a in alphas_for_beta:
        rb = _run(airplane, atmosphere, velocity, np.full_like(beta, a), beta)
        CY, Cl, Cn = (np.asarray(rb[k], dtype=float) for k in ("CY", "Cl", "Cn"))
        beta_data[a] = {
            "CY": CY, "Cl": Cl, "Cn": Cn,
            "CL": np.asarray(rb["CL"], dtype=float), "CD": np.asarray(rb["CD"], dtype=float),
            "CYb": _slope_per_rad(beta, CY, small),
            "Clb": _slope_per_rad(beta, Cl, small),
            "Cnb": _slope_per_rad(beta, Cn, small),
        }

    result = {
        "Re": Re, "velocity": velocity, "rho": rho, "q": q, "c_ref": c_ref, "s_ref": s_ref,
        "b_ref": float(airplane.b_ref), "xyz_ref": list(airplane.xyz_ref),
        "alpha": alpha, "CL": CL, "CD": CD, "Cm": Cm, "LD": LD, "lift_N": lift_N,
        "alpha_trim": alpha_trim, "CL_trim": CL_trim, "CD_trim": CD_trim,
        "LD_trim": CL_trim / CD_trim if np.isfinite(CL_trim) else np.nan,
        "CLa": CLa, "Cma": Cma, "static_margin": static_margin, "x_np": float(x_np),
        "LD_max": float(LD[i_best]), "alpha_LDmax": float(alpha[i_best]), "CL_LDmax": float(CL[i_best]),
        "CLmax": float(np.nanmax(CL)), "alpha_CLmax": float(alpha[int(np.nanargmax(CL))]),
        "CD_min": float(np.nanmin(CD)),
        "beta": beta, "beta_data": beta_data,
        "weight_N": mass_kg * 9.81 if mass_kg is not None else None,
        "CL_required": (mass_kg * 9.81) / (q * s_ref) if mass_kg is not None else None,
    }
    return result


def print_aero_summary(res):
    print(f"\n{'=' * 60}")
    print("AEROBUILDUP SUMMARY (whole aircraft)")
    print(f"{'=' * 60}")
    print(f"Re = {res['Re']:.0f} (c_ref = {res['c_ref']:.3f} m)  ->  V = {res['velocity']:.1f} m/s "
          f"({res['velocity'] * 3.6:.0f} km/h), q = {res['q']:.0f} Pa")
    print(f"Moment reference x = {res['xyz_ref'][0]:.3f} m,  s_ref = {res['s_ref']:.3f} m2,  b_ref = {res['b_ref']:.3f} m")
    print("\nLongitudinal")
    print(f"  CL_alpha        = {res['CLa']:.3f} /rad")
    print(f"  Cm_alpha        = {res['Cma']:.3f} /rad   ({'stable' if res['Cma'] < 0 else 'UNSTABLE'})")
    print(f"  Static margin   = {res['static_margin'] * 100:.1f} % MAC  (neutral point x = {res['x_np']:.3f} m)")
    print(f"  Trim (no elevon): alpha = {res['alpha_trim']:.2f} deg, CL = {res['CL_trim']:.3f}, "
          f"CD = {res['CD_trim']:.4f}, L/D = {res['LD_trim']:.1f}")
    print(f"  Max L/D         = {res['LD_max']:.1f} at alpha = {res['alpha_LDmax']:.1f} deg (CL = {res['CL_LDmax']:.2f})")
    print(f"  CL_max          = {res['CLmax']:.2f} at alpha = {res['alpha_CLmax']:.1f} deg,  CD_min = {res['CD_min']:.4f}")
    if res["CL_required"] is not None:
        print(f"  CL required for level flight at this speed = {res['CL_required']:.3f}")
    print("\nLateral-directional (per rad)")
    print(f"  {'alpha':>6} {'CY_beta':>9} {'Cl_beta':>9} {'Cn_beta':>9}")
    for a, d in res["beta_data"].items():
        print(f"  {a:6.1f} {d['CYb']:9.3f} {d['Clb']:9.3f} {d['Cnb']:9.3f}")
    print("  Expected for a stable aircraft: CY_beta < 0, Cl_beta < 0, Cn_beta > 0 "
          "(check the sign convention if Cn_beta comes out negative)")
    print(f"{'=' * 60}\n")


def plot_aero_results(res, save_dir=None):
    a = res["alpha"]
    at = res["alpha_trim"]

    # ---- Figure 1: longitudinal ----
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"AeroBuildup, longitudinal (beta = 0, Re = {res['Re'] / 1e3:.0f}k, V = {res['velocity']:.1f} m/s)",
                 fontsize=14, fontweight="bold")

    ax[0, 0].plot(a, res["CL"], lw=2)
    ax[0, 0].set(xlabel="alpha (deg)", ylabel="$C_L$", title="Lift coefficient")
    if res["CL_required"] is not None:
        ax[0, 0].axhline(res["CL_required"], color="k", ls="--", lw=1, label="CL required (level flight)")
        ax[0, 0].legend()

    ax[0, 1].plot(a, res["Cm"], lw=2, color="tab:red")
    ax[0, 1].axhline(0, color="k", lw=0.8)
    if np.isfinite(at):
        ax[0, 1].plot(at, 0, "ko", label=f"trim alpha = {at:.2f} deg")
        ax[0, 1].legend()
    ax[0, 1].set(xlabel="alpha (deg)", ylabel="$C_m$",
                 title=f"Pitching moment (SM = {res['static_margin'] * 100:.1f}% MAC)")

    ax[0, 2].plot(res["CL"], res["Cm"], lw=2, color="tab:red")
    ax[0, 2].axhline(0, color="k", lw=0.8)
    ax[0, 2].set(xlabel="$C_L$", ylabel="$C_m$", title="$C_m$ vs $C_L$ (slope = -static margin)")

    ax[1, 0].plot(res["CD"], res["CL"], lw=2, color="tab:green")
    ax[1, 0].plot(res["CD"][np.nanargmax(res["LD"])], res["CL_LDmax"], "k*", ms=12,
                  label=f"max L/D = {res['LD_max']:.1f}")
    if np.isfinite(res["CL_trim"]):
        ax[1, 0].plot(res["CD_trim"], res["CL_trim"], "ro", label=f"trim, L/D = {res['LD_trim']:.1f}")
    ax[1, 0].set(xlabel="$C_D$", ylabel="$C_L$", title="Drag polar")
    ax[1, 0].legend()

    ax[1, 1].plot(a, res["LD"], lw=2, color="tab:purple")
    ax[1, 1].set(xlabel="alpha (deg)", ylabel="L/D", title="Lift-to-drag ratio")

    ax[1, 2].plot(a, res["lift_N"], lw=2, color="tab:orange")
    if res["weight_N"] is not None:
        ax[1, 2].axhline(res["weight_N"], color="k", ls="--", lw=1, label=f"weight = {res['weight_N']:.1f} N")
        ax[1, 2].legend()
    ax[1, 2].set(xlabel="alpha (deg)", ylabel="Lift (N)", title="Lift force")

    for x in ax.flat:
        x.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ---- Figure 2: lateral-directional ----
    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 5))
    fig2.suptitle(f"AeroBuildup, sideslip sweeps (Re = {res['Re'] / 1e3:.0f}k)", fontsize=14, fontweight="bold")
    for alpha_b, d in res["beta_data"].items():
        lab = f"alpha = {alpha_b:g} deg"
        ax2[0].plot(res["beta"], d["CY"], lw=2, label=lab)
        ax2[1].plot(res["beta"], d["Cl"], lw=2, label=lab)
        ax2[2].plot(res["beta"], d["Cn"], lw=2, label=lab)
    ax2[0].set(xlabel="beta (deg)", ylabel="$C_Y$", title="Side force")
    ax2[1].set(xlabel="beta (deg)", ylabel="$C_l$", title="Rolling moment")
    ax2[2].set(xlabel="beta (deg)", ylabel="$C_n$", title="Yawing moment (slope = $C_{n\\beta}$, should be > 0)")
    for x in ax2:
        x.axhline(0, color="k", lw=0.8)
        x.grid(True, alpha=0.3)
        x.legend()
    fig2.tight_layout(rect=[0, 0, 1, 0.94])

    # ---- Figure 3: derivatives vs alpha ----
    alphas_b = list(res["beta_data"].keys())
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 5))
    ax3.plot(alphas_b, [res["beta_data"][k]["Cnb"] for k in alphas_b], "o-", lw=2, label="$C_{n\\beta}$")
    ax3.plot(alphas_b, [res["beta_data"][k]["Clb"] for k in alphas_b], "s-", lw=2, label="$C_{l\\beta}$")
    ax3.plot(alphas_b, [res["beta_data"][k]["CYb"] for k in alphas_b], "^-", lw=2, label="$C_{Y\\beta}$")
    ax3.axhline(0, color="k", lw=0.8)
    ax3.set(xlabel="alpha (deg)", ylabel="derivative (per rad)", title="Lateral-directional derivatives vs alpha")
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    fig3.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, "aero_longitudinal.png"), dpi=150)
        fig2.savefig(os.path.join(save_dir, "aero_sideslip.png"), dpi=150)
        fig3.savefig(os.path.join(save_dir, "aero_lateral_derivatives.png"), dpi=150)

    plt.show()


def run_alpha_beta_map(
    airplane,
    Re=450_000,
    altitude=1000.0,
    velocity=None,
    alpha_range=(-90, 90, 91),
    beta_range=(-90, 90, 91),
    xyz_cg=None,
):
    """
    AeroBuildup on a full alpha x beta grid (degrees). Defaults give a 2 deg step;
    use 150 x 150 points as in the tutorial if you want smoother contours.
    Returns 2D arrays shaped like the grid.
    """
    airplane = _prepare_airplane(airplane, xyz_cg)
    atmosphere, rho, velocity, Re, q = _flight_condition(airplane, Re, altitude, velocity)

    Beta, Alpha = np.meshgrid(np.linspace(*beta_range), np.linspace(*alpha_range))
    r = _run(airplane, atmosphere, velocity, Alpha.flatten(), Beta.flatten())

    m = {k: np.asarray(r[k], dtype=float).reshape(Alpha.shape)
         for k in ("CL", "CD", "CY", "Cl", "Cm", "Cn")}
    m["LD"] = m["CL"] / m["CD"]
    m.update(Alpha=Alpha, Beta=Beta, Re=Re, velocity=velocity)
    return m


def plot_alpha_beta_maps(m, coefficients=("CL", "CD", "LD", "Cm", "Cn", "Cl", "CY"), save_dir=None):
    """One contour map per coefficient on the beta (x) - alpha (y) plane."""
    import aerosandbox.tools.pretty_plots as p

    specs = {
        "CL": dict(label="Lift Coefficient $C_L$ [-]", cmap="RdBu", sym=True, fmt=".2f"),
        "CD": dict(label="Drag Coefficient $C_D$ [-]", cmap="YlOrRd", log=True, fmt=".2f"),
        "LD": dict(label="Finesse $C_L / C_D$ [-]", cmap="RdBu", sym=True, fmt=".0f", levels=15),
        "Cm": dict(label="Pitching Moment $C_m$ [-]", cmap="RdBu", sym=True, fmt=".2f"),
        "Cn": dict(label="Yawing Moment $C_n$ [-]", cmap="RdBu", sym=True, fmt=".3f"),
        "Cl": dict(label="Rolling Moment $C_l$ [-]", cmap="RdBu", sym=True, fmt=".3f"),
        "CY": dict(label="Side Force $C_Y$ [-]", cmap="RdBu", sym=True, fmt=".2f"),
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for name in coefficients:
        s = specs[name]
        Z = m[name]
        kwargs = dict(
            colorbar_label=s["label"],
            linelabels_format=lambda x, f=s["fmt"]: format(x, f),
            linelabels_fontsize=7,
            cmap=s["cmap"],
            alpha=0.6,
        )
        if "levels" in s:
            kwargs["levels"] = s["levels"]
        if s.get("log"):
            kwargs["z_log_scale"] = True

        fig, ax = plt.subplots(figsize=(7, 6))
        p.contour(m["Beta"], m["Alpha"], Z, **kwargs)
        if s.get("sym"):
            finite = Z[np.isfinite(Z)]
            plt.clim(*np.array([-1, 1]) * np.max(np.abs(finite)))

        p.set_ticks(15, 5, 15, 5)
        p.equal()
        p.show_plot(
            f"AeroBuildup, Re = {m['Re'] / 1e3:.0f}k, V = {m['velocity']:.1f} m/s",
            r"Sideslip angle $\beta$ [deg]",
            r"Angle of Attack $\alpha$ [deg]",
            set_ticks=False,
            savefig=os.path.join(save_dir, f"map_{name}.png") if save_dir else None,
            show=False,
        )

    plt.show()   # one show for all maps, same as plot_aero_results