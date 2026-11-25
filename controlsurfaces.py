import numpy as np
import pandas as pd
import inputs


def _get_const(const=None):
    return const if const is not None else inputs.DesignConstants

def flap_effectiveness(cf_c, eta=0.9):
    theta_f = np.arccos(2.0 * cf_c - 1.0)
    tau = 1.0 - (theta_f - np.sin(theta_f)) / np.pi
    return eta * tau

def wing_lift_slope(AR, mach=0.0):
    beta = np.sqrt(max(1.0 - mach ** 2, 1e-6))
    return 2.0 * np.pi * AR / (2.0 + np.sqrt(4.0 + (AR * beta / 1.0) ** 2))

def _chord(y, wing_geo):
    b = wing_geo["wingspan"]
    cr, ct = wing_geo["root_chord"], wing_geo["tip_chord"]
    return cr + (ct - cr) * 2.0 * y / b

def _int_c(y1, y2, wing_geo):
    b = wing_geo["wingspan"]
    cr, ct = wing_geo["root_chord"], wing_geo["tip_chord"]
    return cr * (y2 - y1) + (ct - cr) * (y2 ** 2 - y1 ** 2) / b

def _int_c_y(y1, y2, wing_geo):
    b = wing_geo["wingspan"]
    cr, ct = wing_geo["root_chord"], wing_geo["tip_chord"]
    return cr * (y2 ** 2 - y1 ** 2) / 2.0 + (ct - cr) * (2.0 / b) * (y2 ** 3 - y1 ** 3) / 3.0

def _int_c_y2(y1, y2, wing_geo):
    b = wing_geo["wingspan"]
    cr, ct = wing_geo["root_chord"], wing_geo["tip_chord"]
    return cr * (y2 ** 3 - y1 ** 3) / 3.0 + (ct - cr) * (2.0 / b) * (y2 ** 4 - y1 ** 4) / 4.0

def elevon_derivatives(wing_geo, cf_c, y1, y2, CL_alpha, CD0, l_e, eta=0.9):
    S, b, MAC = wing_geo["S_wing"], wing_geo["wingspan"], wing_geo["MAC"]
    tau = flap_effectiveness(cf_c, eta)

    S_e_one_side = cf_c * _int_c(y1, y2, wing_geo)
    S_e = 2.0 * S_e_one_side

    # Roll power (antisymmetric deflection of both elevons)
    C_ldelta = 2.0 * CL_alpha * tau / (S * b) * _int_c_y(y1, y2, wing_geo)

    # Roll damping (strip theory), reference pb/2V
    C_lp = -4.0 * (CL_alpha + CD0) / (S * b ** 2) * _int_c_y2(0.0, b / 2.0, wing_geo)

    # Pitch power (symmetric deflection)
    C_mdelta = -CL_alpha * tau * (S_e / S) * (l_e / MAC)

    return {
        "tau": tau,
        "S_e": S_e,
        "S_e_over_S": S_e / S,
        "C_ldelta": C_ldelta,
        "C_lp": C_lp,
        "C_mdelta": C_mdelta,
    }


def roll_rate(C_ldelta, C_lp, delta_a, V, b):
    """Steady roll rate, rad/s, for aileron-type deflection delta_a (rad)."""
    return -(C_ldelta / C_lp) * delta_a * (2.0 * V / b)


def hover_control(wing_geo, m0_final, cf_c, y1, y2, prop_y, eta=0.9,
                  slipstream_eff=0.8, l_e=0.1, thrust_ratio=1.0,
                  I_pitch=None, I_axial=None, delta_max=np.radians(25.0),
                  fuselage_length=0.71, const=None):

    const = _get_const(const)
    g = const.g
    rho = const.air_density_sea_level
    W = m0_final * g
    n_eng = const.eng_q
    D = const.prop_D * 0.0254
    A = np.pi * D ** 2 / 4.0

    T = thrust_ratio * W / n_eng
    V_slip = np.sqrt(2.0 * T / (rho * A))
    q_slip = 0.5 * rho * V_slip ** 2 * slipstream_eff

    # Elevon portion inside each prop wash (one prop per semi-span)
    ya, yb = max(y1, prop_y - D / 2.0), min(y2, prop_y + D / 2.0)
    overlap = max(yb - ya, 0.0)
    S_imm_one = cf_c * _int_c(ya, yb, wing_geo) if overlap > 0 else 0.0
    S_imm = 2.0 * S_imm_one

    tau = flap_effectiveness(cf_c, eta)
    CL_a_2d = 2.0 * np.pi  # strip in a high-energy slipstream: 2D slope

    dM_pitch = q_slip * S_imm * CL_a_2d * tau * l_e                 # N*m / rad
    dM_axial = q_slip * S_imm_one * CL_a_2d * tau * prop_y * 2.0    # N*m / rad

    if I_pitch is None:
        I_pitch = m0_final * fuselage_length ** 2 / 12.0
    if I_axial is None:
        I_axial = m0_final * wing_geo["wingspan"] ** 2 / 12.0

    M_pitch = dM_pitch * delta_max
    M_axial = dM_axial * delta_max
    return {
        "V_slip": V_slip,
        "q_slip": q_slip,
        "S_immersed": S_imm,
        "wash_overlap_m": overlap,
        "M_pitch_avail": M_pitch,
        "M_axial_avail": M_axial,
        "alpha_ddot_pitch": M_pitch / I_pitch,   # rad/s^2
        "alpha_ddot_axial": M_axial / I_axial,   # rad/s^2
        "I_pitch": I_pitch,
        "I_axial": I_axial,
    }


# ---------------------------------------------------------------------------
# Elevon sizing driver
# ---------------------------------------------------------------------------
def size_control_surfaces(wing_geo, m0_final, aero_params,
                          delta_max_deg=25.0,
                          roll_rate_req_dps=60.0,
                          cm_trim=-0.02,
                          trim_fraction_max=0.5,
                          hover_accel_req=6.0,       # rad/s^2, pitch and axial
                          hover_margin=1.5,
                          cg_frac_mac=0.20,
                          prop_y=None,
                          y_outer_frac=0.95,
                          cf_c_grid=(0.20, 0.25, 0.30, 0.35),
                          y_inner_frac_grid=(0.50, 0.40, 0.30, 0.20, 0.10),
                          eta=0.9, V_roll=None,
                          I_pitch=None, I_axial=None, const=None):
    const = _get_const(const)
    b = wing_geo["wingspan"]
    MAC = wing_geo["MAC"]
    delta_max = np.radians(delta_max_deg)
    V = V_roll if V_roll is not None else const.cruise_speed_ms

    CL_alpha = wing_lift_slope(b ** 2 / wing_geo["S_wing"],
                               mach=const.cruise_speed_ms / const.cruise_speed_of_sound)
    CD0 = aero_params["zero_lift_drag"]

    if prop_y is None:
        # prop disc clear of the fuselage (half width 0.1 m) + 2 cm gap
        prop_y = const.prop_D * 0.0254 / 2.0 + 0.12
    prop_y = min(prop_y, b / 2.0)

    rows = []
    for cf_c in cf_c_grid:
        for y1f in y_inner_frac_grid:
            y1, y2 = y1f * b / 2.0, y_outer_frac * b / 2.0
            if y2 <= y1:
                continue
            # arm: elevon centroid is about (1 - cf_c/2) of chord aft of the
            # leading edge, CG sits at cg_frac_mac * MAC.
            l_e = (1.0 - cf_c / 2.0 - cg_frac_mac) * MAC

            d = elevon_derivatives(wing_geo, cf_c, y1, y2, CL_alpha, CD0, l_e, eta)
            delta_trim = -cm_trim / d["C_mdelta"]
            delta_a = delta_max - abs(delta_trim)
            p = roll_rate(d["C_ldelta"], d["C_lp"], max(delta_a, 0.0), V, b)

            h = hover_control(wing_geo, m0_final, cf_c, y1, y2, prop_y, const=const,
                              eta=eta, l_e=l_e, I_pitch=I_pitch,
                              I_axial=I_axial, delta_max=delta_max)

            ok_roll = np.degrees(p) >= roll_rate_req_dps
            ok_trim = abs(delta_trim) <= trim_fraction_max * delta_max
            ok_hover = (h["alpha_ddot_pitch"] >= hover_margin * hover_accel_req
                        and h["alpha_ddot_axial"] >= hover_margin * hover_accel_req)

            rows.append({
                "cf_c": cf_c, "y_in_m": y1, "y_out_m": y2,
                "S_e_m2": d["S_e"], "S_e/S": d["S_e_over_S"],
                "C_ldelta": d["C_ldelta"], "C_lp": d["C_lp"],
                "C_mdelta": d["C_mdelta"],
                "trim_deg": np.degrees(delta_trim),
                "roll_rate_dps": np.degrees(p),
                "hover_pitch_rad_s2": h["alpha_ddot_pitch"],
                "hover_axial_rad_s2": h["alpha_ddot_axial"],
                "ok_roll": ok_roll, "ok_trim": ok_trim, "ok_hover": ok_hover,
                "ok_all": bool(ok_roll and ok_trim and ok_hover),
                "_hover": h, "_l_e": l_e,
            })

    table = pd.DataFrame(rows)
    passing = table[table["ok_all"]]
    if len(passing):
        best = passing.sort_values("S_e_m2").iloc[0]
        feasible = True
    else:
        # nothing passes: report the least-bad candidate (most requirement
        # checks passed, then largest elevon)
        score = table[["ok_roll", "ok_trim", "ok_hover"]].sum(axis=1)
        best = table.assign(_score=score).sort_values(
            ["_score", "S_e_m2"], ascending=[False, False]).iloc[0]
        feasible = False

    return {
        "feasible": feasible,
        "best": best.drop(labels=["_hover", "_l_e"]).to_dict(),
        "hover": best["_hover"],
        "l_e": best["_l_e"],
        "CL_alpha": CL_alpha,
        "prop_y": prop_y,
        "delta_max_deg": delta_max_deg,
        "table": table.drop(columns=["_hover", "_l_e"]),
        "inputs": {
            "roll_rate_req_dps": roll_rate_req_dps, "cm_trim": cm_trim,
            "hover_accel_req": hover_accel_req, "hover_margin": hover_margin,
            "trim_fraction_max": trim_fraction_max, "V_roll": V,
        },
    }


def print_control_summary(result):
    b, h, inp = result["best"], result["hover"], result["inputs"]
    print(f"\n{'=' * 60}")
    print("ELEVON SIZING SUMMARY")
    print(f"{'=' * 60}")
    print("Requirements met by all checks" if result["feasible"]
          else "WARNING: no candidate passes every check; showing best effort")
    print(f"Chord ratio c_f/c:     {b['cf_c']:.2f}")
    print(f"Span extent:           {b['y_in_m']:.3f} to {b['y_out_m']:.3f} m (each side)")
    print(f"Elevon area (both):    {b['S_e_m2']:.4f} m^2  ({b['S_e/S'] * 100:.1f} % of wing)")
    print(f"C_l_delta / C_l_p:     {b['C_ldelta']:.4f} / {b['C_lp']:.4f}  per rad")
    print(f"C_m_delta:             {b['C_mdelta']:.4f}  per rad")
    print(f"Trim deflection:       {b['trim_deg']:.2f} deg (limit {inp['trim_fraction_max'] * result['delta_max_deg']:.1f})")
    print(f"Roll rate:             {b['roll_rate_dps']:.0f} deg/s (need {inp['roll_rate_req_dps']:.0f}) at {inp['V_roll']:.1f} m/s")
    print(f"Hover slipstream:      {h['V_slip']:.1f} m/s, {h['S_immersed']:.4f} m^2 elevon immersed")
    print(f"Hover pitch accel:     {h['alpha_ddot_pitch']:.1f} rad/s^2 (need {inp['hover_accel_req'] * inp['hover_margin']:.1f})")
    print(f"Hover axial accel:     {h['alpha_ddot_axial']:.1f} rad/s^2 (need {inp['hover_accel_req'] * inp['hover_margin']:.1f})")
    print(f"I_pitch / I_axial:     {h['I_pitch']:.4f} / {h['I_axial']:.4f} kg*m^2 (check these!)")
    print(f"{'=' * 60}\n")


def size_rudder(wing_geo, vstab_geo, cr_c=0.30, eta=0.9,
                CL_alpha_v=None, delta_max_deg=25.0, l_v=None, const=None):
    const = _get_const(const)
    S, b = wing_geo["S_wing"], wing_geo["wingspan"]
    S_v = vstab_geo["total_area"]
    l_v = l_v if l_v is not None else vstab_geo["arm"]
    if CL_alpha_v is None:
        AR_v = const.vertical_stab_aspect_ratio
        CL_alpha_v = wing_lift_slope(AR_v)
    tau_r = flap_effectiveness(cr_c, eta)
    V_v = S_v * l_v / (S * b)
    C_ndr = -CL_alpha_v * tau_r * V_v * cr_c
    return {
        "tau_r": tau_r, "V_v": V_v, "S_rudder": cr_c * S_v,
        "C_ndelta_r": C_ndr,
        "yaw_moment_at_max_per_q": abs(C_ndr) * np.radians(delta_max_deg) * S * b,
    }


def asb_control_derivatives(airplane, op_point, sym_name="elevon_sym",
                            asym_name="elevon_asym", d_deg=2.0):
    import aerosandbox as asb

    d = np.radians(d_deg)

    def run(sym, asym):
        ap = airplane.with_control_deflections({sym_name: sym, asym_name: asym})
        return asb.AeroBuildup(airplane=ap, op_point=op_point).run()

    plus_s, minus_s = run(+d_deg, 0.0), run(-d_deg, 0.0)
    plus_a, minus_a = run(0.0, +d_deg), run(0.0, -d_deg)
    return {
        "Cm_delta": float((plus_s["Cm"] - minus_s["Cm"]) / (2 * d)),
        "Cl_delta": float((plus_a["Cl"] - minus_a["Cl"]) / (2 * d)),
    }

def control_params(result, rudder=None, rudder_chord_ratio=0.30, rudder_delta_max_deg=25.0):
    b = result["best"]
    params = {
        "cf_c": float(b["cf_c"]),
        "y_inboard": float(b["y_in_m"]),
        "y_outboard": float(b["y_out_m"]),
        "elevon_delta_max_deg": float(result["delta_max_deg"]),
        "Cm_delta_e": float(b["C_mdelta"]),
        "Cl_delta_a": float(b["C_ldelta"]),
    }
    if rudder is not None:
        params.update({
            "rudder_chord_ratio": float(rudder_chord_ratio),
            "rudder_delta_max_deg": float(rudder_delta_max_deg),
            "Cn_delta_r": float(rudder["C_ndelta_r"]),
        })
    return params