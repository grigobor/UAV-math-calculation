import numpy as np

def calculate_wing_loading_params(const):
    """Calculate wing loading at stall condition."""
    stall_speed_ms = const.stall_speed / 3.6
    dynamic_pressure = 0.5 * const.air_density_sea_level * stall_speed_ms ** 2
    wing_loading = const.max_lift_coefficient_stat * dynamic_pressure
    return {
        "stall_speed_ms": stall_speed_ms,
        "dynamic_pressure": dynamic_pressure,
        "wing_loading": wing_loading
    }

def calculate_wing_geometry(m0_final, wing_loading, const):
    """Calculate wing area, span, and chord dimensions."""
    g = 9.81
    take_off_weight = m0_final * g
    S_wing = take_off_weight / wing_loading
    wingspan = np.sqrt(const.AR * S_wing)
    root_chord = (2 * S_wing) / (wingspan * (1 + const.taper_ratio))
    tip_chord = const.taper_ratio * root_chord
    MAC = (2/3) * root_chord * (1 + const.taper_ratio + const.taper_ratio ** 2) / (1 + const.taper_ratio)
    
    return {
        "S_wing": S_wing,
        "wingspan": wingspan,
        "root_chord": root_chord,
        "tip_chord": tip_chord,
        "MAC": MAC
    }

def calculate_cruise_aerodynamics(const, wing_geo):
    """Calculate cruise aerodynamic parameters."""
    fric_coef = 0.74 / (const.Re_cruise ** 0.2)
    zero_lift_drag = const.SwetS * fric_coef
    induced_drag_factor_K = 0.3183 / const.AR / const.e
    
    dynamic_pressure_cruise = 0.5 * const.air_density_cruise * const.cruise_speed_ms ** 2
    wing_loading = wing_geo["S_wing"] * const.g / (const.payload_mass * const.g)  # rough estimate
    
    CL_opt = np.sqrt(zero_lift_drag / induced_drag_factor_K)
    CD_opt = zero_lift_drag + induced_drag_factor_K * CL_opt ** 2
    aerodynamic_quality = CL_opt / CD_opt
    
    return {
        "zero_lift_drag": zero_lift_drag,
        "induced_drag_factor_K": induced_drag_factor_K,
        "dynamic_pressure_cruise": dynamic_pressure_cruise,
        "CL_opt": CL_opt,
        "CD_opt": CD_opt,
        "aerodynamic_quality": aerodynamic_quality
    }