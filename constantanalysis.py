import matplotlib.pyplot as plt
import numpy as np
import aerosandbox.numpy as np

def calculate_constraint_diagram(const, m0_final, wing_geo, aero_params):
    """Calculate all power and disc loading constraints for the constraint diagram."""
    g = 9.81
    weight_takeoff = m0_final * g
    
    disc_loading_max = weight_takeoff / const.total_disc_area_max
    wing_loading_stall = const.max_lift_coefficient * (const.air_density_sea_level * (const.stall_speed / 3.6) ** 2) / const.prop_efficiency
    
    wing_loading_range = np.arange(150, 600, 1, dtype=float)
    disc_loading_range = np.linspace(20, 200, 200)
    
    # Fixed-wing mode constraints (horizontal flight)
    q = 0.5 * const.air_density_cruise * const.cruise_speed_ms ** 2
    n_load = 1.5
    
    def pw_cruise(wl):
        return (const.cruise_speed_ms / const.prop_efficiency) * (q * aero_params["zero_lift_drag"] / wl + 
                                                                   aero_params["induced_drag_factor_K"] * wl / q)
    
    def pw_turn(wl):
        return (const.cruise_speed_ms / const.prop_efficiency) * (q * aero_params["zero_lift_drag"] / wl + 
                                                                   aero_params["induced_drag_factor_K"] * n_load ** 2 * wl / q)
    
    def pw_climb(wl):
        return (const.climbe_rate / const.prop_efficiency) + (const.cruise_speed_ms / const.prop_efficiency) * \
               (q * aero_params["zero_lift_drag"] / wl + aero_params["induced_drag_factor_K"] * wl / q)
    
    def pw_ceiling(wl):
        return (const.ceiling_rate_of_climb / const.prop_efficiency) + (const.cruise_speed_ms / const.prop_efficiency) * \
               4 * np.sqrt(aero_params["induced_drag_factor_K"] * aero_params["zero_lift_drag"] / 3)
    
    # VTOL mode constraints
    def pw_hover(dl):
        A = weight_takeoff / dl
        P_ind = const.induced_power_factor_hover * np.sqrt(weight_takeoff) / np.sqrt(2 * const.air_density_sea_level * A)
        P_prof = const.air_density_sea_level * A * const.Vtip ** 3 * const.solidity * const.Cd_blade / (8 * weight_takeoff)
        return P_ind + P_prof
    
    def pw_vertical_climb(dl, Vy):
        term1 = Vy * (1 - const.induced_power_factor_climb / 2)
        term2 = (const.induced_power_factor_climb / 2) * np.sqrt(Vy ** 2 + 2 * dl / const.air_density_sea_level)
        term3 = const.air_density_sea_level * const.Vtip ** 3 * const.solidity * const.Cd_blade / (8 * dl)
        return term1 + term2 + term3
    
    # Compute constraint arrays
    pw_cruise_arr = np.array([pw_cruise(wl) for wl in wing_loading_range])
    pw_turn_arr = np.array([pw_turn(wl) for wl in wing_loading_range])
    pw_climb_arr = np.array([pw_climb(wl) for wl in wing_loading_range])
    pw_ceiling_arr = np.array([pw_ceiling(wl) for wl in wing_loading_range])
    
    pw_hover_arr = np.array([pw_hover(dl) for dl in disc_loading_range])
    pw_vclimb_arr = np.array([pw_vertical_climb(dl, const.Vy_climb) for dl in disc_loading_range])
    pw_vceiling_arr = np.array([pw_vertical_climb(dl, const.Vy_ceiling) for dl in disc_loading_range])
    
    return {
        "wing_loading_range": wing_loading_range,
        "disc_loading_range": disc_loading_range,
        "pw_cruise": pw_cruise_arr,
        "pw_turn": pw_turn_arr,
        "pw_climb": pw_climb_arr,
        "pw_ceiling": pw_ceiling_arr,
        "pw_hover": pw_hover_arr,
        "pw_vclimb": pw_vclimb_arr,
        "pw_vceiling": pw_vceiling_arr,
        "wing_loading_stall": wing_loading_stall,
        "disc_loading_max": disc_loading_max
    }

def plot_constraint_diagram(constraints, m0_final, const):
    """Plot hybrid VTOL+Fixed-Wing constraint diagram."""
    fig, ax1 = plt.subplots(figsize=(16, 10))
    
    # Left axis: Wing Loading vs P/W (horizontal flight)
    ax1.plot(constraints['pw_turn'], constraints['wing_loading_range'], ':', color='blue', linewidth=3, label='Turn')
    ax1.plot(constraints['pw_climb'], constraints['wing_loading_range'], '--', color='blue', linewidth=3, label='Climb')
    ax1.plot(constraints['pw_cruise'], constraints['wing_loading_range'], '-.', color='blue', linewidth=3, label='Cruise')
    ax1.plot(constraints['pw_ceiling'], constraints['wing_loading_range'], '-', color='blue', linewidth=3, label='Ceiling')
    ax1.axhline(y=constraints['wing_loading_stall'], color='red', linewidth=3, label='Stall Limit')
    
    ax1.set_xlabel('Power density, P/W (W/N)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Wing loading, W/S (N/m²)', fontsize=14, color='navy')
    ax1.set_title('Hybrid UAV Constraints Diagram (VTOL + Fixed-Wing)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    
    # Right axis: Disc Loading vs P/W (vertical flight)
    ax2 = ax1.twinx()
    ax2.plot(constraints['pw_hover'], constraints['disc_loading_range'], '--', color='orange', lw=3, label='Hover')
    ax2.plot(constraints['pw_vclimb'], constraints['disc_loading_range'], '-.', color='orange', lw=3, label='Vert. climb')
    ax2.plot(constraints['pw_vceiling'], constraints['disc_loading_range'], ':', color='orange', lw=3, label='VTOL ceiling')
    ax2.axhline(y=constraints['disc_loading_max'], color='black', linestyle=':', lw=3, label='Prop diameter limit')
    
    ax2.set_ylabel('Disc loading, W/A (N/m²)', fontsize=14, color='darkred')
    ax2.tick_params(axis='y', labelcolor='darkred')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(0.01, 0.99), 
              fontsize=11, framealpha=0.95)
    
    # Annotation box
    ax1.text(0.02, 0.05, f"Mass: {m0_final:.1f} kg\nSpeed: {const.cruise_speed} km/h\nFlight time: {const.flight_time} min",
            transform=ax1.transAxes, fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="gray"))
    
    plt.tight_layout()
    plt.show()