def calculate_zero_approximation_mass(payload_mass):
    """Calculate zero approximation of takeoff mass based on payload."""
    return 5.147 * payload_mass ** 0.849

def iterate_takeoff_mass(const, tol=0.5, max_iter=50):
    """
    Iteratively converge takeoff mass until tolerance is met.
    Returns dict with final values and iteration history.
    """
    m0 = calculate_zero_approximation_mass(const.payload_mass)
    g = 9.81
    history = []

    for i in range(max_iter):
        W0 = m0 * g
        V_cruise = const.cruise_speed / 3.6

        # Required cruise power
        P_req = W0 * V_cruise / (const.LD_ratio_cruise * const.prop_efficiency)
        P_per_motor = P_req / const.eng_q

        # Motor mass (statistical, 6S)
        U_pack = 6 * 4.2
        m_motor_one = 0.889 * (P_per_motor ** -0.288) * (U_pack ** 0.1588)

        # Propeller mass
        m_prop_one = const.k * (const.prop_D ** const.n) / 1000.0

        # Powerplant total
        m_powerplant = const.eng_q * (m_motor_one + m_prop_one) + const.prop_fasteners * const.eng_q

        # Battery
        E_req_wh = P_req * const.flight_time_hr
        m_battery = (E_req_wh / const.battery_specific_energy) * const.battery_mass_reserve

        # Structure & equipment
        m_structure = const.k_structure * m0
        m_eq = 0.3  # equipment fixed mass

        # New mass estimate
        m0_new = (const.payload_mass + m_structure + m_eq + m_powerplant + m_battery)
        m0_new *= const.takeoff_mass_reserve

        error = abs(m0_new - m0) / m0_new * 100.0
        history.append({
            "iter": i+1, "m0_kg": round(m0, 4), "m0_new_kg": round(m0_new, 4),
            "error_%": round(error, 3), "P_req_W": round(P_req, 1),
            "m_battery_kg": round(m_battery, 4), "m_powerplant_kg": round(m_powerplant, 4)
        })

        if error < tol:
            print(f"✓ Mass convergence in {i+1} iterations (Δ < {tol} %)")
            break

        m0 = m0_new

    result = {
        "final_m0_kg": m0_new,
        "m_battery_kg": m_battery,
        "m_powerplant_kg": m_powerplant,
        "P_req_W": P_req,
        "history": pd.DataFrame(history)
    }
    return result

def print_mass_summary(mass_result, const):
    """Print summary of mass iteration."""
    print(f"\n{'='*60}")
    print(f"MASS CONVERGENCE SUMMARY")
    print(f"{'='*60}")
    print(f"Final takeoff mass: {mass_result['final_m0_kg']:.3f} kg")
    print(f"Battery mass: {mass_result['m_battery_kg']:.3f} kg")
    print(f"Powerplant mass: {mass_result['m_powerplant_kg']:.3f} kg")
    print(f"Required power: {mass_result['P_req_W']:.1f} W")
    print(f"{'='*60}\n")