def calculate_battery_params(const, m_battery):
    """Calculate battery pack parameters from cell specifications."""
    battery_weight = m_battery * const.g
    
    cell_total = const.series_cells_n * const.parallel_cells_n
    pack_cell_U_min = const.series_cells_n * const.cell_U_min
    pack_cell_U_max = const.series_cells_n * const.cell_U_max
    pack_cell_U = const.series_cells_n * const.cell_U
    pack_cell_capacity_A = const.parallel_cells_n * const.cell_capacity_nom_A
    pack_cell_capacity_W = const.parallel_cells_n * const.cell_capacity_nom_W
    pack_max_current = const.parallel_cells_n * const.max_current_cell
    pack_cell_mass = cell_total * const.cell_mass * const.battery_mass_coef
    
    total_battery_mass = pack_cell_mass
    total_battery_capacity = total_battery_mass / const.battery_mass_coef / const.cell_mass * const.parallel_cells_n
    
    return {
        "cell_total": cell_total,
        "pack_U_min": pack_cell_U_min,
        "pack_U_max": pack_cell_U_max,
        "pack_U": pack_cell_U,
        "pack_capacity_Ah": pack_cell_capacity_A,
        "pack_capacity_Wh": pack_cell_capacity_W,
        "pack_max_current": pack_max_current,
        "pack_mass": total_battery_mass,
        "battery_weight": battery_weight
    }

def print_battery_summary(battery_params):
    """Print battery parameters summary."""
    print(f"\n{'='*60}")
    print(f"BATTERY PACK SUMMARY")
    print(f"{'='*60}")
    print(f"Total cells: {battery_params['cell_total']}")
    print(f"Pack voltage: {battery_params['pack_U']:.1f} V (range: {battery_params['pack_U_min']:.1f}-{battery_params['pack_U_max']:.1f} V)")
    print(f"Pack capacity: {battery_params['pack_capacity_Ah']:.1f} Ah / {battery_params['pack_capacity_Wh']:.1f} Wh")
    print(f"Max current: {battery_params['pack_max_current']:.1f} A")
    print(f"Pack mass: {battery_params['pack_mass']:.3f} kg")
    print(f"{'='*60}\n")