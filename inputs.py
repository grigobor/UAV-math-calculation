import math

class DesignConstants:
    g = 9.81 

    eng_q = 2 # Number of engines
    con_cell_num = 6 # Number of cells in series in one battery string
    cell_U = 4.2 # Nominal cell voltage, V
    prop_D = 22 # Propeller diameter, inches

    LD_ratio_cruise = 14 # Cruise lift-to-drag ratio (L/D)
    prop_efficiency = 0.75 # Propeller efficiency η_p
    battery_specific_energy = 216 # Battery specific energy, Wh/kg
    payload_mass = 1.5 # Payload mass, kg
    flight_time = 90 # Required flight time, minutes
    flight_time_hr = flight_time / 60 # Required flight time, hours
    cruise_speed = 90 # Cruise speed, km/h
    cruise_speed_ms = cruise_speed / 3.6 # Cruise speed, m/s
    takeoff_mass_reserve = 1.07 # Take-off mass margin factor
    battery_mass_reserve = 1.1 # Battery mass margin factor
    k = 0.3 # Empirical coefficient for propeller mass formula
    n = 2.3 # Empirical exponent for propeller mass formula
    prop_fasteners = 0.015 # Additional mass of propeller fasteners per powerplant
    e = 0.78 # Oswald efficiency factor
    k_structure = 0.3 # Structure mass fraction

    stall_speed = 55 # stall speed (kph)
    air_density_sea_level = 1.225 # air density on takeoff (sea level)
    takeoff_altitude = 0 # Take off altitude, m
    cruise_altitude = 3000 # Cruise altitude, m
    max_lift_coefficient_stat = 1.4 # maximum wing lift coefficient (statistically)
    min_lift_coefficient_stat = -0.8 # minimum wing lift coefficient (from the aerofoil polar)
    derivative_max_lift_coefficient_stat = 5.25 # derivative of max lift coefficient (by statistic)
    WP = 0.1 # energy armament (by prototype)
    cruise_speed_of_sound = 327.7 # Speed of sound at cruise altitude, m/s

    # Battery parameters
    series_cells_n = 6
    parallel_cells_n = 4 
    cell_U_min = 3
    cell_U_max = 4.2
    cell_capacity_nom_A = 4.2 
    cell_capacity_nom_W = 15.5
    max_current_cell = 45
    cell_mass = 0.07
    battery_mass_coef = 1.12
    screws_n = 2
    blades_n = 2
    Kp = 0.0995

    screws_n = 2
    blades_n = 2
    Kp = 0.0995

    # 
    Re_cruise = 511979
    SwetS = 2 # According to the table from the sources
    AR = 8 # Aspect ratio (Presumably, need to correct or use Aerosandbox optimization)
    air_density_cruise = 1.1117
    dynamic_viscosity_cruise = 0.00001787
    prop_efficiency_cruise = 0.75
    battery_depth = 0.3

    # Wing parametrs
    taper_ratio = 0.57

    # Speed on transition
    Vy_hover = 0.0
    Vy_climb = 3.0
    Vy_ceiling = 0.5

    # Prop characteristics
    Vtip = 80  # м/с
    solidity = 0.1
    Cd_blade = 0.02
    induced_power_factor_hover = 1.15
    induced_power_factor_climb = 1.15
    # disc_loading_max_diameter = 0.246
    total_disc_area_max = 0.2463 * 2

    # Angels
    alpha_rotor_trans = 0

    # Wing aerodynamic
    wing_loading_optimum_airborne = 300  # Н/м²

    climbe_rate = 3 # m/s
    ceiling_rate_of_climb = 0.5 # m/s
    max_lift_coefficient = 1.4

    # Airfoil parameters
    number_point_on_chord = 30
    phi = math.radians(90 / (number_point_on_chord - 1))
    profile_thickness = 0.12
    profile_curv = 0.04
    max_curv_point = 0.4
    profile_max_lift_coefficient = 1.55

    # Wing characteristics
    wing_narrowing = 0.5
    sweep_angel_1_2 = 0
    angel_of_incedence = 0
    dihedral_angel = 0
    wing_tip_twist = 0

    drag_coefficient_constant = 0.045161009

    vertical_stab_aspect_ratio = 2

    # Геометрические параметры килей (примерные, будут масштабироваться)
    # Центральный киль (вверх)
    central_root_chord = 0.50
    central_tip_chord  = 0.25
    central_span       = 0.80   # высота киля

    # Боковые кили (вниз)
    side_root_chord = 0.40
    side_tip_chord  = 0.20
    side_span       = 0.60

    fraction_central = 0.40   # 40% общей площади — центральный киль вверх