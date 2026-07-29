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
import masscalculation
import wingparametrs
import battery
import constantanalysis
import wingprofile
import verticalstabilisator
import buildinguav
import dynamicmodel

print("\n" + "="*80)
print("STARTING UAV DESIGN CALCULATION".center(80))
print("="*80)

# STEP 1: MASS CONVERGENCE
print("\n[1/7] Computing takeoff mass convergence...")
mass_result = masscalculation.iterate_takeoff_mass(inputs.DesignConstants, tol=0.5, max_iter=50)
m0_final = mass_result['final_m0_kg']
m_battery = mass_result['m_battery_kg']
masscalculation.print_mass_summary(mass_result, inputs.DesignConstants)


# STEP 2: WING LOADING AND STALL CONDITIONS
print("\n[2/7] Computing wing loading parameters...")
wing_load_params = wingparametrs.calculate_wing_loading_params(inputs.DesignConstants)
wing_loading = wing_load_params['wing_loading']
print(f"✓ Stall speed: {wing_load_params['stall_speed_ms']:.2f} m/s")
print(f"✓ Wing loading at stall: {wing_loading:.1f} N/m²")


# STEP 3: WING GEOMETRY
print("\n[3/7] Computing wing geometry...")
wing_geo = wingparametrs.calculate_wing_geometry(m0_final, wing_loading, inputs.DesignConstants)
print(f"✓ Wing area: {wing_geo['S_wing']:.3f} m²")
print(f"✓ Wingspan: {wing_geo['wingspan']:.3f} m")
print(f"✓ Root chord: {wing_geo['root_chord']:.3f} m")
print(f"✓ Tip chord: {wing_geo['tip_chord']:.3f} m")
print(f"✓ MAC: {wing_geo['MAC']:.3f} m")


# STEP 4: CRUISE AERODYNAMICS
print("\n[4/7] Computing cruise aerodynamics...")
aero_params = wingparametrs.calculate_cruise_aerodynamics(inputs.DesignConstants, wing_geo)
print(f"✓ Zero-lift drag coefficient: {aero_params['zero_lift_drag']:.5f}")
print(f"✓ Induced drag factor K: {aero_params['induced_drag_factor_K']:.5f}")
print(f"✓ Optimal L/D: {aero_params['aerodynamic_quality']:.2f}")


# STEP 5: BATTERY PARAMETERS
print("\n[5/7] Computing battery parameters...")
battery_params = battery.calculate_battery_params(inputs.DesignConstants, m_battery)
battery.print_battery_summary(battery_params)


# STEP 6: WING CHARACTERISTICS. Using only for calculating wing airfoil from handbook.
# print("\n[6/7] Computing wing characteristics and polar...")
# wing_chars = calculate_wing_characteristics(DesignConstants, wing_geo)
# wing_polar = calculate_wing_polar(DesignConstants, wing_chars)
# print(f"✓ Max lift coefficient: {wing_chars['max_lift_coeff']:.3f}")
# print(f"✓ Critical angle: {wing_chars['critical_angle']:.1f}°")


# STEP 6: VERTICAL STABILIZER
print("\n[7/7] Computing vertical stabilizer...")
vstab_geo = verticalstabilisator.calculate_vertical_stabilizer(wing_geo['wingspan'], wing_geo['S_wing'], inputs.DesignConstants)
print(f"✓ Vertical stabilizer area: {vstab_geo['total_area']:.3f} m²")
print(f"✓ Scale factor (central): {vstab_geo['scale_central']:.3f}x")
print(f"✓ Scale factor (sides): {vstab_geo['scale_side']:.3f}x")


# VISUALIZATIONS
# print("\n" + "="*80)
# print("GENERATING VISUALIZATIONS".center(80))
# print("="*80)

# STEP 7: Wing Profile
print("\n→ Plotting airfoil profile...")
# profile_data = calculate_airfoil_coordinates(DesignConstants)
# plot_airfoil_profile(profile_data)
my_profile_data = wingprofile.create_airfoil()

# 2. Wing Polar
# print("\n→ Plotting wing drag polar...")
# plot_wing_polar(wing_polar)

# STEP 8: Constraint Diagram
print("\n→ Plotting constraint diagram...")
constraints = constantanalysis.calculate_constraint_diagram(inputs.DesignConstants, m0_final, wing_geo, aero_params)
constantanalysis.plot_constraint_diagram(constraints, m0_final, inputs.DesignConstants)

# STEP 9: Airplane with take-off wing profile 3-View
print("\n→ Plotting airplane with take-off wing profile three-view...")
airplane_take_off_wing = buildinguav.build_airplane_model(inputs.DesignConstants, wing_geo, vstab_geo, my_profile_data[1], my_profile_data[1])
buildinguav.plot_airplane_views(airplane_take_off_wing)

# STEP 10: Airplane with cruise wing profile 3-View
print("\n→ Plotting airplane with cruise wing profile three-view...")
airplane_cruise_wing = buildinguav.build_airplane_model(inputs.DesignConstants, wing_geo, vstab_geo, my_profile_data[0], my_profile_data[1])
buildinguav.plot_airplane_views(airplane_cruise_wing)

# STEP 11: Dynamic model of aircraft with help of JSBSim
dynamicmodel.generate_jsbsim_aircraft_xml(wing_geo, 
                                          vstab_geo,
                                          mass_result['final_m0_kg'], 
                                          "Adaptive_Tailsitter.xml", 
                                          my_profile_data[2],  
                                          my_profile_data[3], 
                                          my_profile_data[4], 
                                          my_profile_data[5])

dynamicmodel.run_jsbsim()

# SUMMARY REPORT
print("\n" + "="*80)
print("DESIGN SUMMARY REPORT".center(80))
print("="*80)

summary_report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                         HYBRID VTOL + FIXED-WING UAV                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📦 MASS BREAKDOWN
  ├─ Takeoff Mass:        {m0_final:.2f} kg
  ├─ Battery Mass:        {m_battery:.2f} kg
  ├─ Powerplant Mass:     {mass_result['m_powerplant_kg']:.2f} kg
  └─ Payload Mass:        {inputs.DesignConstants.payload_mass:.2f} kg

✈️  WING PARAMETERS
  ├─ Wing Area:           {wing_geo['S_wing']:.3f} m²
  ├─ Wingspan:            {wing_geo['wingspan']:.3f} m
  ├─ Mean Aero Chord:     {wing_geo['MAC']:.3f} m
  ├─ Aspect Ratio:        {inputs.DesignConstants.AR:.1f}
  ├─ Taper Ratio:         {inputs.DesignConstants.taper_ratio:.2f}
  └─ Root/Tip Chord:      {wing_geo['root_chord']:.3f} / {wing_geo['tip_chord']:.3f} m

⚡ POWER & PERFORMANCE
  ├─ Required Power:      {mass_result['P_req_W']:.0f} W
  ├─ Cruise Speed:        {inputs.DesignConstants.cruise_speed} km/h ({inputs.DesignConstants.cruise_speed_ms:.1f} m/s)
  ├─ Stall Speed:         {inputs.DesignConstants.stall_speed} km/h ({wing_load_params['stall_speed_ms']:.1f} m/s)
  ├─ L/D Optimal:         {aero_params['aerodynamic_quality']:.2f}
  └─ Flight Time:         {inputs.DesignConstants.flight_time} min

🔋 BATTERY PACK
  ├─ Total Cells:         {battery_params['cell_total']}
  ├─ Pack Voltage:        {battery_params['pack_U']:.1f} V
  ├─ Capacity:            {battery_params['pack_capacity_Ah']:.1f} Ah
  └─ Mass:                {battery_params['pack_mass']:.3f} kg

🎯 VERTICAL STABILIZER
  ├─ Total Area:          {vstab_geo['total_area']:.3f} m²
  ├─ Central Fin Area:    {vstab_geo['S_central']:.3f} m²
  ├─ Side Fins Area:      {vstab_geo['S_side']:.3f} m² (each)
  └─ Span:                {vstab_geo['span']:.3f} m
"""

print(summary_report)
print("="*80)
print("✓ All calculations completed successfully!".center(80))
print("="*80 + "\n")