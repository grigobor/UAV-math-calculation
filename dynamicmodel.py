import jsbsim
import os
import numpy as np
import matplotlib.pyplot as plt

def run_jsbsim():
    # 1. Initialize the JSBSim Flight Dynamics Model (FDM) Executive
    # Point root_dir to the folder containing your 'aircraft' directory
    fdm = jsbsim.FGFDMExec(root_dir=os.getcwd())

    # 2. Load your freshly generated aircraft model
    # JSBSim look inside aircraft/Adaptive_Tailsitter/Adaptive_Tailsitter.xml
    fdm.load_model('Adaptive_Tailsitter')

    # 3. Configure Initial Conditions for a Tailsitter (VTOL Mode)
    fdm['ic/vc-kts'] = 0.0          # Zero initial forward airspeed
    fdm['ic/h-sl-ft'] = 3.3         # Start 3.3 feet (1 meter) above ground level
    fdm['ic/theta-deg'] = 90.0      # Crucial: 90-degree pitch angle (nose straight up on the pad)
    fdm['ic/phi-deg'] = 0.0         # Roll angle
    fdm['ic/psi-true-deg'] = 0.0    # Heading

    # Apply initial conditions to populate state vectors
    fdm.run_ic()

    # 4. Prepare data logging containers
    sim_time = []
    altitude = []
    pitch_angle = []
    thrust_cmd = []

    # 5. The Simulation Loop (Step through time)
    # Default JSBSim time step is usually 1/120th of a second (dt = 0.008333s)
    sim_duration_sec = 15.0 

    while fdm.run():
        current_time = fdm['simulation/sim-time-sec']
        sim_time.append(current_time)
        
        # Log properties of interest
        altitude.append(fdm['position/h-sl-ft'])
        pitch_angle.append(fdm['attitude/theta-deg'])
        thrust_cmd.append(fdm['fcs/throttle-cmd-norm'])
        
        # Example: Inject control inputs or pilot commands over time
        if current_time < 2.0:
            fdm['fcs/throttle-cmd-norm'] = 0.85  # Punch throttle to lift off vertically
        elif 2.0 <= current_time < 6.0:
            fdm['fcs/throttle-cmd-norm'] = 0.60  # Steady hover power
            fdm['fcs/elevator-cmd-norm'] = 0.1   # Command a slight pitch-forward moment to begin transition
        else:
            # Simulate moving into forward flight
            fdm['fcs/elevator-cmd-norm'] = 0.0
        
        # Break loop once target time is achieved
        if current_time >= sim_duration_sec:
            break

    print(f"Calculation complete. Simulated {len(sim_time)} iterations successfully.")

    # 6. Plotting your Dynamic Aerodynamic Response
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(sim_time, altitude, 'b-', label='Altitude (ft)')
    ax1.set_ylabel('Altitude [ft]')
    ax1.grid(True)

    ax2.plot(sim_time, pitch_angle, 'r-', label='Pitch Angle (deg)')
    ax2.axhline(90, color='gray', linestyle='--', label='Vertical Launch Orientation')
    ax2.set_ylabel('Pitch [deg]')
    ax2.set_xlabel('Simulation Time [sec]')
    ax2.grid(True)
    ax2.legend()
    plt.show()

def generate_jsbsim_aircraft_xml(wing_geometry, stabilizer_geometry, mass_kg, filename, alphas, cls, cds, cms):
    """Generates a JSBSim flight dynamics model configuration from Python variables."""
    
    alphas = np.array(alphas).flatten()
    cls = np.array(cls).flatten()
    cds = np.array(cds).flatten()
    cms = np.array(cms).flatten()

    Ixx = (1.0 / 12.0) * mass_kg * (wing_geometry['wingspan']**2)
    Iyy = (1.0 / 12.0) * mass_kg * (wing_geometry['MAC']**2)
    Izz = Ixx + Iyy

    # Reference point locations (Structural frame: X=0 at nose, pointing aft)
    # For a stable tailless flying wing, CG must be slightly ahead of the Aerodynamic Center (AERORP)
    aerorp_x = 0.25 * wing_geometry['MAC']  # Aerodynamic center at 25% MAC
    static_margin = 0.05       # 5% static stability margin stability requirement
    cg_x = aerorp_x - (static_margin * wing_geometry['MAC']) # CG is forward of AERORP

    # Convert numpy arrays to space-separated strings for XML tables
    table_cl = "\n".join([f"    {a:6.1f} {cl:6.4f}" for a, cl in zip(alphas, cls)])
    table_cd = "\n".join([f"    {a:6.1f} {cd:6.4f}" for a, cd in zip(alphas, cds)])
    table_cm = "\n".join([f"    {a:6.1f} {cm:6.4f}" for a, cm in zip(alphas, cms)])

    xml_content = f"""<?xml version="1.0"?>
<fdm_config name="Adaptive_Tailsitter" version="2.0" release="BETA">
    <fileheader>
        <author>Grigorii Borodachev</author>
        <organization>
            Department of Aeronautical Engineering,
            Moscow Aviation Institut,
            Russia
        </organization>
        <version>0.1</version>
        <description>Model a 2026 G1-T UAV</description>
    </fileheader>

    <metrics>
        <wingarea unit="M2"> {wing_geometry['S_wing']:.4f} </wingarea>
        <wingspan unit="M"> {wing_geometry['wingspan']:.4f} </wingspan>
        <chord unit="M"> {wing_geometry['MAC']:.4f} </chord>
        <vtailarea unit="M2"> {stabilizer_geometry['total_area']:.4f} </vtailarea>
        <vtailarm unit="M"> {stabilizer_geometry['arm']:.4f} </vtailarm>
        <location name="AERORP" unit="M">
            <x> {aerorp_x:.4f} </x> <y> 0.0 </y> <z> 0.0 </z>
        </location>
        <location name="VRP" unit="M">
            <x> 0.0 </x> <y> 0.0 </y> <z> 0.0 </z>
        </location>
    </metrics>

    <mass_balance>
        <ixx unit="KG*M2"> {Ixx:.4f} </ixx>
        <iyy unit="KG*M2"> {Iyy:.4f} </iyy>
        <izz unit="KG*M2"> {Izz:.4f} </izz>
        <emptywt unit="KG"> {mass_kg:.4f} </emptywt>
        <location name="CG" unit="M">
            <x> {cg_x:.4f} </x> <y> 0.0 </y> <z> 0.0 </z>
        </location>
    </mass_balance>

    <ground_reactions>
        <contact type="STRUCTURE" name="WING_TIPS">
            <location unit="M">
                <x>-0.2</x>
                <y>0.0</y>
                <z>0.0</z>
            </location>
            <static_friction_coefficient>0.8</static_friction_coefficient>
            <dynamic_friction_coefficient>0.6</dynamic_friction_coefficient>
        </contact>
    </ground_reactions>

    <aerodynamics aero_ref_pt_hx="0.0546" aero_ref_pt_hy="0.0" aero_ref_pt_hz="0.0">
        <axis name="LIFT">
            <function name="CL">
                <table>
                    <independentVar lookup="row">aero/alpha-deg</independentVar>
                    <tableData>
{table_cl}
                    </tableData>
                </table>
            </function>
            <function name="CD">
                <table>
                    <independentVar lookup="row">aero/alpha-deg</independentVar>
                    <tableData>
{table_cd}
                    </tableData>
                </table>
            </function>
            <function name="Cm">
                <table>
                    <independentVar lookup="row">aero/alpha-deg</independentVar>
                    <tableData>
{table_cm}
                    </tableData>
                </table>
            </function>
        </axis>
    </aerodynamics>
</fdm_config>
"""
    with open(filename, "w") as f:
        f.write(xml_content)
    print(f"JSBSim model saved successfully to: {filename}")