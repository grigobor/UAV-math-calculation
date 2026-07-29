import jsbsim
import numpy as np

def run_jsbsim_test():
    # 1. Initialize the FDM engine
    # Ensure your script points to the folder containing the generated file
    fdm = jsbsim.FGFDMExec(root_dir=".") 
    fdm.load_model("Adaptive_Tailsitter")
    
    # 2. Define the initial launch state for a Tailsitter (VTOL Launch)
    fdm["ic/vc-kts"] = 0.0          # Airspeed 0
    fdm["ic/h-sl-ft"] = 100.0       # Starting altitude above sea level
    fdm["ic/theta-deg"] = 90.0      # Crucial: Nose pointed directly up for VTOL!
    
    # Run the initialization setup
    fdm.run_ic()
    
    # 3. Run a brief 5-second dynamic simulation loop
    time_steps = []
    pitch_angles = []
    altitudes = []
    
    while fdm.run():
        current_time = fdm["simulation/sim-time-sec"]
        if current_time >= 5.0:
            break
            
        time_steps.append(current_time)
        pitch_angles.append(fdm["attitude/theta-deg"])
        altitudes.append(fdm["position/h-sl-meters"])
        
    return time_steps, pitch_angles, altitudes

def generate_jsbsim_aircraft_xml(wing_geometry, mass_kg, filename, alphas, cls, cds, cms):
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
        <author>Grigory</author>
        <description>Auto-generated from AeroSandbox/NeuralFoil Optimization</description>
    </fileheader>

    <metrics>
        <wingarea unit="M2"> {wing_geometry['S_wing']:.4f} </wingarea>
        <wingspan unit="M"> {wing_geometry['wingspan']:.4f} </wingspan>
        <chord unit="M"> {wing_geometry['MAC']:.4f} </chord>
        <location name="AERORP" unit="M">
            <x> {aerorp_x:.4f} </x> <y> 0.0 </y> <z> 0.0 </z>
        </location>
        <location name="VRP" unit="M">
            <x> 0.0 </x> <y> 0.0 </y> <z> 0.0 </z>
        </location>
    </metrics>

    <mass_balance>
        <ixx unit="KGM2"> {Ixx:.4f} </ixx>
        <iyy unit="KGM2"> {Iyy:.4f} </iyy>
        <izz unit="KGM2"> {Izz:.4f} </izz>
        <emptywt unit="KG"> {mass_kg:.4f} </emptywt>
        <location name="CG" unit="M">
            <x> {cg_x:.4f} </x> <y> 0.0 </y> <z> 0.0 </z>
        </location>
    </mass_balance>

    <aerodynamics>
        <axis name="LIFT">
            <function name="aero/force/Lift_alpha">
                <product>
                    <property>aero/qbar-area</property>
                    <table>
                        <independentVar lookup="row">aero/alpha-deg</independentVar>
                        <tableData>
{table_cl}
                        </tableData>
                    </table>
                </product>
            </function>
        </axis>

        <axis name="DRAG">
            <function name="aero/force/Drag_alpha">
                <product>
                    <property>aero/qbar-area</property>
                    <table>
                        <independentVar lookup="row">aero/alpha-deg</independentVar>
                        <tableData>
{table_cd}
                        </tableData>
                    </table>
                </product>
            </function>
        </axis>

        <axis name="PITCH">
            <function name="aero/moment/Pitch_alpha">
                <product>
                    <property>aero/qbar-area</property>
                    <property>metrics/cbar-m</property>
                    <table>
                        <independentVar lookup="row">aero/alpha-deg</independentVar>
                        <tableData>
{table_cm}
                        </tableData>
                    </table>
                </product>
            </function>
        </axis>
    </aerodynamics>
</fdm_config>
"""
    with open(filename, "w") as f:
        f.write(xml_content)
    print(f"JSBSim model saved successfully to: {filename}")