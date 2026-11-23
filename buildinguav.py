import aerosandbox as asb
import numpy as np

def build_airplane_model(const, wing_geo, vstab_geo, profile_1, profile_2, controls=None):
    """Build and return Aerosandbox airplane model."""
    a = 0.05 # Wingtip offset (m)
    b = 0.15 # Wingroot offset (m)
    rudder_chord_ratio=controls.get('rudder_chord_ratio') if controls else None
    airplane = asb.Airplane(
        name="Hybrid VTOL Fixed-Wing UAV",
        xyz_ref=[wing_geo['root_chord'] * 0.25, 0, 0],
        s_ref=wing_geo['S_wing'],
        c_ref=wing_geo['MAC'],
        b_ref=wing_geo['wingspan'],
        #b_ref=const.AR,
        wings=[
            # Main Wing
            asb.Wing(
                name="Main Wing",
                symmetric=True,
                xsecs=_main_wing_cross_section_xsecs(wing_geo, profile_1.to_airfoil(), a, b, controls)
                ),
            # Central Vertical Fin (up)
            asb.Wing(
                name="Central Fin Up",
                symmetric=False,
                xsecs=_fin_cross_section_xsecs(
                    [0.17, 0, 0],
                    const.central_root_chord * vstab_geo['scale_central'],
                    [0.17 + 0.05, 0, const.central_span * vstab_geo['scale_central']],
                    const.central_tip_chord * vstab_geo['scale_central'],
                    "rudder_central", rudder_chord_ratio),
                    ),
            # Left Vertical Fin (down)
            asb.Wing(
                name="Left Fin Down",
                symmetric=False,
                xsecs=_fin_cross_section_xsecs(
                    [0.17, wing_geo['wingspan']/4, 0],
                    const.side_root_chord * vstab_geo['scale_side'],
                    [0.17 + 0.05, wing_geo['wingspan']/4, -const.side_span * vstab_geo['scale_side']],
                    const.side_tip_chord * vstab_geo['scale_side'],
                    "rudder_left", rudder_chord_ratio),
                ),
            # Right Vertical Fin (down)
            asb.Wing(
                name="Right Fin Down",
                symmetric=False,
                xsecs=_fin_cross_section_xsecs(
                    [0.17, -wing_geo['wingspan']/4, 0],
                    const.side_root_chord * vstab_geo['scale_side'],
                    [0.17 + 0.05, -wing_geo['wingspan']/4, -const.side_span * vstab_geo['scale_side']],
                    const.side_tip_chord * vstab_geo['scale_side'],
                    "rudder_right", rudder_chord_ratio),
            ),
        ],
        fuselages=[
            asb.Fuselage(
                name="Fuselage",
                xsecs=[
                    asb.FuselageXSec(xyz_c=[-0.41, 0, 0], width=0, height=0),
                    asb.FuselageXSec(xyz_c=[-0.4, 0, 0], width=0.02, height=0.01),
                    asb.FuselageXSec(xyz_c=[-0.35, 0, 0], width=0.09, height=0.045),
                    asb.FuselageXSec(xyz_c=[-0.3, 0, 0], width=0.12, height=0.07),
                    asb.FuselageXSec(xyz_c=[-0.2, 0, 0], width=0.17, height=0.12),
                    asb.FuselageXSec(xyz_c=[-0.1, 0, 0], width=0.2, height=0.15),
                    asb.FuselageXSec(xyz_c=[0.0, 0, 0], width=0.2, height=0.15),
                    asb.FuselageXSec(xyz_c=[0.1, 0, 0], width=0.2, height=0.15),
                    asb.FuselageXSec(xyz_c=[0.2, 0, 0], width=0.17, height=0.12),
                    asb.FuselageXSec(xyz_c=[0.3, 0, 0], width=0.1, height=0.05),
                ],
            ),
            asb.Fuselage(
                name="Engine 1",
                xsecs=[
                    # For future. Create little offset for first point if mesh will degenerate
                    asb.FuselageXSec(xyz_c=[0.002, wing_geo['wingspan']/4, 0], width = 0, height= 0),
                    asb.FuselageXSec(xyz_c=[0.002, wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.07, wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.14, wing_geo['wingspan']/4, 0], width = 0, height= 0),
                ]
            ),
            asb.Fuselage(
                name="Engine 2",
                xsecs=[
                    # For future. Create little offset for first point if mesh will degenerate
                    asb.FuselageXSec(xyz_c=[0.002, -wing_geo['wingspan']/4, 0], width = 0, height= 0),
                    asb.FuselageXSec(xyz_c=[0.002, -wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.07, -wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.14, -wing_geo['wingspan']/4, 0], width = 0, height= 0),
                ]
            )
        ]
    )
    return airplane

def plot_airplane_views(airplane):
    """Plot three-view drawing of airplane."""
    airplane.draw_three_view()

def _main_wing_cross_section_xsecs(wing_geo, airfoil, tip_offset, root_offset, controls=None):
    root_section, tip_section, half_section_lenght = wing_geo['root_chord'], wing_geo['tip_chord'], wing_geo['wingspan']/2
    chord = lambda y: root_section+(tip_section - root_section)*y/half_section_lenght
    twist_section_y = [0.0, root_offset, half_section_lenght-tip_offset, half_section_lenght]
    twist_angel = [2.5,2.5,1,1]

    station_of_sections = [0,root_offset,half_section_lenght-tip_offset,half_section_lenght]
    if controls is not None:
        station_of_sections+=[controls['y_inboard'], controls['y_outboard']]
    station_of_sections = sorted(station_of_sections)
    final_station_of_sections=[station_of_sections[0]]
    for y in station_of_sections[1:]:
        if y-final_station_of_sections[-1]>1e-6:
            final_station_of_sections.append(y)

    xsecs=[] #Creating variable for cross-section of wing
    for y in final_station_of_sections:
        control_surfaces = []
        if controls is not None and controls['y_inboard'] - 1e-6 <= y < controls['y_outboard'] - 1e-6:
            hinge = 1 - controls['cf_c']
            control_surfaces = [asb.ControlSurface(name="elevon_sym", symmetric=True, hinge_point=hinge),
                                asb.ControlSurface(name="elevon_asym", symmetric=False, hinge_point=hinge)]
        xsecs.append(asb.WingXSec(
            xyz_le=[root_section - chord(y), y, 0],
            chord=chord(y),
            twist=float(np.interp(y, twist_section_y, twist_angel)),
            airfoil=airfoil,
            control_surfaces=control_surfaces
        ))

    return xsecs

def _fin_cross_section_xsecs(xyz_root, root_chord, xyz_tip, tip_chord, rudder_name, rudder_chord_ratio):
    fin_control_surfaces = []
    if rudder_chord_ratio is not None:
        fin_control_surfaces = [asb.ControlSurface(name=rudder_name, symmetric=True, hinge_point=1-rudder_chord_ratio)]
    return [
        asb.WingXSec(xyz_le=xyz_root, chord=root_chord, twist=0, airfoil=asb.Airfoil("naca0012"), control_surfaces=fin_control_surfaces),
        asb.WingXSec(xyz_le=xyz_tip, chord=tip_chord, twist=0, airfoil=asb.Airfoil("naca0012"))
    ]

def build_airplane_model_old(const, wing_geo, vstab_geo, profile_1, profile_2):
    """Build and return Aerosandbox airplane model."""
    a = 0.05 # Wingtip offset (m)
    b = 0.15 # Wingroot offset (m)
    airplane = asb.Airplane(
        name="Hybrid VTOL Fixed-Wing UAV",
        xyz_ref=[wing_geo['root_chord'] * 0.25, 0, 0],
        s_ref=wing_geo['S_wing'],
        c_ref=wing_geo['MAC'],
        b_ref=wing_geo['wingspan'],
        #b_ref=const.AR,
        wings=[
            # Main Wing
            asb.Wing(
                name="Main Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=wing_geo['root_chord'],
                        twist=2.5,
                        #twist=3.5,
                        airfoil=profile_1.to_airfoil(),
                    ),
                    asb.WingXSec(
                        # Using a complex formula to keep the wing trapezoidal regardless of the offset "b"
                        xyz_le=[wing_geo['root_chord'] - (wing_geo['tip_chord'] + (((wing_geo['wingspan']/2) - b) / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord'])), b, 0],
                        chord=wing_geo['tip_chord'] + (((wing_geo['wingspan']/2) - b) / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord']),
                        twist=2.5,
                        #twist=3.5,
                        airfoil=profile_1.to_airfoil(),
                    ),
                    asb.WingXSec(
                        # Using a complex formula to keep the wing trapezoidal regardless of the offset "a"
                        xyz_le=[wing_geo['root_chord'] - (wing_geo['tip_chord'] + (a / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord'])), wing_geo['wingspan']/2 - a, 0],
                        chord=wing_geo['tip_chord'] + (a / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord']),
                        twist=1,
                        #twist=1.5,
                        airfoil=profile_1.to_airfoil(),
                    ),
                    asb.WingXSec(
                        xyz_le=[wing_geo['root_chord'] - wing_geo['tip_chord'], wing_geo['wingspan']/2, 0],
                        chord=wing_geo['tip_chord'],
                        twist=1,
                        #twist=1.5,
                        airfoil=profile_1.to_airfoil(),
                    ),
                ],
            ),
            # Central Vertical Fin (up)
            asb.Wing(
                name="Central Fin Up",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0.17, 0, 0],
                        chord=const.central_root_chord * vstab_geo['scale_central'],
                        twist=0,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                    asb.WingXSec(
                        xyz_le=[0.17 + 0.05, 0, const.central_span * vstab_geo['scale_central']],
                        chord=const.central_tip_chord * vstab_geo['scale_central'],
                        twist=0,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                ],
            ),
            # Left Vertical Fin (down)
            asb.Wing(
                name="Left Fin Down",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0.17, wing_geo['wingspan']/4, 0],
                        chord=const.side_root_chord * vstab_geo['scale_side'],
                        twist=0,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                    asb.WingXSec(
                        xyz_le=[0.17 + 0.05, wing_geo['wingspan']/4, -const.side_span * vstab_geo['scale_side']],
                        chord=const.side_tip_chord * vstab_geo['scale_side'],
                        twist=0,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                ],
            ),
            # Right Vertical Fin (down)
            asb.Wing(
                name="Right Fin Down",
                symmetric=False,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0.17, -wing_geo['wingspan']/4, 0],
                        chord=const.side_root_chord * vstab_geo['scale_side'],
                        twist=0,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                    asb.WingXSec(
                        xyz_le=[0.17 + 0.05, -wing_geo['wingspan']/4, -const.side_span * vstab_geo['scale_side']],
                        chord=const.side_tip_chord * vstab_geo['scale_side'],
                        twist=0,
                        airfoil=asb.Airfoil("naca0012"),
                    ),
                ],
            ),
        ],
        fuselages=[
            asb.Fuselage(
                name="Fuselage",
                xsecs=[
                    asb.FuselageXSec(xyz_c=[-0.41, 0, 0], width=0, height=0),
                    asb.FuselageXSec(xyz_c=[-0.4, 0, 0], width=0.02, height=0.01),
                    asb.FuselageXSec(xyz_c=[-0.35, 0, 0], width=0.09, height=0.045),
                    asb.FuselageXSec(xyz_c=[-0.3, 0, 0], width=0.12, height=0.07),
                    asb.FuselageXSec(xyz_c=[-0.2, 0, 0], width=0.17, height=0.12),
                    asb.FuselageXSec(xyz_c=[-0.1, 0, 0], width=0.2, height=0.15),
                    asb.FuselageXSec(xyz_c=[0.0, 0, 0], width=0.2, height=0.15),
                    asb.FuselageXSec(xyz_c=[0.1, 0, 0], width=0.2, height=0.15),
                    asb.FuselageXSec(xyz_c=[0.2, 0, 0], width=0.17, height=0.12),
                    asb.FuselageXSec(xyz_c=[0.3, 0, 0], width=0.1, height=0.05),
                ],
            ),
            asb.Fuselage(
                name="Engine 1",
                xsecs=[
                    # For future. Create little offset for first point if mesh will degenerate
                    asb.FuselageXSec(xyz_c=[0.002, wing_geo['wingspan']/4, 0], width = 0, height= 0),
                    asb.FuselageXSec(xyz_c=[0.002, wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.07, wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.14, wing_geo['wingspan']/4, 0], width = 0, height= 0),
                ]
            ),
            asb.Fuselage(
                name="Engine 2",
                xsecs=[
                    # For future. Create little offset for first point if mesh will degenerate
                    asb.FuselageXSec(xyz_c=[0.002, -wing_geo['wingspan']/4, 0], width = 0, height= 0),
                    asb.FuselageXSec(xyz_c=[0.002, -wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.07, -wing_geo['wingspan']/4, 0], width = 0.0556, height= 0.0556),
                    asb.FuselageXSec(xyz_c=[0.14, -wing_geo['wingspan']/4, 0], width = 0, height= 0),
                ]
            )
        ],
    )
    return airplane