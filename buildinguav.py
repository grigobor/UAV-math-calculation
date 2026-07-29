import aerosandbox as asb

def build_airplane_model(const, wing_geo, vstab_geo, profile_1, profile_2):
    """Build and return Aerosandbox airplane model."""
    a = 0.05 # Wingtip offset (m)
    b = 0.15 # Wingroot offset (m)
    airplane = asb.Airplane(
        name="Hybrid VTOL Fixed-Wing UAV",
        xyz_ref=[wing_geo['root_chord'] * 0.25, 0, 0],
        s_ref=wing_geo['S_wing'],
        c_ref=wing_geo['MAC'],
        b_ref=const.AR,
        wings=[
            # Main Wing
            asb.Wing(
                name="Main Wing",
                symmetric=True,
                xsecs=[
                    asb.WingXSec(
                        xyz_le=[0, 0, 0],
                        chord=wing_geo['root_chord'],
                        twist=3.5,
                        airfoil=profile_2.to_airfoil(),
                    ),
                    asb.WingXSec(
                        # Using a complex formula to keep the wing trapezoidal regardless of the offset "b"
                        xyz_le=[wing_geo['root_chord'] - (wing_geo['tip_chord'] + (((wing_geo['wingspan']/2) - b) / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord'])), b, 0],
                        chord=wing_geo['tip_chord'] + (((wing_geo['wingspan']/2) - b) / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord']),
                        twist=3.5,
                        airfoil=profile_1.to_airfoil(),
                    ),
                    asb.WingXSec(
                        # Using a complex formula to keep the wing trapezoidal regardless of the offset "a"
                        xyz_le=[wing_geo['root_chord'] - (wing_geo['tip_chord'] + (a / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord'])), wing_geo['wingspan']/2 - a, 0],
                        chord=wing_geo['tip_chord'] + (a / (wing_geo['wingspan']/2)) * (wing_geo['root_chord'] - wing_geo['tip_chord']),
                        twist=1.5,
                        airfoil=profile_1.to_airfoil(),
                    ),
                    asb.WingXSec(
                        xyz_le=[wing_geo['root_chord'] - wing_geo['tip_chord'], wing_geo['wingspan']/2, 0],
                        chord=wing_geo['tip_chord'],
                        twist=1.5,
                        airfoil=profile_2.to_airfoil(),
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
            )
        ],
    )
    return airplane

def plot_airplane_views(airplane):
    """Plot three-view drawing of airplane."""
    airplane.draw_three_view()