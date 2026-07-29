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

def create_airfoil():
    
    # Set up the Optimization environment
    opti = asb.Opti()

    CL_multipoint_targets = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    CL_multipoint_weights = np.array([3, 4, 5, 6, 7])

    # Starting point is NACA0012
    initial_airfoil = asb.KulfanAirfoil("naca0012")

    # Define upper line of airfoil parameters
    upper_weights = opti.variable(
        init_guess=initial_airfoil.upper_weights, 
        lower_bound=-0.5, 
        upper_bound=0.25
        )
    
    # Define lower line of airfoil parameters
    lower_weights = opti.variable(
        init_guess=initial_airfoil.lower_weights, 
        lower_bound=-0.25, 
        upper_bound=0.5
        )

    # Leading and trailind edges have to be equal for symmetrical and asymmetrical airfoils.
    opti.subject_to(upper_weights[0:2] == -lower_weights[0:2])
    opti.subject_to(upper_weights[-1] == lower_weights[-1])

    # Choosing characteristics for my asymmetric airfoil (for better CL at cruise)
    asymetrical_optimized_airfoil = asb.KulfanAirfoil(
        name="Asymmetric Cruise Airfoil",
        upper_weights=upper_weights,
        lower_weights=lower_weights,
        leading_edge_weight=opti.variable(init_guess=0, lower_bound=-0.05, upper_bound=0.05),
        TE_thickness=0,
    )

    # Alpha is the angle of attack, which can be also be optimized.
    alpha = opti.variable(
        init_guess=np.degrees(CL_multipoint_targets / (2 * np.pi)), # Estimated angle of attack, start of countdown
        lower_bound=-5, 
        upper_bound=15,
    )

    # Use NeuralFoil to get Aerodynamics
    aero = asymetrical_optimized_airfoil.get_aero_from_neuralfoil(
        alpha=alpha,
        Re=inputs.DesignConstants.Re_cruise,
        mach=inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound,
        model_size="large" # "large" or "xlarge" for better accuracy
    )

    # Setting additional restrictions on the airfoil shape.
    opti.subject_to(
        [
            aero["CL"] >= CL_multipoint_targets,
            aero["CM"] >= 0.015,
            asymetrical_optimized_airfoil.local_thickness(x_over_c=0.005) >= 0.01,
            asymetrical_optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.05, 
            asymetrical_optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
            #asymetrical_optimized_airfoil.max_thickness() >= 0.10,
            asymetrical_optimized_airfoil.local_thickness() <= 0.12,
            asymetrical_optimized_airfoil.local_thickness() > 0,
        ]
    )

    # Prevention of micro-rippling
    get_wiggliness = lambda af: sum( 
        [
            np.sum(np.diff(np.diff(array)) ** 2)
            for array in [af.lower_weights, af.upper_weights]
        ]
    )
    opti.subject_to(
        get_wiggliness(asymetrical_optimized_airfoil) < 2 * get_wiggliness(initial_airfoil),
    )

    opti.minimize(np.mean(aero["CD"] * CL_multipoint_weights))

    # Solve
    try:
        sol = opti.solve()

        opt_upper = sol.value(upper_weights)
        opt_lower = sol.value(lower_weights)
        opt_le = sol.value(asymetrical_optimized_airfoil.leading_edge_weight)

        sym_weights = (opt_upper - opt_lower) / 2

        symmetric_airfoil = asb.KulfanAirfoil(
            name="Symmetric State",
            upper_weights=sym_weights,
            lower_weights=-sym_weights,
            leading_edge_weight=opt_le,
            TE_thickness=0
        )

        #Building comparison plot
        plot_comparison(sol.value(asymetrical_optimized_airfoil), symmetric_airfoil, aoa=sol.value(alpha))
        airfoil_characteristics(sol.value(asymetrical_optimized_airfoil), symmetric_airfoil, aoa=sol.value(alpha))

        #Returning both airfoils coordinates for futher use in the wing design
        return sol.value(asymetrical_optimized_airfoil), symmetric_airfoil, sol.value(alpha), sol.value(aero["CL"]), sol.value(aero["CD"]), sol.value(aero["CM"])
        
    except RuntimeError:
        print("Optimization failed! Plotting the 'broken' airfoil for debug...")
        # This pulls the last known values from the solver's memory
        failed_af = opti.debug.value(asymetrical_optimized_airfoil)
        failed_af.draw()
        raise # Still stop the code so you can read the error

def plot_comparison(af_asym, af_sym, aoa):
    airfoils_and_colors = {
        "Asymmetric (Cruise) Aurfoil":(af_asym, 'blue'),
        "Symmetric (Take-off) Airfoil":(af_sym, 'red'),
        "NACA0012 (Baseline)":(asb.KulfanAirfoil("naca0012"), 'green'),
        "MH-60 (Reference)":(asb.KulfanAirfoil("mh60"), 'orange')
    }

    Re_plot = inputs.DesignConstants.Re_cruise
    mach_plot = inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound

    fig, ax = plt.subplots(2, 1, figsize=(8, 8))

    for i, (name, (af, color)) in enumerate(airfoils_and_colors.items()):
        ax[0].fill(
            af.x(),
            af.y(),
            facecolor=(*p.adjust_lightness(color, 1.5), 0.1),
            edgecolor=color,
            linewidth=2,
            label=name,
            linestyle='-' if "Cruise" in name else '--',
            zorder=4 if "Cruise" in name else 3,
        )
        alphas = np.linspace(0, 12, 50)
        aero = af.get_aero_from_neuralfoil(
            alpha=alphas,
            Re=Re_plot,
            mach=mach_plot,
        )
        ax[1].plot(
            aero["CD"],
            aero["CL"],
            color=color,
            # linewidth=2,
            label=name,
            alpha=0.8
        )

    # fig.suptitle("Adaptive Wing Morphing: Performance & Stability Analysis", fontsize=15, fontweight='bold', y=0.98)

    ax[0].legend(fontsize=11, loc="lower center", ncol=len(airfoils_and_colors) // 2)
    ax[0].set_title("I. Airfoil Profile Geometry", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("$x/c$")
    ax[0].set_ylabel("$y/c$")
    ax[0].axis("equal")
    # ax[0].grid(True, alpha=0.3)

    ax[1].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[1].set_title(f"II. Aerodynamic Polar ($Re={Re_plot/1e3:.0f}k$)", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Drag Coefficient $C_D$")
    ax[1].set_ylabel("Lift Coefficient $C_L$")
    ax[1].set_xlim(0, 0.035)
    ax[1].set_ylim(-0.2, 1.6)
    #ax[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    #plt.tight_layout()
    p.show_plot(
        #"Comparison of Adaptive Wing States",
        #show=True
        title=None,
        legend=False
    )

def airfoil_characteristics(af_asym, af_sym, aoa):
    airfoils_and_colors = {
        "Asymmetric (Cruise) Aurfoil":(af_asym, 'blue'),
        "Symmetric (Take-off) Airfoil":(af_sym, 'red')
    }

    Re_plot = inputs.DesignConstants.Re_cruise
    mach_plot = inputs.DesignConstants.cruise_speed_ms / inputs.DesignConstants.cruise_speed_of_sound

    fig, ax = plt.subplots(3, 1, figsize=(8, 12))

    for i, (name, (af, color)) in enumerate(airfoils_and_colors.items()):
        alphas = np.linspace(-20, 20, 100)
        aero = af.get_aero_from_neuralfoil(
            alpha=alphas,
            Re=Re_plot,
            mach=mach_plot,
        )
        ax[0].plot(
            alphas,
            aero["CL"],
            color=color,
            # linewidth=2,
            label=name,
            alpha=0.8
        )
        ax[1].plot(
            alphas,
            aero["CD"],
            color=color,
            label=name,
            alpha=0.8,
            linewidth=2
        )
        ax[2].plot(
            alphas,
            aero["CM"],
            color=color,
            label=name,
            alpha=0.8
        )

    # fig.suptitle("Adaptive Wing Morphing: Performance & Stability Analysis", fontsize=15, fontweight='bold', y=0.98)

    ax[0].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[0].set_title(f"I. Lift Coefficient vs. Alpha ($Re={Re_plot/1e3:.0f}k$)", fontsize=12, fontweight='bold')
    ax[0].set_xlabel("Angle of Attack $\\alpha$ (degrees)")
    ax[0].set_ylabel("Lift Coefficient $C_L$")
    ax[0].grid(True, alpha=0.3)
    ax[0].set_ylim(-1.8, 1.8)
    #ax[1].grid(True, alpha=0.3)

    ax[1].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[1].set_title("II. Drag Coefficient vs. Alpha", fontsize=12, fontweight='bold')
    ax[1].set_xlabel("Angle of Attack $\\alpha$ (degrees)")
    ax[1].set_ylabel("Drag Coefficient $C_D$")
    ax[1].grid(True, alpha=0.3)
    ax[1].set_ylim(0, 0.15)

    ax[2].legend(fontsize=11, loc="lower right", ncol=len(airfoils_and_colors) // 2)
    ax[2].set_title("III. Pitching Moment vs. Alpha", fontsize=12, fontweight='bold')
    ax[2].set_xlabel("Angle of Attack $\\alpha$ (degrees)")
    ax[2].set_ylabel("Pitching Moment Coefficient $C_m$")
    ax[2].grid(True, alpha=0.3)
    ax[2].set_ylim(-0.25, 0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    #plt.tight_layout()
    p.show_plot(
        #"Comparison of Adaptive Wing States",
        #show=True
        title=None,
        legend=False
    )