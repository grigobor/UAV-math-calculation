import numpy as np

def calculate_vertical_stabilizer(wingspan, S_wing, const):
    """Calculate vertical stabilizer dimensions and scaling."""
    vertical_static_stab_moment = 0.001 * wingspan + 0.03
    vertical_shoulder_stab = -0.002 * wingspan + 0.3
    vertical_relative_stab_area = vertical_static_stab_moment / vertical_shoulder_stab
    vertical_stab_area = vertical_relative_stab_area * S_wing
    vertical_stab_span = np.sqrt(vertical_stab_area * const.vertical_stab_aspect_ratio)
    
    # Distribute area between central and side fins
    S_vt_central = vertical_stab_area * const.fraction_central
    S_vt_side = vertical_stab_area * (1 - const.fraction_central) / 2
    
    # Calculate scaling factors
    central_area_base = (const.central_root_chord + const.central_tip_chord) / 2 * const.central_span
    scale_central = np.sqrt(S_vt_central / central_area_base)
    
    side_area_base = (const.side_root_chord + const.side_tip_chord) / 2 * const.side_span
    scale_side = np.sqrt(S_vt_side / side_area_base)
    
    return {
        "total_area": vertical_stab_area,
        "span": vertical_stab_span,
        "S_central": S_vt_central,
        "S_side": S_vt_side,
        "scale_central": scale_central,
        "scale_side": scale_side,
        "arm": vertical_shoulder_stab
    }