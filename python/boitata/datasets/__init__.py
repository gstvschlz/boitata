"""Example datasets, generated on the call from a seed; nothing is downloaded.

All of them are synthetic and come with the same data for the same seed, on any machine and thread count. Each covers
one kind of problem: `clustered_metal_survey` has a preferential sample with a known truth, `vein_gold_grade_control`
has extreme skew, and so on. Coordinates are local meters (`topsoil_metals`: km) with no CRS, except `seismic_cube`
(EPSG:23031).
"""

from boitata.datasets._deposits import (
    iron_formation_plateau,
    nickel_laterite_profile,
    phosphate_weathering_profile,
    porphyry_geometallurgy,
    stacked_sulphide_lenses,
    tailings_reprocessing,
    vein_gold_grade_control,
)
from boitata.datasets._maps import (
    channel_training_image,
    clustered_metal_survey,
    clustered_metal_survey_exhaustive,
    coal_seam_thickness,
    seismic_cube,
    soil_geochemistry_survey,
    topsoil_metals,
)

__all__ = [
    "channel_training_image",
    "clustered_metal_survey",
    "clustered_metal_survey_exhaustive",
    "coal_seam_thickness",
    "iron_formation_plateau",
    "nickel_laterite_profile",
    "phosphate_weathering_profile",
    "porphyry_geometallurgy",
    "seismic_cube",
    "soil_geochemistry_survey",
    "stacked_sulphide_lenses",
    "tailings_reprocessing",
    "topsoil_metals",
    "vein_gold_grade_control",
]
