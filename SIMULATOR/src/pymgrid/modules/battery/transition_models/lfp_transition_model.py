import yaml

from .custom_transition_model import CustomChemistryTransitionModel


class LfpTransitionModel(CustomChemistryTransitionModel):
    """LFP chemistry model matching ESS_UNIPI_LFP Voc/R0 logic."""

    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader
    yaml_tag = u"!LfpTransitionModel"

    def __init__(self,
                 reference_cell_capacity_ah: float = 3.2,
                 nominal_cell_voltage: float = 3.7,
                 **kwargs):
        super().__init__(
            parameters_mat="parameters_cell_LFP.mat",
            reference_cell_capacity_ah=reference_cell_capacity_ah,
            nominal_cell_voltage=nominal_cell_voltage,
            **kwargs,
        )
