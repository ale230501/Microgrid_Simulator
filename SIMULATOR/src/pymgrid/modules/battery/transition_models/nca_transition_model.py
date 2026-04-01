import yaml

from .custom_transition_model import CustomChemistryTransitionModel


class NcaTransitionModel(CustomChemistryTransitionModel):
    """NCA chemistry model matching ESS_UNIPI_NCA Voc/R0 logic."""

    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader
    yaml_tag = u"!NcaTransitionModel"

    def __init__(self,
                 reference_cell_capacity_ah: float = 87.671,
                 nominal_cell_voltage: float = 3.65,
                 **kwargs):
        super().__init__(
            parameters_mat="parameters_cell_NCA.mat",
           # reference_cell_capacity_ah=100.0,
            reference_cell_capacity_ah=reference_cell_capacity_ah,
            nominal_cell_voltage=nominal_cell_voltage,
            **kwargs,
        )
