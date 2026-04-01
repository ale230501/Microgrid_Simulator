import yaml

from .custom_transition_model import CustomChemistryTransitionModel


class NmcTransitionModel(CustomChemistryTransitionModel):
    """NMC chemistry model matching ESS_UNIPI_NMC Voc/R0 logic."""

    yaml_dumper = yaml.SafeDumper
    yaml_loader = yaml.SafeLoader
    yaml_tag = u"!NmcTransitionModel"

    def __init__(self,
                 reference_cell_capacity_ah: float = 3.2,
                 nominal_cell_voltage: float = 3.7,
                 wear_a: float = 1354.0,
                 wear_b: float = 1.614,
                 wear_c: float = 0.068,
                 wear_B: float = None,
                 **kwargs):
        super().__init__(
            parameters_mat="parameters_cell_NMC.mat",
            reference_cell_capacity_ah=reference_cell_capacity_ah,
            nominal_cell_voltage=nominal_cell_voltage,
            wear_a=wear_a,
            wear_b=wear_b,
            wear_c=wear_c,
            wear_B=wear_B,
            **kwargs,
        )
