"""
Trainer variants for the LLA (SLAug Local Location-scale Aug) on/off comparison.

Each variant bakes its condition into __init__ (via the same env vars that
run_training.py would set), so:
  * the condition is reproducible without remembering CLI flags, and
  * the trainer class name lands in the output folder
        nnUNet_results/<Dataset>/<ClassName>__<plans>__<config>/fold_<fold>
    -> the two `fold all` runs no longer overwrite each other.

__init__ runs AFTER run_training.py has set its env vars, so these settings
take precedence (you do NOT need to pass --slaug_lla on the command line).

Usage:
    nnUNetv2_train 002 3d_fullres all -p nnUNetResEncUNetXLPlans -tr nnUNetTrainer_noLLA
    nnUNetv2_train 002 3d_fullres all -p nnUNetResEncUNetXLPlans -tr nnUNetTrainer_LLA
"""

import os

import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_noLLA(nnUNetTrainer):
    """Baseline: LLA explicitly OFF (region + gaussian as usual)."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        os.environ['nnUNet_slaug_lla'] = '0'
        os.environ['nnUNet_slaug_gla'] = '0'
        super().__init__(plans, configuration, fold, dataset_json, device)


class nnUNetTrainer_LLA(nnUNetTrainer):
    """LLA ON (smooth-field variant, the default LLA), GLA off."""
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        os.environ['nnUNet_slaug_lla'] = '1'
        os.environ['nnUNet_slaug_lla_p'] = '0.2'
        os.environ['nnUNet_slaug_lla_mode'] = 'field'
        os.environ['nnUNet_slaug_gla'] = '0'
        super().__init__(plans, configuration, fold, dataset_json, device)
