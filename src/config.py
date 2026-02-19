from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SRC = PROJECT_ROOT / 'src'
DIST = PROJECT_ROOT / 'dist'
GW_DATASETS = DIST / 'gw_datasets'
GLITCH_DATASETS = DIST / 'glitch_datasets'
MIXED_DATASETS = DIST / 'mixed_datasets'
EMPTY_DATASETS = DIST / 'empty_datasets'
PSD_PATH = DIST / 'psd_estimates'

orbits_path = str(DIST / 'orbits.h5')
simulation_path = str(DIST / 'default_simulation_output.h5')

glitch_path = str(DIST / 'default_glitch_output.h5')
glitch_dataset_path = str(GLITCH_DATASETS / 'glitch_dataset.h5')

gw_path = str(DIST / 'default_gw_output.h5')
gw_dataset_path = str(GW_DATASETS / 'gw_dataset.h5')
gw_lm_dataset_path = str(GW_DATASETS / 'gw_lm_dataset.h5')
gw_hm_dataset_path = str(GW_DATASETS / 'gw_hm_dataset.h5')
gw_ehm_dataset_path = str(GW_DATASETS / 'gw_ehm_dataset.h5')

mixed_dataset_path = str(MIXED_DATASETS / 'mixed_dataset.h5')
mixed_lm_dataset_path = str(MIXED_DATASETS / 'mixed_lm_dataset.h5')
mixed_hm_dataset_path = str(MIXED_DATASETS / 'mixed_hm_dataset.h5')
mixed_ehm_dataset_path = str(MIXED_DATASETS / 'mixed_ehm_dataset.h5')

empty_dataset_path = str(EMPTY_DATASETS / 'empty_dataset.h5')