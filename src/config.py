from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PIPE = {'t0': 10368000, 'dt': 0.25,'size': 10 * 3600 / 0.25, 't_inj': 5 * 3600}
SPECS = {'low_mass': {'trange':(-3, 3), 'frange':(1e-3, 1e-1), 
                      'Qvals':np.linspace(29, 31, 5), 'Q':30,
                      'drange':np.logspace(3, 4, 5)},
         'high_mass': {'trange':(-3, 3), 'frange':(1e-3, 1e-1), 
                      'Qvals':np.linspace(15, 17, 5), 'Q':16,
                      'drange':np.logspace(3, 5, 5)},
         'ext_high_mass': {'trange':(-3, 3), 'frange':(1e-4, 1e-2), 
                       'Qvals':np.linspace(5, 7, 5), 'Q':6,
                       'drange':np.logspace(4, 5, 5)}}

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
glitch_lm_dataset_path = str(GLITCH_DATASETS / 'glitch_lm_dataset.h5')
glitch_hm_dataset_path = str(GLITCH_DATASETS / 'glitch_hm_dataset.h5')
glitch_ehm_dataset_path = str(GLITCH_DATASETS / 'glitch_ehm_dataset.h5')

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
empty_lm_dataset_path = str(EMPTY_DATASETS / 'empty_lm_dataset.h5')
empty_hm_dataset_path = str(EMPTY_DATASETS / 'empty_hm_dataset.h5')
empty_ehm_dataset_path = str(EMPTY_DATASETS / 'empty_ehm_dataset.h5')