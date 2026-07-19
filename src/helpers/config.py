from pathlib import Path
import numpy as np
import h5py

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

SRC = PROJECT_ROOT / 'src'
DIST = PROJECT_ROOT / 'dist'
TRAINING_DATASETS = DIST / 'training_datasets'
TRAINING_RESULTS = DIST / 'training_results'
INFERENCE_DATASETS = DIST / 'inference_datasets'
PSD_PATH = DIST / 'psd_estimates'
FIGURES = DIST / 'figures'

GW_DATASETS = TRAINING_DATASETS / 'gw_datasets'
GLITCH_DATASETS = TRAINING_DATASETS / 'glitch_datasets'
MIXED_DATASETS = TRAINING_DATASETS / 'mixed_datasets'
EMPTY_DATASETS = TRAINING_DATASETS / 'empty_datasets'
LPF_DATASETS = INFERENCE_DATASETS / 'lpf_datasets'

orbits_path = str(DIST/ 'orbits' / 'orbits.h5')
simulation_path = str(DIST / 'default_simulation_output.h5')
glitch_path = str(DIST / 'default_glitch_output.h5')
gw_path = str(DIST / 'default_gw_output.h5')

lpf_library_path = str(LPF_DATASETS / 'lpf-glitch-library.h5')
lpf_ord_param_path = str(LPF_DATASETS / 'glitch_params_ordinary.txt')
lpf_cold_param_path = str(LPF_DATASETS / 'glitch_params_cold.txt')

glitch_dataset_path = str(GLITCH_DATASETS / 'glitch_dataset.h5')
gw_dataset_path = str(GW_DATASETS / 'gw_dataset.h5')
mixed_dataset_path = str(MIXED_DATASETS / 'mixed_dataset.h5')
empty_dataset_path = str(EMPTY_DATASETS / 'empty_dataset.h5')
training_dataset_path = str(TRAINING_DATASETS / 'training_dataset.h5')

glitch_testset_path = str(GLITCH_DATASETS / 'glitch_testset.h5')
gw_testset_path = str(GW_DATASETS / 'gw_testset.h5')
mixed_testset_path = str(MIXED_DATASETS / 'mixed_testset.h5')
empty_testset_path = str(EMPTY_DATASETS / 'empty_testset.h5')

with h5py.File(orbits_path, 'r') as orb:
    orb_t0 = orb.attrs['t0']
    orb_size = orb.attrs['size']
    orb_dt = orb.attrs['dt']

PIPE = {'t0': orb_t0 + orb_size*orb_dt/2, 'dt': 0.25,'size': int(10 * 3600 / 0.25),
        't_inj': 5 * 3600, 'keep_noises': ['test-mass', 'oms']}
SPECS = {'q16': {'trange':(-3, 3), 'frange':(1e-4, 1e-1), 'Q':16},
         'q6': {'trange':(-3, 3), 'frange':(1e-4, 1e-1), 'Q':6}}