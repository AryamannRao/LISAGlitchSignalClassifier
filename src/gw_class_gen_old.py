import numpy as np
import os
import time

from simulation import run_simulation, run_tdi
from qtransform import varq_transform, generate_qscan
from h5file_helpers import create_dataset, append_gw_sample
from config import *

import warnings
warnings.filterwarnings("ignore")

def get_mass_specific_params(m1, m2, specs):
    if m1 < 1e5 or m2 < 1e5:
        spec = specs['low_mass']
    elif m1 > 1e6 or m2 > 1e6:
        spec = specs['high_mass']
    else:
        spec = specs['mid_mass']
    return spec

# DATASET GENERATION PARAMETERS
glitches = [{'type':'OneSidedDoubleExpGlitch', 't_inj': 0,
             'level':0, 't_rise':1, 't_fall':1, 'inj_point':'readout_tmi_carrier_12'}]

specs = {'low_mass': {'trange':(-5, 1), 'frange':(1e-3, 1e-1), 
                      'Qvals':np.linspace(29, 31, 5), 'drange':np.logspace(3, 4, 5), 'size':10, 't_inj':7},
         'mid_mass': {'trange':(-5, 1), 'frange':(1e-3, 1e-1), 
                      'Qvals':np.linspace(15, 17, 5), 'drange':np.logspace(3, 5, 5), 'size':10, 't_inj':7},
         'high_mass': {'trange':(-5, 2), 'frange':(1e-4, 1e-2), 
                       'Qvals':np.linspace(5, 7, 5), 'drange':np.logspace(4, 5, 5), 'size':10, 't_inj':7}}

mass_arr = [1e7]
q_arr = [1]
spin_arr = [0.0]

res = 512
H, W = res, res

# DATASET GENERATION
psd_data = np.load(psd_path)
f_psd, psd = psd_data["f"], psd_data["psd"]

if os.path.exists(gw_dataset_path):
    os.remove(gw_dataset_path)
    
h5file = create_dataset(gw_dataset_path, resolution=(H, W))

count = 0
start = time.time()
for m in mass_arr:
    for q in q_arr:
        # Determine mass specific parameters
        m1, m2 = m, m*q
        spec = get_mass_specific_params(m1, m2, specs)
        d_arr = spec['drange']

        pipe = {'t0':10368000, 'dt':0.25, 'size':spec['size']*3600/0.25, 'gw_beta':0, 'gw_lambda':np.pi/7}
        for d in d_arr:
            for spin in spin_arr:

                gws = [{'type':'BinaryInspiralGW', 'm1':m1, 'm2':m2, 'd':d, 'spin1':spin, 'spin2':spin,
                't_inj':spec['t_inj']*3600, 'domain':'freq'}]

                run_simulation(gws, glitches, pipe,
                            gw_path, glitch_path, orbits_path, simulation_path, disable_noise=False,)
                tdi_dict = run_tdi(simulation_path, pipe, f_psd, psd)

                event = gws[0]
                t_arr, f_arr, varQT = varq_transform(tdi_dict, 'X', pipe, event, resolution=res,
                             frange=spec['frange'], trange=spec['trange'], Qvals=spec['Qvals'])
                append_gw_sample(h5file, varQT, event['m1'], event['m2'], event['d'], event['spin1'])
end = time.time()
print(f"Total time taken: {end - start} seconds")