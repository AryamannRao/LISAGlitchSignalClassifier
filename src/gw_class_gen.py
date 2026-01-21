import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import time
import shutil

from simulation import run_simulation, run_tdi
from qtransform import varq_transform
from h5file_helpers import create_dataset, append_gw_sample
from config import *

from concurrent.futures import ProcessPoolExecutor, as_completed

import warnings
warnings.filterwarnings("ignore")

psd_data = np.load(psd_path)
F_PSD = psd_data["f"]
PSD = psd_data["psd"]

def get_mass_specific_params(m1, m2, specs):
    if m1 < 1e5 or m2 < 1e5:
        spec = specs['low_mass']
    elif m1 > 1e6 or m2 > 1e6:
        spec = specs['high_mass']
    else:
        spec = specs['mid_mass']
    return spec

def run_one_sample(args):
    m1, m2, d, spin, spec, pipe = args

    gws = [{
        'type': 'BinaryInspiralGW',
        'm1': m1,
        'm2': m2,
        'd': d,
        'spin1': spin,
        'spin2': spin,
        't_inj': spec['t_inj'] * 3600,
        'domain': 'freq'
    }]

    glitches = [{'type':'OneSidedDoubleExpGlitch', 't_inj': 0,
             'level':0, 't_rise':1, 't_fall':1, 
             'inj_point':'readout_tmi_carrier_12'}]


    pid = os.getpid()
    local_sim_path = f"{simulation_path}_{pid}.h5"
    local_gw_path     = f"{gw_path}_{pid}.h5"
    local_glitch_path = f"{glitch_path}_{pid}.h5"

    run_simulation(
        gws, glitches, pipe,
        local_gw_path, local_glitch_path, orbits_path,
        local_sim_path,
        disable_noise=False
    )

    tdi_dict = run_tdi(local_sim_path, pipe, F_PSD, PSD)

    event = gws[0]
    _, _, varQT = varq_transform(
        tdi_dict, 'X', pipe, event,
        resolution=512,
        frange=spec['frange'],
        trange=spec['trange'],
        Qvals=spec['Qvals']
    )

    os.remove(local_sim_path)
    os.remove(local_gw_path)
    os.remove(local_glitch_path)

    return varQT, m1, m2, d, spin

def main():

    start = time.time()

    mass_arr = [1e6, 1e7]
    q_arr = [1]
    spin_arr = [0.0]

    specs = {'low_mass': {'trange':(-5, 1), 'frange':(1e-3, 1e-1), 
                      'Qvals':np.linspace(29, 31, 5), 
                      'drange':np.logspace(3, 4, 5), 'size':10, 't_inj':7},
         'mid_mass': {'trange':(-5, 1), 'frange':(1e-3, 1e-1), 
                      'Qvals':np.linspace(15, 17, 5), 
                      'drange':np.logspace(3, 5, 5), 'size':10, 't_inj':7},
         'high_mass': {'trange':(-5, 2), 'frange':(1e-4, 1e-2), 
                       'Qvals':np.linspace(5, 7, 5), 
                       'drange':np.logspace(4, 5, 5), 'size':10, 't_inj':7}}

    # Prepare job list
    jobs = []

    for m in mass_arr:
        for q in q_arr:
            m1, m2 = m, m * q
            spec = get_mass_specific_params(m1, m2, specs)

            pipe = {
                't0': 10368000,
                'dt': 0.25,
                'size': spec['size'] * 3600 / 0.25,
                'gw_beta': 0,
                'gw_lambda': np.pi / 7
            }

            for d in spec['drange']:
                for spin in spin_arr:
                    jobs.append((m1, m2, d, spin, spec, pipe))

    # Create dataset
    if os.path.exists(gw_dataset_path):
        os.remove(gw_dataset_path)

    h5file = create_dataset(gw_dataset_path, resolution=(512, 512))

    n_workers = 6
    print(f"Running with {n_workers} workers")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_one_sample, job) for job in jobs]

        for future in as_completed(futures):
            varQT, m1, m2, d, spin = future.result()
            append_gw_sample(h5file, varQT, m1, m2, d, spin)

    h5file.close()

    end = time.time()
    print(f"Total time taken: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()