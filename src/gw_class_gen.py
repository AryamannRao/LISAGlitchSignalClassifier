import numpy as np
import h5py
from simulation import run_simulation, run_tdi
from qtransform import generate_qscan

orbits_path = '../dist/orbits.h5'
simulation_path = '../dist/default_simulation_output.h5'
tdi_path = '../dist/default_tdi_output.h5'
gw_path = '../dist/default_gw_output.h5'
glitch_path = '../dist/default_glitch_output.h5'

pipe = {'t0':10368000, 'dt':0.25, 'size':72000, 'gw_beta':0, 'gw_lambda':np.pi/7}

glitches = [{'type':'OneSidedDoubleExpGlitch', 't_inj': 0,
             'level':0, 't_rise':1, 't_fall':1, 'inj_point':'readout_tmi_carrier_12'}]

mass_arr = [1e5]
q_arr = [1, 10]

for m in mass_arr:
    for q in q_arr:

        gws = [{'type':'BinaryInspiralGW', 'm1':m, 'm2':m*q, 'd':1e4, 
        'e':0.0, 't_inj':2.5*3600, 'domain':'freq'}]

        run_simulation(gws, glitches, pipe,
                    gw_path, glitch_path, orbits_path, simulation_path, disable_noise=False,)
        tdi_dict = run_tdi(simulation_path, pipe)

        event = gws[0]
        t_arr, f_arr, QT = generate_qscan(tdi_dict, Q=15, channel='X', pipe=pipe, event=event, resolution=512, 
                        frange=(1e-3, 1e-1), trange=(-1, 1))