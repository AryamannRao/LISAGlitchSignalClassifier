import numpy as np
from scipy.signal import welch
from config import *
from simulation import *

def estimate_psd(data, sample_freq, nperseg, noverlap=None):
    
    f_psd, psd = welch(
        data,
        fs=sample_freq,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
        return_onesided=True
    )
    return f_psd, psd

def main():
    pipe = {'t0':10368000, 'dt':0.25, 'size':10*3600/0.25, 'gw_beta':0, 'gw_lambda':0}
    
    glitches = [{'type':'OneSidedDoubleExpGlitch', 't_inj': 0,
             'level':0, 't_rise':1, 't_fall':1, 'inj_point':'readout_tmi_carrier_12'}]
    gws = [{'type':'ReducedOneSidedDoubleExpGW', 't_inj': 0, 'amp':0, 't_rise':1, 't_fall':1}]
    
    run_simulation(gws, glitches, pipe,
                   gw_path, glitch_path, orbits_path,
                   simulation_path,
                   disable_noise=False)
    tdi_dict = run_tdi(simulation_path, pipe)

    X, Y, Z = tdi_dict['X'].value, tdi_dict['Y'].value, tdi_dict['Z'].value
    A = (Z - X)/np.sqrt(2)
    E = (X - 2*Y + Z)/np.sqrt(6)
    T = (X + Y + Z)/np.sqrt(3)

    channels = [X, Y, Z, A, E, T]
    channel_names = ['X', 'Y', 'Z', 'A', 'E', 'T']

    if os.path.exists(psd_path) == False:
        os.makedirs(psd_path)
    for i, channel in enumerate(channels):
        f_psd, psd = estimate_psd(channel, 1/pipe['dt'], nperseg=1e4/pipe['dt'], noverlap=None)
        np.savez(f'{psd_path}psd_{channel_names[i]}.npz', f_psd=f_psd, psd=psd)

if __name__ == "__main__":
    main()
