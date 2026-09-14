# Estimate and save noise power spectral densities for the TDI channels.
import sys
from pathlib import Path

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
from scipy.signal import welch
from helpers.config import *
from helpers.simulation import *

def estimate_psd(data, sample_freq, nperseg, noverlap=None):
    # Use Welch's method with a Hann window to estimate a one-sided PSD.
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
    # Simulate noise only, derive A/E/T channels, and save a PSD per channel.
    glitches, gws = [], []
    
    run_simulation(gws, glitches, PIPE,
                   None, None, orbits_path,
                   simulation_path,
                   disable_noise=PIPE['keep_noises'])
    tdi_dict = run_tdi(simulation_path, PIPE)

    # Read the first-generation TDI channels and form their orthogonal combinations.
    X, Y, Z = tdi_dict['X'].value, tdi_dict['Y'].value, tdi_dict['Z'].value
    A = (Z - X)/np.sqrt(2)
    E = (X - 2*Y + Z)/np.sqrt(6)
    T = (X + Y + Z)/np.sqrt(3)

    channels = [X, Y, Z, A, E, T]
    channel_names = ['X', 'Y', 'Z', 'A', 'E', 'T']

    # Create the output directory the first time PSDs are generated.
    if os.path.exists(PSD_PATH) == False:
        os.makedirs(PSD_PATH)
    for i, channel in enumerate(channels):
        f_psd, psd = estimate_psd(channel, 1/PIPE['dt'], nperseg=1e4/PIPE['dt'], noverlap=None)
        np.savez(PSD_PATH / f'psd_{channel_names[i]}.npz', f_psd=f_psd, psd=psd)

if __name__ == "__main__":
    main()
