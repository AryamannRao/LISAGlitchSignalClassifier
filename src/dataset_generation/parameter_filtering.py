import numpy as np
import h5py

import sys
from pathlib import Path

# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from helpers.simulation import *
from helpers.qtransform import *
from helpers.config import *

from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

from sklearn.ensemble import RandomForestClassifier
from joblib import dump

import warnings
warnings.filterwarnings("ignore")

TRANSIENT_TYPE = 'gw'
if TRANSIENT_TYPE == 'glitch':
    FILE_PATH = glitch_testset_path
    FOREST_PATH = glitch_forest_path
    TO_FIND = 1
elif TRANSIENT_TYPE == 'gw':
    FILE_PATH = gw_testset_path
    FOREST_PATH = gw_forest_path
    TO_FIND = 1
elif TRANSIENT_TYPE == 'mixed':
    FILE_PATH = mixed_testset_path
    FOREST_PATH = mixed_forest_path
    TO_FIND = 2

def filter_by_snr(dataset_path, to_find=1, threshold=8):
    with h5py.File(dataset_path, 'r') as h5:
        t_qscan = h5['t_qscan'][:]
        f_qscan = h5['f_qscan'][:]
        qt = h5['QT'][:]
        tcen = h5['tcen'][:]

    idx = np.argmin(np.abs(tcen), axis=1)
    images = qt[np.arange(qt.shape[0]), idx]
    t_axes = t_qscan[np.arange(qt.shape[0]), idx]
    f_axes = f_qscan[np.arange(qt.shape[0]), idx]
    snrs = np.sqrt(images**2 - 1)

    good_idx = []
    for index in range(images.shape[0]):
        T_opt = []
        for j in range(snrs.shape[-1]):
            snr = snrs[index, :, :, j]
            t_axis = t_axes[index, :, j]
            f_axis = f_axes[index, :, j]
            _, T_grid = np.meshgrid(f_axis, t_axis, indexing="ij")

            T_opt.append(T_grid[snr > threshold])
        
        T_opt = np.concatenate(T_opt)/3600
        if len(T_opt) < 2:
            if to_find == 0:
                good_idx.append(index)
            continue
        try:
            times = np.linspace(np.min(t_axes[index])/3600, np.max(t_axes[index])/3600, 250)
            pdf = gaussian_kde(T_opt)(times)
            peaks, _ = find_peaks(pdf, width=1, prominence=0.05)

            if len(peaks) >= to_find:
                good_idx.append(index)
        except:
            pass

    return np.array(good_idx)

def fit_random_forest(good_samples, bad_samples):
    good_x, good_y = good_samples
    bad_x, bad_y = bad_samples

    X_good = np.column_stack((good_x, good_y))
    X_bad = np.column_stack((bad_x, bad_y))

    X = np.vstack((X_good, X_bad))
    y = np.hstack((np.ones(len(X_good)), np.zeros(len(X_bad))))

    clf = RandomForestClassifier(n_estimators=500, random_state=42)
    clf.fit(X, y)

    dump(clf, FOREST_PATH)

def load_gw_testset():
    with h5py.File(gw_testset_path, 'r') as h5:
        m1_arr = h5['m1'][:]
        m2_arr = h5['m2'][:]
        d_arr = h5['d'][:]

    Mc_arr = chirp_mass(m1_arr, m2_arr)
    return np.log10(Mc_arr), np.log10(d_arr)

def load_glitch_testset():
    with h5py.File(glitch_testset_path, 'r') as h5:
        amp_arr = h5['amp'][:]
        beta_arr = h5['beta'][:]

    return np.log10(amp_arr), np.log10(beta_arr)

def main():
    good_idx = filter_by_snr(FILE_PATH, to_find=TO_FIND, threshold=8)

    if TRANSIENT_TYPE == 'glitch':
        print('Loading glitch testset...')
        log_amp, log_beta = load_glitch_testset()
        good_amp, good_beta = log_amp[good_idx], log_beta[good_idx]

        all_idx = np.arange(0, len(log_amp))
        bad_idx = np.setdiff1d(all_idx, good_idx)
        bad_amp, bad_beta = log_amp[bad_idx], log_beta[bad_idx]

        print('Fitting random forest...')
        fit_random_forest(good_samples=[good_beta, good_amp],
                          bad_samples=[bad_beta, bad_amp])
        print('Weights saved.')
    elif TRANSIENT_TYPE == 'gw':
        print('Loading GW testset...')
        log_Mc, log_d = load_gw_testset()
        good_Mc, good_d = log_Mc[good_idx], log_d[good_idx]
        
        all_idx = np.arange(0, len(log_Mc))
        bad_idx = np.setdiff1d(all_idx, good_idx)
        bad_Mc, bad_d = log_Mc[bad_idx], log_d[bad_idx]

        print('Fitting random forest...')
        fit_random_forest(good_samples=[good_d, good_Mc],
                            bad_samples=[bad_d, bad_Mc])
        print('Weights saved.')
if __name__ == "__main__":
    main()

    
    
