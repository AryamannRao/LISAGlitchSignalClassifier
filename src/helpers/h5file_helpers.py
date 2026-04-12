import os
import sys
import h5py
import numpy as np
from pathlib import Path
# Ensure src is on Python path regardless of where code is run from
SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from helpers.simulation import chirp_mass
from helpers.config import *

def create_gw_dataset(dataset_path, siglen):
    #H, W = resolution

    if not os.path.exists(dataset_path):
        h5file = h5py.File(dataset_path, "w")
        h5file.create_dataset(
            "X", 
            shape=(0, siglen),
            maxshape=(None, siglen),  # unlimited along axis 0
            dtype="float32",
            chunks=(1, siglen),       # good practice
            compression="gzip"
        )

        h5file.create_dataset(
            "Y",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )
        h5file.create_dataset(
            "Z",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )

        h5file.create_dataset(
            "m1",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "m2",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "d",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "spin1",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "spin2",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "gw_beta",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "gw_lambda",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "t0",   shape=(0,), maxshape=(None,), dtype="float32"
        )
    else:
        h5file = h5py.File(dataset_path, "a")  # open in append mode

    return h5file

def append_gw_sample(h5, tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda, t0):
    X = tdi_dict["X"]
    Y = tdi_dict["Y"]
    Z = tdi_dict["Z"]

    n = h5["X"].shape[0]  # current length

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)

    m1 = np.atleast_1d(m1)
    m2 = np.atleast_1d(m2)
    d = np.atleast_1d(d)
    spin1 = np.atleast_1d(spin1)
    spin2 = np.atleast_1d(spin2)
    gw_beta = np.atleast_1d(gw_beta)
    gw_lambda = np.atleast_1d(gw_lambda)
    t0 = np.atleast_1d(t0)
    
    batch_size = X.shape[0]
    
    # current length
    n = h5["X"].shape[0]
    new_n = n + batch_size

    # resize datasets
    h5["X"].resize((new_n, X.shape[1]))
    h5["Y"].resize((new_n, Y.shape[1]))
    h5["Z"].resize((new_n, Z.shape[1]))

    h5["m1"].resize((new_n,))
    h5["m2"].resize((new_n,))
    h5["d"].resize((new_n,))
    h5["spin1"].resize((new_n,))
    h5["spin2"].resize((new_n,))
    h5["gw_beta"].resize((new_n,))
    h5["gw_lambda"].resize((new_n,))
    h5["t0"].resize((new_n,))

    # store batch
    h5["X"][n:new_n] = X.astype("float32")
    h5["Y"][n:new_n] = Y.astype("float32")
    h5["Z"][n:new_n] = Z.astype("float32")

    h5["m1"][n:new_n] = m1
    h5["m2"][n:new_n] = m2
    h5["d"][n:new_n] = d
    h5["spin1"][n:new_n] = spin1
    h5["spin2"][n:new_n] = spin2
    h5["gw_beta"][n:new_n] = gw_beta
    h5["gw_lambda"][n:new_n] = gw_lambda
    h5["t0"][n:new_n] = t0

def sort_gw_dataset(h5):
    chirp_masses = chirp_mass(h5["m1"][:], h5["m2"][:])
    lm = chirp_masses < 1e5
    hm = (chirp_masses >= 1e5) & (chirp_masses < 1e6)
    ehm = chirp_masses >= 1e6

    h5_lm = create_gw_dataset(gw_lm_dataset_path, h5["X"].shape[1])
    h5_hm = create_gw_dataset(gw_hm_dataset_path, h5["X"].shape[1])
    h5_ehm = create_gw_dataset(gw_ehm_dataset_path, h5["X"].shape[1])

    tdi_dict_lm = {"X": h5["X"][lm], "Y": h5["Y"][lm], "Z": h5["Z"][lm]}
    tdi_dict_hm = {"X": h5["X"][hm], "Y": h5["Y"][hm], "Z": h5["Z"][hm]}
    tdi_dict_ehm = {"X": h5["X"][ehm], "Y": h5["Y"][ehm], "Z": h5["Z"][ehm]}

    append_gw_sample(h5_lm, tdi_dict_lm, h5["m1"][lm], h5["m2"][lm], h5["d"][lm],
                     h5["spin1"][lm], h5["spin2"][lm], h5["gw_beta"][lm], h5["gw_lambda"][lm], h5["t0"][lm])
    append_gw_sample(h5_hm, tdi_dict_hm, h5["m1"][hm], h5["m2"][hm], h5["d"][hm],
                     h5["spin1"][hm], h5["spin2"][hm], h5["gw_beta"][hm], h5["gw_lambda"][hm], h5["t0"][hm])
    append_gw_sample(h5_ehm, tdi_dict_ehm, h5["m1"][ehm], h5["m2"][ehm], h5["d"][ehm],
                     h5["spin1"][ehm], h5["spin2"][ehm], h5["gw_beta"][ehm], h5["gw_lambda"][ehm], h5["t0"][ehm])
    
def create_glitch_dataset(dataset_path, siglen):
    if not os.path.exists(dataset_path):
        h5file = h5py.File(dataset_path, "w")
        h5file.create_dataset(
            "X", 
            shape=(0, siglen),
            maxshape=(None, siglen),  # unlimited along axis 0
            dtype="float32",
            chunks=(1, siglen),       # good practice
            compression="gzip"
        )

        h5file.create_dataset(
            "Y",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )
        h5file.create_dataset(
            "Z",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )

        h5file.create_dataset(
            "amp",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "beta",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "inj_point",   shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        h5file.create_dataset(
            "t0",   shape=(0,), maxshape=(None,), dtype="float32"
        )

    else:
        h5file = h5py.File(dataset_path, "a")  # open in append mode

    return h5file

def append_glitch_sample(h5, tdi_dict, amp, beta, inj_point, t0):
    X = tdi_dict["X"]
    Y = tdi_dict["Y"]
    Z = tdi_dict["Z"]

    n = h5["X"].shape[0]  # current length

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)

    amp = np.atleast_1d(amp)
    beta = np.atleast_1d(beta)
    inj_point = np.atleast_1d(inj_point)
    t0 = np.atleast_1d(t0)

    batch_size = X.shape[0]
    
    # current length
    n = h5["X"].shape[0]
    new_n = n + batch_size

    # resize datasets
    h5["X"].resize((new_n, X.shape[1]))
    h5["Y"].resize((new_n, Y.shape[1]))
    h5["Z"].resize((new_n, Z.shape[1]))

    h5["amp"].resize((new_n,))
    h5["beta"].resize((new_n,))
    h5["inj_point"].resize((new_n,))
    h5["t0"].resize((new_n,))

    # store batch
    h5["X"][n:new_n] = X.astype("float32")
    h5["Y"][n:new_n] = Y.astype("float32")
    h5["Z"][n:new_n] = Z.astype("float32")

    h5["amp"][n:new_n] = amp
    h5["beta"][n:new_n] = beta
    h5["inj_point"][n:new_n] = inj_point
    h5["t0"][n:new_n] = t0

def sort_glitch_dataset(h5):
    n = h5["X"].shape[0]

    lm = np.arange(0, n//3, 1)
    hm = np.arange(n//3, 2*n//3, 1)
    ehm = np.arange(2*n//3, n, 1)

    h5_lm = create_glitch_dataset(glitch_lm_dataset_path, h5["X"].shape[1])
    h5_hm = create_glitch_dataset(glitch_hm_dataset_path, h5["X"].shape[1])
    h5_ehm = create_glitch_dataset(glitch_ehm_dataset_path, h5["X"].shape[1])

    tdi_dict_lm = {"X": h5["X"][lm], "Y": h5["Y"][lm], "Z": h5["Z"][lm]}
    tdi_dict_hm = {"X": h5["X"][hm], "Y": h5["Y"][hm], "Z": h5["Z"][hm]}
    tdi_dict_ehm = {"X": h5["X"][ehm], "Y": h5["Y"][ehm], "Z": h5["Z"][ehm]}

    append_glitch_sample(h5_lm, tdi_dict_lm, h5["level"][lm], h5["beta"][lm], h5["inj_point"][lm], h5["t0"][lm])
    append_glitch_sample(h5_hm, tdi_dict_hm, h5["level"][hm], h5["beta"][hm], h5["inj_point"][hm], h5["t0"][hm])
    append_glitch_sample(h5_ehm, tdi_dict_ehm, h5["level"][ehm], h5["beta"][ehm], h5["inj_point"][ehm], h5["t0"][ehm])

def create_mixed_dataset(dataset_path, siglen):
    if not os.path.exists(dataset_path):
        h5file = h5py.File(dataset_path, "w")
        h5file.create_dataset(
            "X", 
            shape=(0, siglen),
            maxshape=(None, siglen),  # unlimited along axis 0
            dtype="float32",
            chunks=(1, siglen),       # good practice
            compression="gzip"
        )

        h5file.create_dataset(
            "Y",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )
        h5file.create_dataset(
            "Z",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )

        # Add datasets for both GW and glitch parameters here
        h5file.create_dataset(
            "m1",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "m2",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "d",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "spin1",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "spin2",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "gw_beta",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "gw_lambda",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "amp",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "beta",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "inj_point",   shape=(0,), maxshape=(None,), dtype=h5py.string_dtype(encoding="utf-8")
        )
        h5file.create_dataset(
            "sep",   shape=(0,), maxshape=(None,), dtype="float32"
        )
        h5file.create_dataset(
            "t0",   shape=(0,), maxshape=(None,), dtype="float32"
        )

    else:
        h5file = h5py.File(dataset_path, "a")  # open in append mode

    return h5file

def append_mixed_sample(h5, tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda,
                        amp, beta, inj_point, sep, t0):
    X = tdi_dict["X"]
    Y = tdi_dict["Y"]
    Z = tdi_dict["Z"]

    n = h5["X"].shape[0]  # current length

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)

    m1 = np.atleast_1d(m1)
    m2 = np.atleast_1d(m2)
    d = np.atleast_1d(d)
    spin1 = np.atleast_1d(spin1)
    spin2 = np.atleast_1d(spin2)
    gw_beta = np.atleast_1d(gw_beta)
    gw_lambda = np.atleast_1d(gw_lambda)
    amp = np.atleast_1d(amp)
    beta = np.atleast_1d(beta)
    inj_point = np.atleast_1d(inj_point)
    sep = np.atleast_1d(sep)
    t0 = np.atleast_1d(t0)
    
    batch_size = X.shape[0]
    
    # current length
    n = h5["X"].shape[0]
    new_n = n + batch_size

    # resize datasets
    h5["X"].resize((new_n, X.shape[1]))
    h5["Y"].resize((new_n, Y.shape[1]))
    h5["Z"].resize((new_n, Z.shape[1]))

    h5["m1"].resize((new_n,))
    h5["m2"].resize((new_n,))
    h5["d"].resize((new_n,))
    h5["spin1"].resize((new_n,))
    h5["spin2"].resize((new_n,))
    h5["gw_beta"].resize((new_n,))
    h5["gw_lambda"].resize((new_n,))
    h5["amp"].resize((new_n,))
    h5["beta"].resize((new_n,))
    h5["inj_point"].resize((new_n,))
    h5["sep"].resize((new_n,))
    h5["t0"].resize((new_n,))

    # store batch
    h5["X"][n:new_n] = X.astype("float32")
    h5["Y"][n:new_n] = Y.astype("float32")
    h5["Z"][n:new_n] = Z.astype("float32")

    h5["m1"][n:new_n] = m1
    h5["m2"][n:new_n] = m2
    h5["d"][n:new_n] = d
    h5["spin1"][n:new_n] = spin1
    h5["spin2"][n:new_n] = spin2
    h5["gw_beta"][n:new_n] = gw_beta
    h5["gw_lambda"][n:new_n] = gw_lambda
    h5["amp"][n:new_n] = amp
    h5["beta"][n:new_n] = beta
    h5["inj_point"][n:new_n] = inj_point
    h5["sep"][n:new_n] = sep
    h5["t0"][n:new_n] = t0

def sort_mixed_dataset(h5):
    chirp_masses = chirp_mass(h5["m1"][:], h5["m2"][:])
    lm = chirp_masses < 1e5
    hm = (chirp_masses >= 1e5) & (chirp_masses < 1e6)
    ehm = chirp_masses >= 1e6

    h5_lm = create_mixed_dataset(mixed_lm_dataset_path, h5["X"].shape[1])
    h5_hm = create_mixed_dataset(mixed_hm_dataset_path, h5["X"].shape[1])
    h5_ehm = create_mixed_dataset(mixed_ehm_dataset_path, h5["X"].shape[1])

    tdi_dict_lm = {"X": h5["X"][lm], "Y": h5["Y"][lm], "Z": h5["Z"][lm]}
    tdi_dict_hm = {"X": h5["X"][hm], "Y": h5["Y"][hm], "Z": h5["Z"][hm]}
    tdi_dict_ehm = {"X": h5["X"][ehm], "Y": h5["Y"][ehm], "Z": h5["Z"][ehm]}

    append_mixed_sample(h5_lm, tdi_dict_lm, h5["m1"][lm], h5["m2"][lm], h5["d"][lm],
                     h5["spin1"][lm], h5["spin2"][lm], h5["gw_beta"][lm], h5["gw_lambda"][lm],
                     h5["amp"][lm], h5["beta"][lm], h5["inj_point"][lm], h5["sep"][lm], h5["t0"][lm])
    append_mixed_sample(h5_hm, tdi_dict_hm, h5["m1"][hm], h5["m2"][hm], h5["d"][hm],
                     h5["spin1"][hm], h5["spin2"][hm], h5["gw_beta"][hm], h5["gw_lambda"][hm],
                     h5["amp"][hm], h5["beta"][hm], h5["inj_point"][hm], h5["sep"][hm], h5["t0"][hm])
    append_mixed_sample(h5_ehm, tdi_dict_ehm, h5["m1"][ehm], h5["m2"][ehm], h5["d"][ehm],
                     h5["spin1"][ehm], h5["spin2"][ehm], h5["gw_beta"][ehm], h5["gw_lambda"][ehm],
                     h5["amp"][ehm], h5["beta"][ehm], h5["inj_point"][ehm], h5["sep"][ehm], h5["t0"][ehm])

def create_empty_dataset(dataset_path, siglen):
    if not os.path.exists(dataset_path):
        h5file = h5py.File(dataset_path, "w")
        h5file.create_dataset(
            "X", 
            shape=(0, siglen),
            maxshape=(None, siglen),  # unlimited along axis 0
            dtype="float32",
            chunks=(1, siglen),       # good practice
            compression="gzip"
        )

        h5file.create_dataset(
            "Y",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )
        h5file.create_dataset(
            "Z",
            shape=(0, siglen),
            maxshape=(None, siglen),
            dtype="float32",
            chunks=(1, siglen),
            compression="gzip",
        )
        h5file.create_dataset(
            "t0",   shape=(0,), maxshape=(None,), dtype="float32"
        )
    else:
        h5file = h5py.File(dataset_path, "a")  # open in append mode

    return h5file

def append_empty_sample(h5, tdi_dict, t0):

    X = tdi_dict["X"]
    Y = tdi_dict["Y"]
    Z = tdi_dict["Z"]

    n = h5["X"].shape[0]  # current length

    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    Z = np.atleast_2d(Z)
    t0 = np.atleast_1d(t0)

    batch_size = X.shape[0]
    
    # current length
    n = h5["X"].shape[0]
    new_n = n + batch_size

    # resize datasets
    h5["X"].resize((new_n, X.shape[1]))
    h5["Y"].resize((new_n, Y.shape[1]))
    h5["Z"].resize((new_n, Z.shape[1]))
    h5["t0"].resize((new_n,))

    # store batch
    h5["X"][n:new_n] = X.astype("float32")
    h5["Y"][n:new_n] = Y.astype("float32")
    h5["Z"][n:new_n] = Z.astype("float32")
    h5["t0"][n:new_n] = t0

def sort_empty_dataset(h5):
    n = h5["X"].shape[0]

    lm = np.arange(0, n//3, 1)
    hm = np.arange(n//3, 2*n//3, 1)
    ehm = np.arange(2*n//3, n, 1)

    h5_lm = create_empty_dataset(empty_lm_dataset_path, h5["X"].shape[1])
    h5_hm = create_empty_dataset(empty_hm_dataset_path, h5["X"].shape[1])
    h5_ehm = create_empty_dataset(empty_ehm_dataset_path, h5["X"].shape[1])

    tdi_dict_lm = {"X": h5["X"][lm], "Y": h5["Y"][lm], "Z": h5["Z"][lm]}
    tdi_dict_hm = {"X": h5["X"][hm], "Y": h5["Y"][hm], "Z": h5["Z"][hm]}
    tdi_dict_ehm = {"X": h5["X"][ehm], "Y": h5["Y"][ehm], "Z": h5["Z"][ehm]}

    append_empty_sample(h5_lm, tdi_dict_lm, h5["t0"][lm])
    append_empty_sample(h5_hm, tdi_dict_hm, h5["t0"][hm])
    append_empty_sample(h5_ehm, tdi_dict_ehm, h5["t0"][ehm])

def create_imageset(dataset_path, time_steps, resolution, channels, keys):
    image_key, tarr_key, farr_key = keys
    h5file = h5py.File(dataset_path, "a")

    h5file.create_dataset(
        image_key, 
        shape=(0, time_steps, resolution, resolution, channels),
        maxshape=(None,time_steps, resolution, resolution, channels),  # unlimited along axis 0
        dtype="float32",
        chunks=(1, time_steps, resolution, resolution, channels),       # good practice
        compression="gzip"
    )
    h5file.create_dataset(
        tarr_key,   shape=(0, time_steps, resolution, channels),
        maxshape=(None,time_steps, resolution, channels), dtype="float32"
    )
    h5file.create_dataset(
        farr_key,   shape=(0, time_steps, resolution, channels),
        maxshape=(None,time_steps, resolution, channels), dtype="float32"
    )
    h5file.create_dataset(
    "tcen",
    shape=(0, time_steps),
    maxshape=(None, time_steps),
    dtype="float32")

    return h5file

def append_image(h5, image, t_arr, f_arr, tcen, keys):
    image_key, tarr_key, farr_key = keys
    n = h5[image_key].shape[0]  # current length

    if image.ndim == 4:
        image = np.expand_dims(image, axis=0)
    if t_arr.ndim == 3:
        t_arr = np.expand_dims(t_arr, axis=0)
    if f_arr.ndim == 3:
        f_arr = np.expand_dims(f_arr, axis=0)
    tcen = np.atleast_2d(tcen)
    
    batch_size = image.shape[0]
    
    # current length
    n = h5[image_key].shape[0]
    new_n = n + batch_size

    # resize datasets
    h5[image_key].resize((new_n, image.shape[1], image.shape[2], image.shape[3], image.shape[4]))
    h5[tarr_key].resize((new_n, t_arr.shape[1], t_arr.shape[2], t_arr.shape[3]))
    h5[farr_key].resize((new_n, f_arr.shape[1], f_arr.shape[2], f_arr.shape[3]))
    h5["tcen"].resize((new_n, tcen.shape[1]))

    # store batch
    h5[image_key][n:new_n] = image.astype("float32")
    h5[tarr_key][n:new_n] = t_arr.astype("float32")
    h5[farr_key][n:new_n] = f_arr.astype("float32")
    h5["tcen"][n:new_n] = tcen.astype("float32")