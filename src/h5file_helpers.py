import os
import h5py
import numpy as np
from simulation import chirp_mass
from config import *

def create_dataset(dataset_path, siglen):
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
    else:
        h5file = h5py.File(dataset_path, "a")  # open in append mode

    return h5file

def append_gw_sample(h5, tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda):
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

def sort_gw_dataset(h5):
    chirp_masses = chirp_mass(h5["m1"][:], h5["m2"][:])
    lm = chirp_masses < 1e5
    hm = (chirp_masses >= 1e5) & (chirp_masses < 1e6)
    ehm = chirp_masses >= 1e6

    h5_lm = create_dataset(gw_lm_dataset_path, h5["X"].shape[1])
    h5_hm = create_dataset(gw_hm_dataset_path, h5["X"].shape[1])
    h5_ehm = create_dataset(gw_ehm_dataset_path, h5["X"].shape[1])

    tdi_dict_lm = {"X": h5["X"][lm], "Y": h5["Y"][lm], "Z": h5["Z"][lm]}
    tdi_dict_hm = {"X": h5["X"][hm], "Y": h5["Y"][hm], "Z": h5["Z"][hm]}
    tdi_dict_ehm = {"X": h5["X"][ehm], "Y": h5["Y"][ehm], "Z": h5["Z"][ehm]}

    append_gw_sample(h5_lm, tdi_dict_lm, h5["m1"][lm], h5["m2"][lm], h5["d"][lm],
                     h5["spin1"][lm], h5["spin2"][lm], h5["gw_beta"][lm], h5["gw_lambda"][lm])
    append_gw_sample(h5_hm, tdi_dict_hm, h5["m1"][hm], h5["m2"][hm], h5["d"][hm],
                     h5["spin1"][hm], h5["spin2"][hm], h5["gw_beta"][hm], h5["gw_lambda"][hm])
    append_gw_sample(h5_ehm, tdi_dict_ehm, h5["m1"][ehm], h5["m2"][ehm], h5["d"][ehm],
                     h5["spin1"][ehm], h5["spin2"][ehm], h5["gw_beta"][ehm], h5["gw_lambda"][ehm])