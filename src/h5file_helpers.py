import h5py

def create_dataset(dataset_path, siglen):
    #H, W = resolution
    h5file = h5py.File(dataset_path, "a")

    if "X" not in h5file:
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

    return h5file

def append_gw_sample(h5, tdi_dict, m1, m2, d, spin1, spin2, gw_beta, gw_lambda):
    X = tdi_dict["X"]
    Y = tdi_dict["Y"]
    Z = tdi_dict["Z"]

    n = h5["X"].shape[0]  # current length

    # resize all datasets by +1
    h5["X"].resize((n + 1, X.shape[0]))
    h5["Y"].resize((n + 1, Y.shape[0]))
    h5["Z"].resize((n + 1, Z.shape[0]))

    h5["m1"].resize((n + 1,))
    h5["m2"].resize((n + 1,))
    h5["d"].resize((n + 1,))
    h5["spin1"].resize((n + 1,))
    h5["spin2"].resize((n + 1,))
    h5["gw_beta"].resize((n + 1,))
    h5["gw_lambda"].resize((n + 1,))

    # store
    h5["X"][n] = X.astype("float32")
    h5["Y"][n] = Y.astype("float32")
    h5["Z"][n] = Z.astype("float32")

    h5["m1"][n] = m1
    h5["m2"][n] = m2
    h5["d"][n] = d
    h5["spin1"][n] = spin1
    h5["spin2"][n] = spin2
    h5["gw_beta"][n] = gw_beta
    h5["gw_lambda"][n] = gw_lambda