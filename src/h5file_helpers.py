import h5py

def create_dataset(dataset_path, resolution=(512, 512)):
    H, W = resolution
    h5file = h5py.File(dataset_path, "a")

    if "QT" not in h5file:
        h5file.create_dataset(
            "QT", 
            shape=(0, H, W),
            maxshape=(None, H, W),  # unlimited along axis 0
            dtype="float32",
            chunks=(1, H, W),       # good practice
            compression="gzip"
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
            "spin",   shape=(0,), maxshape=(None,), dtype="float32"
        )

    return h5file

def append_gw_sample(h5, QT, m1, m2, d, spin):
    n = h5["QT"].shape[0]  # current length

    # resize all datasets by +1
    h5["QT"].resize((n + 1, QT.shape[0], QT.shape[1]))
    h5["m1"].resize((n + 1,))
    h5["m2"].resize((n + 1,))
    h5["d"].resize((n + 1,))
    h5["spin"].resize((n + 1,))

    # store
    h5["QT"][n] = QT.astype("float32")
    h5["m1"][n] = m1
    h5["m2"][n] = m2
    h5["d"][n] = d
    h5["spin"][n] = spin