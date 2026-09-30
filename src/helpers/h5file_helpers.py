import h5py
import numpy as np
import json
import os
from helpers.config import *

def create_or_open_dataset(file_path):
    if os.path.exists(file_path):
        return h5py.File(file_path, 'a')
    
    h5file = h5py.File(file_path, 'w')

    # TDI channels
    tdi = h5file.create_group('tdi')

    for channel in ['X', 'Y', 'Z']:
        tdi.create_dataset(
            channel,
            shape=(0, PIPE['size']),
            maxshape=(None, PIPE['size']),
            dtype=np.float64,
            compression='gzip')

    # Metadata
    metadata = h5file.create_group('metadata')

    metadata.create_dataset(
        't0',
        shape=(0,),
        maxshape=(None,),
        dtype=np.float64)

    metadata.create_dataset(
        'smbhb_data',
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype())

    metadata.create_dataset(
        'glitch_data',
        shape=(0,),
        maxshape=(None,),
        dtype=h5py.string_dtype())

    return h5file

def append_sample(h5file, tdi_dict, metadata):
    # TDI channels
    for channel in ['X', 'Y', 'Z']:
        dset = h5file[f'tdi/{channel}']
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1] = tdi_dict[channel]

    # Metadata
    for key in ['t0', 'smbhb_data', 'glitch_data']:
        dset = h5file[f'metadata/{key}']
        dset.resize(dset.shape[0] + 1, axis=0)

        if key == 't0':
            dset[-1] = metadata[key]
        else:
            dset[-1] = json.dumps(metadata[key])

def create_imageset(h5file, time_steps, resolution, channels):

    h5file.create_dataset(
        'images', 
        shape=(0, resolution, resolution, channels),
        maxshape=(None,resolution, resolution, channels),  # unlimited along axis 0
        dtype="float32",
        chunks=(1, resolution, resolution, channels),       
        compression="gzip")

    h5file.create_dataset(
        'labels', shape=(0, 2), maxshape=(None, 2), dtype="int8",)

    image_metadata = h5file.create_group('image_metadata')

    image_metadata.create_dataset(
        't_axis', shape=(0, resolution, channels),
        maxshape=(None,resolution, channels), dtype="float32")

    image_metadata.create_dataset(
        'f_axis', shape=(0, resolution, channels),
        maxshape=(None,resolution, channels), dtype="float32")

    image_metadata.create_dataset(
        'tcen', shape=(0,),
        maxshape=(None,), dtype="float32")

    image_metadata.create_dataset(
        'sim_idx', shape=(0,),
        maxshape=(None,), dtype="int8")

    image_metadata.create_dataset(
        'time_idx', shape=(0,),
        maxshape=(None,), dtype="int8")

    return h5file

def append_image(h5file, image, label, image_metadata):
    # Images
    dset = h5file['images']
    dset.resize(dset.shape[0] + 1, axis=0)
    dset[-1] = image

    # Labels
    dset = h5file['labels']
    dset.resize(dset.shape[0] + 1, axis=0)
    dset[-1] = label

    # Image metadata
    for key in ['t_axis', 'f_axis', 'tcen', 'sim_idx', 'time_idx']:
        dset = h5file[f'image_metadata/{key}']
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1] = image_metadata[key]