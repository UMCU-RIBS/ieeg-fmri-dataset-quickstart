

import numpy as np
import mne
import pandas as pd
import os
import mne_bids
import nibabel


##

bids_dir='/Fridge/users/julia/project_chill_dataset_paper/data/BIDS2'
subjects = mne_bids.get_entity_vals(bids_dir, 'subject')

subject = '01'
acquisition = 'clinical'
fs_dir = '/Fridge/users/julia/project_chill_dataset_paper/data/freesurfer2/sub-01'

##
electrodes_path = mne_bids.BIDSPath(subject=subject,
                                    session='iemu',
                                    suffix='electrodes',
                                    extension='tsv',
                                    datatype='ieeg',
                                    acquisition=acquisition,
                                    root=bids_dir)
electrodes = pd.read_csv(str(electrodes_path), sep='\t', header=0, index_col=None)
coords = electrodes[['x', 'y', 'z']].values

channels_path = mne_bids.BIDSPath(subject=subject,
                                    session='iemu',
                                    suffix='channels',
                                    extension='tsv',
                                    datatype='ieeg',
                                    task='film',
                                    acquisition=acquisition,
                                    root=bids_dir)
channels = pd.read_csv(str(channels_path.match()[0]), sep='\t', header=0, index_col=None)

##
data_path = mne_bids.BIDSPath(subject=subject,
                                    session='iemu',
                                    suffix='ieeg',
                                    extension='vhdr',
                                    datatype='ieeg',
                                    task='film',
                                    acquisition=acquisition,
                                    root=bids_dir)
raw = mne.io.read_raw_brainvision(str(data_path.match()[0]), scale=1.0, preload=False, verbose=True)
raw.set_channel_types({ch_name: str(x).lower()
                if str(x).lower() in ['ecog', 'seeg', 'eeg'] else 'misc'
                                for ch_name, x in zip(raw.ch_names, channels['type'].values)})
raw.drop_channels([raw.ch_names[i] for i, j in enumerate(raw.get_channel_types()) if j == 'misc'])

##
x = nibabel.load(os.path.join(fs_dir, 'mri', 'orig.mgz'))
vox_coords = np.round(mne.transforms.apply_trans(np.linalg.inv(x.affine), coords)).astype(int)
ras_coords = mne.transforms.apply_trans(x.header.get_vox2ras_tkr(), vox_coords)
ras_coords = ras_coords / 1000

montage = mne.channels.make_dig_montage(ch_pos=dict(zip(raw.ch_names, ras_coords)), coord_frame='mri')
raw.set_montage(montage)

##
fig = mne.viz.plot_alignment(raw.info,
               subject='sub-' + subject,
               subjects_dir=os.path.dirname(fs_dir),
               surfaces=['pial'],
               coord_frame='mri')
mne.viz.set_3d_view(fig, 180, 70, distance=.5)
