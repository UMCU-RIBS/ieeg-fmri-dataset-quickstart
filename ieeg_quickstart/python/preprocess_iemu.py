import numpy as np
import mne
import pandas as pd
import mne_bids

from ieeg_fmri_quickstart.utils import resample, smooth_signal

##

bids_dir='/Fridge/users/julia/project_chill_dataset_paper/data/BIDS2'
subjects = mne_bids.get_entity_vals(bids_dir, 'subject')

subject = '01'
acquisition = 'clinical'
fs_dir = '/Fridge/users/julia/project_chill_dataset_paper/data/freesurfer2/sub-01'

##
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
bad_channels = channels['name'][(channels['type'].isin(['ECOG', 'SEEG'])) & (channels['status'] == 'bad')].tolist()
raw.info['bads'].extend([ch for ch in bad_channels])
raw.drop_channels(raw.info['bads'])

##
raw.load_data()

##
raw.notch_filter(freqs=np.arange(50, 251, 50))

##
raw_car, _ = mne.set_eeg_reference(raw.copy(), 'average')

##
gamma = raw_car.copy().filter(60, 120).apply_hilbert(envelope=True).get_data().T
# temp = mne.time_frequency.tfr_array_morlet(np.expand_dims(self.raw_car.copy()._data, 0), # (n_epochs, n_channels, n_times)
#                                                      sfreq=self.raw.info['sfreq'],
#                                                      freqs=np.arange(60, 120),
#                                                      verbose=True,
#                                                      n_cycles=4.,
#                                                      n_jobs=1)
# gamma = np.mean(np.abs(temp), 2).squeeze().T

##
custom_mapping = {'Stimulus/music': 2, 'Stimulus/speech': 1,
                  'Stimulus/end task': 5}  # 'Stimulus/task end' in laan
events, event_id = mne.events_from_annotations(raw_car, event_id=custom_mapping,
                                                         use_rounding=False)

raw_car.plot(events=events, start=0, duration=180, color='gray', event_color={2: 'g', 1: 'r'}, bgcolor='w')

##
gamma_cropped = gamma[events[0, 0]:events[-1, 0]]

##
gamma_resampled = resample(gamma_cropped, 25, int(raw.info['sfreq']))



##


gamma = np.apply_along_axis(smooth_signal, 0, gamma, 5)