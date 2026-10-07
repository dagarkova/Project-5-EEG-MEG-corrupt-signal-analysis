import numpy as np
import mne
from mne.datasets import sample
import matplotlib.pyplot as plt

'''Generate simulated source data'''


data_path = sample.data_path()
subjects_dir = data_path / "subjects"
subject = "sample"

evoked_fname = data_path / "MEG" / subject / "sample_audvis-ave.fif"
info = mne.io.read_info(evoked_fname)
tstep = 1.0 / info["sfreq"]


fwd_fname = data_path / "MEG" / subject / "sample_audvis-meg-eeg-oct-6-fwd.fif"
fwd = mne.read_forward_solution(fwd_fname)
src = fwd["src"]


# WHERE?
selected_label = mne.read_labels_from_annot(subject, 
                                            regexp="caudalmiddlefrontal-lh", 
                                            subjects_dir=subjects_dir)[0]

# Use the center of the region as a seed.
location = "center"  

# Extent in mm of the region.
extent = 10.0  

label = mne.label.select_sources(subject, 
                                 selected_label, 
                                 location=location, 
                                 extent=extent, 
                                 subjects_dir=subjects_dir)

# WHAT?
source_time_series = np.sin(2.0 * np.pi * 18.0 * np.arange(100) * tstep) * 10e-9
plt.plot(source_time_series)


# make events
n_events = 50
events = np.zeros((n_events, 3), int)
events[:, 0] = 100 + 200 * np.arange(n_events)
events[:, 2] = 1


# make simulation
source_simulator = mne.simulation.SourceSimulator(src, tstep=tstep)
source_simulator.add_data(label, source_time_series, events)


# make raw data
raw = mne.simulation.simulate_raw(info, source_simulator, forward=fwd)
cov = mne.make_ad_hoc_cov(raw.info)
mne.simulation.add_noise(raw, cov, iir_filter=[0.2, -0.2, 0.04])
raw.plot()


# epochs
events = mne.find_events(raw)
epochs = mne.Epochs(raw, events, 1, tmin=-0.05, tmax=0.2)
evoked = epochs.average()
evoked.plot()


