import numpy as np
import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.simulation import simulate_evoked, simulate_stc

'Corrupt known signal with point spread'

seed = 42

method = "sLORETA" # sLORETA
snr = 10.0
lambda2 = 1.0 / snr**2

nave = 100 #np.inf
T = 100
times = np.linspace(0, 1, T)
dt = times[1] - times[0]

'Paths to MEG data'
data_path = sample.data_path()
subjects_dir = data_path / "subjects"
fname_fwd = data_path / "MEG" / "sample" / "sample_audvis-meg-oct-6-fwd.fif"
fname_inv = data_path / "MEG" / "sample" / "sample_audvis-meg-oct-6-meg-fixed-inv.fif"
fname_evoked = data_path / "MEG" / "sample" / "sample_audvis-ave.fif"

'Load the MEG data'
fwd = mne.read_forward_solution(fname_fwd)
fwd = mne.convert_forward_solution(fwd, force_fixed=True, surf_ori=True, use_cps=False)
fwd["info"]["bads"] = []
inv_op = read_inverse_operator(fname_inv)

raw = mne.io.read_raw_fif(data_path / "MEG" / "sample" / "sample_audvis_raw.fif")
raw.info["bads"] = []
raw.set_eeg_reference(projection=True)
events = mne.find_events(raw)
event_id = {"Auditory/Left": 1, "Auditory/Right": 2}
epochs = mne.Epochs(raw, events, event_id, baseline=(None, 0), preload=True)
evoked = epochs.average()
evoked.plot()

labels = mne.read_labels_from_annot("sample", subjects_dir=subjects_dir)
label_names = [label.name for label in labels]
n_labels = len(labels)


'Estimate noise covariance'
cov = mne.compute_covariance(epochs, tmin=None, tmax=0.0)

'Generate sinusoids in two spatially distant labels'
signal = np.zeros((n_labels, T))
idx = label_names.index("inferiorparietal-lh")
signal[idx, :] = 1e-7 * np.sin(5 * 2 * np.pi * times)
idx = label_names.index("rostralmiddlefrontal-rh")
signal[idx, :] = 1e-7 * np.sin(7 * 2 * np.pi * times)


'Find the center vertices in source space of each label'
hemi_to_ind = {"lh": 0, "rh": 1}
for i, label in enumerate(labels):
    labels[i].values.fill(1.0)

    # Restrict the eligible vertices to be those on the surface under
    # consideration and within the label.
    surf_vertices = fwd["src"][hemi_to_ind[label.hemi]]["vertno"]
    restrict_verts = np.intersect1d(surf_vertices, label.vertices)
    com = labels[i].center_of_mass(
        subjects_dir=subjects_dir, restrict_vertices=restrict_verts, surf="white"
    )

    # Convert the center of vertex index from surface vertex list to Label's
    # vertex list.
    cent_idx = np.where(label.vertices == com)[0][0]

    # Create a mask with 1 at center vertex and zeros elsewhere.
    labels[i].values.fill(0.0)
    labels[i].values[cent_idx] = 1.0


'Create source-space data with known signals'
stc_gen = simulate_stc(fwd["src"], labels, signal, times[0], dt, value_fun=lambda x: x)

'Plot original signals'
kwargs = dict(subjects_dir=subjects_dir,
              hemi="both",
              smoothing_steps=4,
              time_unit="s",
              initial_time=0.05,
              size=1200,
              views=["lat"])
brain_gen = stc_gen.plot(**kwargs)

'Simulate sensor-space signals'
evoked_gen = simulate_evoked(fwd, stc_gen, evoked.info, cov, nave, random_state=seed)
evoked_gen.plot()
stc_inv = apply_inverse(evoked_gen, inv_op, lambda2, method=method)


'Plot the point-spread of corrupted signal'
brain_inv = stc_inv.plot(**kwargs)


#Change the method parameter to either 'dSPM' or 'MNE' to explore the effect of the inverse method.
#Try setting evoked_snr to a small, finite value, e.g. 3., to see the effect of noise.



















