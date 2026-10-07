import numpy as np
import mne
from mne import spatial_src_adjacency
import os
import matplotlib.pyplot as plt
from numpy.random import randn
from mne.minimum_norm import apply_inverse
from mne.stats import (
    f_mway_rm,
    f_threshold_mway_rm,
    spatio_temporal_cluster_test,
    summarize_clusters_stc)


path = '/Users/alextumyalis/mne_data'
data_path = f'{path}/MNE-sample-data'
subjects_dir = f'{path}/MNE-sample-data/subjects'
subject = 'sample' # subject name or ID

path = '/Users/alextumyalis/Documents/документы/Преподавание/Вычислительные науки'
os.chdir(path)


'load data'
epochs = mne.read_epochs('ERP_epochs.fif', preload=True)
epochs_eeg = epochs.copy().pick_types(eeg=True)
epochs_eeg.info


'epochs selection'
epochs_vis_left = epochs_eeg['visual/left']
epochs_vis_right = epochs_eeg['visual/right']


'averaging'
epochs_vis_left_av = epochs_vis_left.average()
epochs_vis_right_av = epochs_vis_right.average()
epochs_vis_right_av.info
epochs_vis_left_av.plot()


'Compute and visualize BEM surfaces'
plot_bem_kwargs = dict(subject=subject, 
                       subjects_dir=subjects_dir,
                       brain_surfaces='white',
                       orientation='axial', #‘coronal’ or ‘axial’ or ‘sagittal’.
                       slices=[60, 80, 100, 120, 140, 160, 180, 200, 220])
mne.viz.plot_bem(**plot_bem_kwargs)


'Visualizing the coregistration'
trans = data_path + '/MEG/sample/sample_audvis_raw-trans.fif'
info = epochs_eeg.info
mne.viz.plot_alignment(info, 
                       trans, # trans  "fsaverage"
                       subjects_dir=subjects_dir,
                       subject=subject,
                       dig=True,
                       #meg=["helmet", "sensors"],
                       eeg=dict(original=0.2, projected=0.8),
                       surfaces='head-dense') # eeg: head-dense, auto; meg: white brain


'Compute Source Space'
src = mne.setup_source_space(subject, 
                             spacing='oct6', # The spacing to use. Can be 'ico#' for a recursively subdivided icosahedron, 'oct#' for a recursively subdivided octahedron, 'all' for all points, or an integer to use approximate distance-based spacing (in mm).
                             add_dist='patch',
                             subjects_dir=subjects_dir)


surface = '/Users/alextumyalis/mne_data/MNE-sample-data/subjects/sample/bem/inner_skull.surf'
src = mne.setup_volume_source_space(subject, 
                                    subjects_dir=subjects_dir, 
                                    surface=surface,
                                    add_interpolator=True)
#mne.viz.plot_bem(src=vol_src, **plot_bem_kwargs)

'3d visualization'
fig = mne.viz.plot_alignment(info, 
                             trans, 
                             subject=subject, 
                             subjects_dir=subjects_dir,
                             surfaces='white',
                             coord_frame='mri',
                             src=src)

mne.viz.set_3d_view(fig, 
                    azimuth=175,  # left - right angle
                    elevation=90, # bottom - top angle
                    distance=0.80, 
                    focalpoint=(-0.03, -0.01, 0.03))



'Compute forward solution'
conductivity = (0.3, 0.006, 0.3)  # The conductivities to use for each shell.
model = mne.make_bem_model(subject='sample', 
                           ico=3, # The surface ico downsampling to use, e.g. 5=20484, 4=5120, 3=1280
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model) #  returned surfaces into a ConductorModel suitable for forward calculation
fwd = mne.make_forward_solution(epochs_vis_right.info, 
                                trans=trans, 
                                src=src, 
                                bem=bem,
                                meg=False, 
                                eeg=True, 
                                mindist=5.0, 
                                n_jobs=1,
                                verbose=True)

fwd_left = mne.make_forward_solution(epochs_vis_left.info, 
                                     trans=trans, 
                                     src=src, 
                                     bem=bem,
                                     meg=False, 
                                     eeg=True, 
                                     mindist=5.0, 
                                     n_jobs=1,
                                     verbose=True)


'Compute noice covariance'
noise_cov = mne.compute_covariance(epochs_vis_right, 
                                   tmax=0.0, 
                                   method=["empirical"], # the sample covariance will be computed ['shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']
                                   rank='full', # equal to the number of good channels
                                   verbose=True)
noise_cov_left = mne.compute_covariance(epochs_vis_left, 
                                        tmax=0.0, 
                                        method=["empirical"], # the sample covariance will be computed ['shrunk', 'diagonal_fixed', 'empirical', 'factor_analysis']
                                        rank='full', # equal to the number of good channels
                                        verbose=True)


'Compute inverse operator'
inverse_operator = mne.minimum_norm.make_inverse_operator(epochs_vis_right_av.info, 
                                                          fwd, 
                                                          noise_cov, 
                                                          loose=0.0, #Value that weights the source variances of the dipole components that are parallel (tangential) to the cortical surface. Can be:
                                                                      #float between 0 and 1 (inclusive)
                                                                      #If 0, then the solution is computed with fixed orientation. If 1, it corresponds to free orientations.
                                                          depth=0.8, # How to weight (or normalize) the forward using a depth prior
                                                          rank='full')
   
inverse_operator_left = mne.minimum_norm.make_inverse_operator(epochs_vis_left_av.info, 
                                                               fwd_left, 
                                                               noise_cov_left, 
                                                               loose=0.0, #Value that weights the source variances of the dipole components that are parallel (tangential) to the cortical surface. Can be:
                                                                          #float between 0 and 1 (inclusive)
                                                                          #If 0, then the solution is computed with fixed orientation. If 1, it corresponds to free orientations.
                                                               depth=0.8, # How to weight (or normalize) the forward using a depth prior
                                                               rank='full')
    
'Compute inverse solution'
method = 'sLORETA'  # or MNE, dSPM, sLORETA
snr = 5.
lambda2 = 1. / snr ** 2

stc = mne.minimum_norm.apply_inverse(epochs_vis_right_av, 
                                     inverse_operator,
                                     lambda2=lambda2,
                                     method=method)

stc_left = mne.minimum_norm.apply_inverse(epochs_vis_left_av, 
                                          inverse_operator_left,
                                          lambda2=lambda2,
                                          method=method)

# plot the STC
stc.plot(initial_time=0.09, 
         hemi='both', 
         show_traces=True, 
         time_viewer=True, 
         #views=['lat', 'med'],
         surface='white', # "sphere" "white" 'inflated'
         subjects_dir=subjects_dir)

stc_left.plot(initial_time=0.09, 
              hemi='both', 
              show_traces=True, 
              time_viewer=True, 
              #views=['lat', 'med'],
             surface='white', # "sphere" "white" 'inflated'
             subjects_dir=subjects_dir)


#%%
'Calculate SNR in source space'
fwd_fixed = mne.make_forward_solution(epochs_vis_right.info, 
                                      trans=trans, 
                                      src=src, 
                                      bem=bem,
                                      meg=False, 
                                      eeg=True, 
                                      mindist=5.0, 
                                      n_jobs=1,
                                      verbose=True)
fwd_fixed = mne.convert_forward_solution(fwd_fixed, surf_ori=True, force_fixed=True)


# fixed orientation for fwd 
snr_stc = stc.estimate_snr(epochs_vis_right.info, fwd_fixed, noise_cov)

# Plot an average SNR across source points over time:
ave = np.mean(snr_stc.data, axis=0)

fig, ax = plt.subplots(layout="constrained")
ax.plot(epochs_vis_right_av.times, ave)
ax.set(xlabel="Time (s)", ylabel="SNR EEG")

# maximum SNR
maxidx = np.argmax(ave)

kwargs = dict(
    initial_time=epochs_vis_right_av.times[maxidx],
    hemi="both",
    views=["lat"],
    subjects_dir=subjects_dir,
    size=(600, 600),
    clim=dict(kind="value", lims=(100, 130, 150)),
    transparent=True,
    colormap="viridis",
)
brain = snr_stc.plot(**kwargs)


#%%

'Compute and apply morth matrix'
fsave_vertices = [s["vertno"] for s in src]
morph = mne.compute_source_morph(stc,
                                 "sample",
                                 "fsaverage",
                                 spacing=fsave_vertices,
                                 smooth=20,
                                 subjects_dir=subjects_dir)
stc = morph.apply(stc)



morph_left = mne.compute_source_morph(stc_left,
                                 "sample",
                                 "fsaverage",
                                 spacing=fsave_vertices,
                                 smooth=20,
                                 subjects_dir=subjects_dir)
stc_left = morph.apply(stc_left)


n_vertices_fsave, n_times = stc.data.shape
tstep = stc.tstep * 1000  # convert to milliseconds


'Compute fake subjects data'
np.random.seed(42)
n_subjects = 10
X1 = np.random.randn(n_vertices_fsave, n_times, n_subjects) * 5
X2 = np.random.randn(n_vertices_fsave, n_times, n_subjects) * 5

X1[:, :, :] += stc.data[:, :, np.newaxis]
# make the activity bigger for the second set of subjects
X2[:, :, :] += stc_left.data[:, :, np.newaxis] * 5

# compare the overall activity levels for each subject
X1 = np.abs(X1)  # only magnitude
X2 = np.abs(X2)  # only magnitude

# X needs to be a list of multi-dimensional array of shape samples × time × space
X1 = np.transpose(X1, [2, 1, 0])
X2 = np.transpose(X2, [2, 1, 0])
X = [X1, X2]

'Compute adjacency matrix'
adjacency = spatial_src_adjacency(src)


'Make clustering'
n_permutations = 100
p_threshold = 0.05
#f_threshold = stats.distributions.f.ppf(1.0 - p_threshold / 2.0, 10 - 1, 10 - 1)
f_threshold = 4.0

F_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_test(
    X,
    adjacency=adjacency,
    n_jobs=None,
    n_permutations=n_permutations,
    threshold=f_threshold,
    buffer_size=None)

# Clusters at p < 0.05 
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]



'Cluster visualization'
# cluster represented as "time point"
stc_all_cluster_vis = summarize_clusters_stc(clu,
                                             p_thresh=0.05,
                                             tstep=tstep, 
                                             vertices=fsave_vertices, 
                                             subject="fsaverage")

brain = stc_all_cluster_vis.plot(
    "fsaverage",
    hemi="both",
    views="lateral",
    subjects_dir=subjects_dir,
    time_label="temporal extent (ms)",
    clim=dict(kind="value", lims=[0, 60, 100]),
    surface='inflated')


#%% 
'''RM ANOVA'''

'Transform to source space'
sample_vertices = [inverse_operator["src"][0]["vertno"], np.array([], int)]

conditions = []
for cond in ["audial/left", "audial/right", "visual/left", "visual/right"]: 
    evoked = epochs[cond].average()
    evoked.resample(30).crop(0.0, None)
    condition = apply_inverse(evoked, inverse_operator, lambda2, method)
    condition.crop(0, None)
    conditions.append(condition)

tmin = conditions[0].tmin
tstep = conditions[0].tstep * 1000  # convert to milliseconds


'Transform to common cortical space'
n_vertices_sample, n_times = conditions[0].lh_data.shape

n_subjects = 10
np.random.seed(42)
X = randn(n_vertices_sample, n_times, n_subjects, 4) * 10
for ii, condition in enumerate(conditions):
    X[:, :, :, ii] += condition.lh_data[:, :, np.newaxis]

fsave_vertices = [src[0]["vertno"], []]
morph_mat = mne.compute_source_morph(
    src=inverse_operator["src"],
    subject_to="fsaverage",
    spacing=fsave_vertices,
    subjects_dir=subjects_dir,
    smooth=20).morph_mat

morph_mat = morph_mat[:, :n_vertices_sample]  # just left hemi from src
n_vertices_fsave = morph_mat.shape[0]
X = X.reshape(n_vertices_sample, n_times * n_subjects * 4)
X = morph_mat.dot(X)  # morph_mat is a sparse matrix
X = X.reshape(n_vertices_fsave, n_times, n_subjects, 4)
X = np.transpose(X, [2, 1, 0, 3])
X = [np.squeeze(x) for x in np.split(X, 4, axis=-1)]


'Prepare function for arbitrary contrast'
factor_levels = [2, 2]
effects = "A:B"

n_times = X[0].shape[1]
n_conditions = 4

def stat_fun(*args):
    rm_anova = f_mway_rm(np.swapaxes(args, 1, 0),
                         factor_levels=factor_levels,
                         effects=effects,
                         return_pvals=False)[0]
    return rm_anova

'Compute clustering statistic'
adjacency = mne.spatial_src_adjacency(src[:1])

pthresh = 0.05
#f_thresh = f_threshold_mway_rm(n_subjects, factor_levels, effects, pthresh)
f_thresh = 4.0
n_permutations = 100  


F_obs, clusters, cluster_p_values, H0 = clu = spatio_temporal_cluster_test(
    X,
    adjacency=adjacency,
    n_jobs=None,
    threshold=f_thresh,
    stat_fun=stat_fun,
    n_permutations=n_permutations,
    buffer_size=None)

good_cluster_inds = np.where(cluster_p_values < 0.05)[0]

'Visualize the clusters'
stc_all_cluster_vis = summarize_clusters_stc(clu, 
                                             tstep=tstep, 
                                             vertices=fsave_vertices, 
                                             subject="fsaverage")

brain = stc_all_cluster_vis.plot(subjects_dir=subjects_dir,
                                 views="par",
                                 time_label="temporal extent (ms)",
                                 clim=dict(kind="value", lims=[0, 1, 40]))
brain.show_view("medial")


'interaction effect by time cources'
inds_t, inds_v = [
    (clusters[cluster_ind]) for ii, cluster_ind in enumerate(good_cluster_inds)][0]  

times = np.arange(X[0].shape[1]) * tstep * 1e3

plt.figure()
colors = ["y", "b", "g", "purple"]
event_ids = ["audial/left", "audial/right", "visual/left", "visual/right"]
for ii, (condition, color, eve_id) in enumerate(zip(X, colors, event_ids)):
    condition = condition[:, :, inds_v]
    mean_tc = condition.mean(axis=2).mean(axis=0)
    std_tc = condition.std(axis=2).std(axis=0)
    plt.plot(times, mean_tc.T, color=color, label=eve_id)
    plt.fill_between(times, mean_tc + std_tc, mean_tc - std_tc, color="gray", alpha=0.5, label="")

ymin, ymax = mean_tc.min() - 5, mean_tc.max() + 5
plt.xlabel("Time (ms)")
plt.ylabel("Activation (F-values)")
plt.xlim(times[[0, -1]])
plt.ylim(ymin, ymax)
plt.fill_betweenx((ymin, ymax), times[inds_t[0]], times[inds_t[-1]], color="orange", alpha=0.3)
plt.legend()
plt.title("Interaction between stimulus-modality and location.")
plt.show()

















