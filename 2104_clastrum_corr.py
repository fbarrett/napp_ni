# organize a flat directory into BIDS
#
# fbarrett@jhmi.edu 2024.12.07

import os
# from nilearn import plotting
# matplotlib inline
import pickle
import numpy as np
import pandas as pd
import nilearn.datasets
import itertools            as it
import nilearn.signal       as nsig
import nilearn.connectome   as nconn
import matplotlib.pyplot as plt
from statistics import mode
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

import sys
sys.path.append('/Users/fbarret2/git/napp_nii/python')
from get_correct_claustrum import correct_claustrum

# from nilearn import image
# from nideconv.utils import roi

directory = '/Users/fbarret2/Documents/_data/2104_msit_claustrum'
# croot = '/Users/fbarret2/git/napp_nii/claustrum/rois'

# Fetch the Schaefer atlas
schaefer_atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100,yeo_networks=17)
# atlas_image = list(map(str,schaefer_atlas['maps']))
# atlas_labels = list(map(str,schaefer_atlas['labels']))
# defA_mask = np.where(['DefaultA' in label for label in atlas_labels])
# defB_mask = np.where(['DefaultB' in label for label in atlas_labels])
# defC_mask = np.where(['DefaultC' in label for label in atlas_labels])
# dattnA_mask = np.where(['DorsAttnA' in label for label in atlas_labels])
# dattnB_mask = np.where(['DorsAttnB' in label for label in atlas_labels])
# svaA_mask = np.where(['SalVentAttnA' in label for label in atlas_labels])
# svaB_mask = np.where(['SalVentAttnB' in label for label in atlas_labels])
# contA_mask = np.where(['ContA' in label for label in atlas_labels])
# contB_mask = np.where(['ContB' in label for label in atlas_labels])
# contC_mask = np.where(['ContC' in label for label in atlas_labels])

masker = NiftiLabelsMasker(labels_img=schaefer_atlas.maps, standardize=True, verbose=1,
                           memory="nilearn_cache", memory_level=2)
masker.fit()

confounds = sorted(set(['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04',
                        'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
                        'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
                        'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
                        'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
                        'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
                        'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2'] +
                        [f"motion_outlier{i:0{2}d}" for i in range(0, 10)]))

# cLmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'mean--L-claustrum.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
# iLmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'L_Ins12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
# pLmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'L_Put12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
# cRmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'mean--R-claustrum.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
# iRmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'R_Ins12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
# pRmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'R_Put12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
# cLmask.fit(); iLmask.fit(); pLmask.fit(); cRmask.fit(); iRmask.fit(); pRmask.fit()

correlation_measure = ConnectivityMeasure(kind='correlation')

# collect the files and the confounds
corMtxs = {}
corMtxsSet = {}
for root, dirs, files in os.walk(directory):
    for dir in dirs:
        if dir == 'func':
            subdir = os.path.join(root, dir)
            print(f'---- Searching in directory: {subdir} ----')
            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                for file in subfiles:
                    if os.path.getsize(os.path.join(subdir,file)) == 0:
                        print(f'{file} EMPTY, continuing...')
                        continue
                    elif file.endswith('.nii'):
                        fparts = file.split('_')
                        subid = fparts[0]
                        sesid = fparts[1]
                        stem = os.path.splitext(os.path.basename(file))[0]

                        # get confounds
                        confound_file = os.path.join(subdir,f'{fparts[0]}_{fparts[1]}_{fparts[2]}_{fparts[3]}_desc-confounds_timeseries.tsv')
                        if not os.path.exists(confound_file):
                            print(f'cant find confounds for {file}, SKIPPING')
                            continue
                        data_conf = pd.read_csv(confound_file,sep='\t')
                        dc_vars    = set(data_conf.columns)
                        conf_local = dc_vars.intersection(confounds)
                        data_conf  = data_conf[list(conf_local)]
                        columns_with_na = data_conf.columns[data_conf.isna().any()].tolist()
                        data_conf.drop(columns_with_na, axis=1, inplace=True)

                        # get claustrum file
                        claustrum_file = os.path.join(subdir,f'{stem}_claustrumCorrect.csv')
                        if not os.path.exists(claustrum_file):
                            print(f'cant find claustrum timeseries for {file}, COMPUTING!!')
                            correct_claustrum(os.path.join(subdir,file))
                            # continue
                        data_claus = pd.read_csv(claustrum_file)
                        data_claus_clean = nsig.clean(data_claus.to_numpy(),
                                                      confounds=data_conf,
                                                      detrend=True,
                                                      standardize='zscore_sample')

                        print(f'extracting atlas timeseries for {file} with {confound_file}')
                        time_series = masker.fit_transform(os.path.join(subdir,file), confounds=data_conf)
                        time_series = np.concatenate((time_series,data_claus_clean[:,[6,7]]),axis=1)

                        if not sesid in corMtxs.keys(): corMtxs[sesid] = {}; corMtxsSet[sesid] = [];
                        if not subid in corMtxs[sesid].keys(): corMtxs[sesid][subid] = {};
                        corMtxs[sesid][subid] = correlation_measure.fit_transform([time_series])[0]
                        corMtxsSet[sesid].append(nconn.sym_matrix_to_vec(corMtxs[sesid][subid]))
                        print(f'Added {file} to {sesid}')

# save data to pickles
with open(os.path.join(directory,'2104_corMtxs.pkl'), 'wb') as file:
    pickle.dump(corMtxs, file)
with open(os.path.join(directory,'2104_corMtxsSet.pkl'), 'wb') as file:
    pickle.dump(corMtxsSet, file)

# initial analysis?
meanCorMtxs = dict.fromkeys(corMtxsSet.keys())
diffCorMtxs = dict.fromkeys(corMtxsSet.keys())
sep = ' / '
for ses in corMtxsSet.keys():
    lengths = [len(item) for item in corMtxsSet[ses]]
    pruned_mtx = [x for x in corMtxsSet[ses] if len(x) == mode(lengths)]
    meanCorMtxs[ses] = np.mean(np.arctanh(pruned_mtx),axis=0)
    diffCorMtxs[ses] = {}
    print(f'Mean connectome for {ses} computed')
for combo in it.combinations(corMtxsSet.keys(),2):
    ckey = sep.join(combo)
    diffCorMtxs[ckey] = nconn.vec_to_sym_matrix(np.tanh(meanCorMtxs[combo[0]] - meanCorMtxs[combo[1]]))
    print(f'Difference between {combo[0]} and {combo[1]} computed')

    # plot the connectomes
    figpath = os.path.join(directory,'figures',f'{combo[0]}_{combo[1]}_connectome_diff.pdf')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im1 = axes[0].imshow(nconn.vec_to_sym_matrix(np.tanh(meanCorMtxs[combo[0]])), cmap='coolwarm', vmin=-1, vmax=1)
    axes[0].set_title(f'{combo[0]}')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(nconn.vec_to_sym_matrix(np.tanh(meanCorMtxs[combo[1]])), cmap='coolwarm', vmin=-1, vmax=1)
    axes[1].set_title(f'{combo[1]}')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(diffCorMtxs[ckey], cmap='coolwarm', vmin=-1, vmax=1)
    axes[2].set_title(f'{ckey} Difference')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    fig.suptitle(f'Connectomes, {ckey} Difference')
    plt.savefig(figpath, format="pdf", bbox_inches="tight")
    plt.show()