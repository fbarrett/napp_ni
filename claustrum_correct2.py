# Description: This script fits a GLM to the claustrum timeseries data and saves the residuals as a new CSV file.
#
# fbarrett@jhmi.edu 2024.11.30

### IMPORTS ###
import os
import time
import numpy                as np
import pandas               as pd
import itertools            as it
import nilearn.image        as nimg
import nilearn.signal       as nsig
import statsmodels.api      as sm
import matplotlib.pyplot    as plt
import nilearn.connectome   as nconn
import nilearn.datasets
from nilearn.connectome import ConnectivityMeasure

### VARIABLES ###
# define correlation process object
correlation_measure = ConnectivityMeasure(kind='correlation',standardize='zscore_sample')

# Define the directory containing the CSV files
directory = '/Users/fbarret2/Documents/_data/BOLDconsortium/ROI_timeseries_csvFiles/non-denoised_timeseries'

# Fetch the Schaefer atlas
schaefer_atlas = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100,yeo_networks=17)
atlas_image = list(map(str,schaefer_atlas['maps']))
atlas_labels = list(map(str,schaefer_atlas['labels']))
defA_mask = np.where(['DefaultA' in label for label in atlas_labels])
defB_mask = np.where(['DefaultB' in label for label in atlas_labels])
defC_mask = np.where(['DefaultC' in label for label in atlas_labels])
dattnA_mask = np.where(['DorsAttnA' in label for label in atlas_labels])
dattnB_mask = np.where(['DorsAttnB' in label for label in atlas_labels])
svaA_mask = np.where(['SalVentAttnA' in label for label in atlas_labels])
svaB_mask = np.where(['SalVentAttnB' in label for label in atlas_labels])
contA_mask = np.where(['ContA' in label for label in atlas_labels])
contB_mask = np.where(['ContB' in label for label in atlas_labels])
contC_mask = np.where(['ContC' in label for label in atlas_labels])
schaefer_masks = [defA_mask,defB_mask,defC_mask,dattnA_mask,dattnB_mask,svaA_mask,svaB_mask,contA_mask,contB_mask,contC_mask]
mask_labels = ['DefaultA','DefaultB','DefaultC','DorsAttnA','DorsAttnB','SalVentAttnA','SalVentAttnB','ContA','ContB','ContC']

claus_confounds = sorted(set(['a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04',
                              'trans_x','trans_x_derivative1','trans_x_power2','trans_x_derivative1_power2',
                              'trans_y','trans_y_derivative1','trans_y_power2','trans_y_derivative1_power2',
                              'trans_z','trans_z_derivative1','trans_z_power2','trans_z_derivative1_power2',
                              'rot_x','rot_x_derivative1','rot_x_power2','rot_x_derivative1_power2',
                              'rot_y','rot_y_derivative1','rot_y_power2','rot_y_derivative1_power2',
                              'rot_z','rot_z_derivative1','rot_z_power2','rot_z_derivative1_power2'] +
                              [f"motion_outlier{i:0{2}d}" for i in range(0, 10)]))

atlases = ['schaefer100','schaefer200','schaefer400']
confounds = sorted(set(['global_signal','a_comp_cor_00','a_comp_cor_01','a_comp_cor_02','a_comp_cor_03','a_comp_cor_04'] + [f"aroma_motion_{i:0{2}d}" for i in range(0, 20)]))

tstamp = time.strftime('%Y%m%d')

### MAIN ###
# create corrected claustrum timeseries for each scan
for root, dirs, files in os.walk(directory,topdown=False):
    for file in files:
        if file.endswith('claustrum_TS_noDenosing.csv'):
            fname = os.path.join(root, file)
            if os.path.getsize(fname) == 0:
                print(f'{file} EMPTY, continuing...')
                continue
            else:
                print(f'Processing {file}')
            
                # Load the CSV file
                csv_data = pd.read_csv(fname, header=0, names=['clausL','insL','putL','clausR','insR','putR'])
            
                # Fit the GLM
                glmL = sm.GLM(csv_data['clausL'], sm.add_constant(csv_data[['insL','putL']]))
                glmR = sm.GLM(csv_data['clausR'], sm.add_constant(csv_data[['insR','putR']]))
            
                resultsL = glmL.fit()
                resultsR = glmR.fit()
            
                # Save the residuals
                csv_data['claustrumL_corrected'] = resultsL.resid_response
                csv_data['claustrumR_corrected'] = resultsR.resid_response
            
                # Save the corrected timeseries as a new CSV file
                froot,fext = os.path.splitext(fname)
                output_fname = f'{froot}_corrected{fext}'
                csv_data.to_csv(output_fname, index=False)

# check for all corrected files
for root, dirs, files in os.walk(directory,topdown=False):
    for file in files:
        # if file.endswith('claustrum_TS_noDenosing_corrected.csv'):
        if file.endswith('claustrum_TS_noDenosing.csv'):
            print(f'Found {file}\n')

            # Search all files in each subdirectory, one directory at a time
            for root, dirs, files in os.walk(directory):
                for dir in dirs:
                    subdir = os.path.join(root, dir)
                    print(f'Searching in directory: {subdir}')
                    for subroot, subdirs, subfiles in os.walk(subdir):
                        for subfile in subfiles:
                            print(f'Found file: {subfile}')


# compute claustrum connectome against schaefer atlases
for root, dirs, files in os.walk(directory):
    for dir in dirs:
        #print(f'Searching in directory: {dir}')
        if dir == 'func':
            subdir = os.path.join(root, dir)
            print(f'---- Searching in directory: {subdir} ----')
            fn_claus = ''
            fn_conf  = ''
            fn_atlas = {}
            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                for file in subfiles:
                    if os.path.getsize(os.path.join(subdir,file)) == 0:
                        #print(f'{file} EMPTY, continuing...')
                        continue
                    # elif file.endswith('claustrum_TS_noDenosing_corrected.csv'):
                    elif file.endswith('claustrum_TS_noDenosing.csv'):
                        fn_claus = os.path.join(root, dir, file)
                        #print(f'Found {fn_claus} in {dir}')
                    elif file.endswith('confounds_timeseries.tsv'):
                        fn_conf = os.path.join(root, dir, file)
                        #print(f'Found {fn_conf} in {dir}')
                    else:
                        for atlas in atlases:
                            if atlas in file and not 'corMtx' in file:
                                fn_atlas[atlas] = os.path.join(root, dir, file)
                                #print(f'Found {fn_atlas[atlas]} in {dir}')
                                
                if bool(fn_atlas) and fn_conf and fn_claus:
                    print(f'Found all files for {file} in {dir}')

                    # load claustrum & confounds
                    data_claus       = pd.read_csv(fn_claus, header=0, names=['clausL','insL','putL','clausR','insR','putR'], skip_blank_lines=False)

                    data_conf_raw    = pd.read_csv(fn_conf,sep='\t')
                    dc_vars          = set(data_conf_raw.columns)

                    conf_claus_local = dc_vars.intersection(claus_confounds)
                    data_conf_claus  = data_conf_raw[list(conf_claus_local)]
                    columns_with_na  = data_conf_claus.columns[data_conf_claus.isna().any()].tolist()
                    data_conf_claus.drop(columns_with_na, axis=1, inplace=True)

                    # generate corrected claustrum timecourse
                    conf_regs = np.concatenate((data_claus[['insL','putL']],data_conf_claus[:-1]),axis=1)
                    clausLsig = nsig.clean(data_claus['clausL'].to_numpy(),
                                             confounds=conf_regs,
                                             detrend=False,
                                             standardize=0)
                    conf_regs = np.concatenate((data_claus[['insR','putR']],data_conf_claus[:-1]),axis=1)
                    clausRsig = nsig.clean(data_claus['clausR'].to_numpy(),
                                             confounds=conf_regs,
                                             detrend=False,
                                             standardize=0)
                
                    for atlas in fn_atlas.keys():
                        print(f'Computing connectome for {file} against {atlas} atlas')

                        # Load data frames
                        data_conf_claus['global_signal'] = data_conf_raw['global_signal'].to_numpy()
                        data_atlas = pd.read_csv(fn_atlas[atlas],header=0,names=[f"schaefer_{i:0{3}d}" for i in range(0, int(atlas[-3:]))])
                        data_atlas_clean = nsig.clean(data_atlas.to_numpy(),
                                                      confounds=data_conf_claus[:-1].to_numpy(),
                                                      detrend=True,
                                                      standardize='zscore_sample')

                        data_atlas_clean = np.concatenate((data_atlas_clean,clausLsig.reshape(-1,1),clausRsig.reshape(-1,1)),axis=1)

                        # Compute the correlation matrix, save to disk
                        correlation_matrix = correlation_measure.fit_transform([data_atlas_clean])[0]
                        froot,fext = os.path.splitext(fn_atlas[atlas])
                        corfname = f'{froot}_claus_corMtx{fext}'
                        np.savetxt(corfname, correlation_matrix, delimiter=",")
                        print(f'Connectome saved to {corfname}')

# collect, average, compare connectomes between and across drug conditions
corMtxs = {}
corMtxsSet = {}
for root, dirs, files in os.walk(directory):
    for dir in dirs:
        #print(f'Searching in directory: {dir}')
        if dir == 'func':
            subdir = os.path.join(root, dir)
            print(f'---- Searching in directory: {subdir} ----')
            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                for file in subfiles:
                    if os.path.getsize(os.path.join(subdir,file)) == 0:
                        print(f'{file} EMPTY, continuing...')
                        continue
                    elif file.endswith('corMtx.csv') and 'schaefer100' in file:
                        print(f'Found {file} in {dir}')
                        dirparts = subdir.split('/')
                        src = dirparts[len(directory.split('/'))] # get source study
                        subid = next((s for s in dirparts if 'sub-' in s), None)
                        sesid = next((s for s in dirparts if 'ses-' in s), None)
                        if not src in corMtxs.keys(): corMtxs[src] = {}; corMtxsSet[src] = {}
                        if not sesid in corMtxs[src].keys(): corMtxs[src][sesid] = {}; corMtxsSet[src][sesid] = []
                        if not subid in corMtxs[src][sesid].keys(): corMtxs[src][sesid][subid] = []
                        corMtxs[src][sesid][subid] = pd.read_csv(os.path.join(subdir,file),header=None)
                        corMtxsSet[src][sesid].append(nconn.sym_matrix_to_vec(corMtxs[src][sesid][subid].to_numpy()))
                        print(f'Added {file} to {sesid} in {src}')

meanCorMtxs = dict.fromkeys(corMtxsSet.keys())
diffCorMtxs = dict.fromkeys(corMtxsSet.keys())
sep = ' / '
for src in corMtxsSet.keys():
    meanCorMtxs[src] = dict.fromkeys(corMtxsSet[src].keys())
    diffCorMtxs[src] = {}
    sesids = list(corMtxsSet[src].keys())
    for ses in sesids:
        meanCorMtxs[src][ses] = np.mean(np.arctanh(corMtxsSet[src][ses]),axis=0)
        print(f'Mean connectome for {ses} in {src} computed')
    for combo in it.combinations(sesids,2):
        ckey = sep.join(combo)
        diffCorMtxs[src][ckey] = nconn.vec_to_sym_matrix(np.tanh(meanCorMtxs[src][combo[0]] - meanCorMtxs[src][combo[1]]))
        print(f'Difference between {combo[0]} and {combo[1]} computed for {src}')

        # plot the connectomes
        figpath = os.path.join(directory,'figures',f'{src}_{combo[0]}_{combo[1]}_{tstamp}_connectome_diff.pdf')
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        im1 = axes[0].imshow(nconn.vec_to_sym_matrix(np.tanh(meanCorMtxs[src][combo[0]])), cmap='coolwarm', vmin=-1, vmax=1)
        axes[0].set_title(f'{combo[0]}')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(nconn.vec_to_sym_matrix(np.tanh(meanCorMtxs[src][combo[1]])), cmap='coolwarm', vmin=-1, vmax=1)
        axes[1].set_title(f'{combo[1]}')
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(diffCorMtxs[src][ckey], cmap='coolwarm', vmin=-1, vmax=1)
        axes[2].set_title(f'{ckey} Difference')
        plt.colorbar(im3, ax=axes[2])

        plt.tight_layout()
        fig.suptitle(f'{src} Connectomes, {ckey} Difference')
        plt.savefig(figpath, format="pdf", bbox_inches="tight")
        plt.show()

        # plt.figure(figsize=(10, 10))
        # sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm", center=0)
        # plt.title("Schaefer Atlas 100 Correlation Matrix")
        # plt.show()

##
# plot claustrum connectivity across Yeo networks
clausNetCorrs = {}
for src in meanCorMtxs.keys():
    clausNetCorrs[src] = {}
    sesids = list(meanCorMtxs[src].keys())
    for ses in sesids:
        clausNetCorrs[src][ses] = {}

        # get means for each network, for L/R claustrum
        clausNetCorrs[src][ses]['L'] = []
        clausNetCorrs[src][ses]['R'] = []
        mtx = nconn.vec_to_sym_matrix(meanCorMtxs[src][ses])
        for mask in schaefer_masks:
            clausNetCorrs[src][ses]['L'].append(np.tanh(np.mean(mtx[100,mask])))
            clausNetCorrs[src][ses]['R'].append(np.tanh(np.mean(mtx[101,mask])))

    for combo in it.combinations(sesids,2):
        ckey = sep.join(combo)

        x = np.arange(len(schaefer_masks))  # the label locations
        width = 0.25  # the width of the bars

        ## Left Claustrum: plot mean connectivity with each network
        lConn = np.array([clausNetCorrs[src][combo[0]]['L'],clausNetCorrs[src][combo[1]]['L']])
        figpath = os.path.join(directory,'figures',f'{src}_{combo[0]}_{combo[1]}_{tstamp}_lClaus_Yeo.pdf')

        fig, ax = plt.subplots(layout='constrained')

        multiplier = 0
        for i in range(2):
            offset = width * multiplier
            rects = ax.bar(x + offset, lConn[i,:], width, label=f'{sesids[i]}')
            #ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('correlation')
        ax.set_title(f'{src}: r(left claustrum,Yeo networks)')
        ax.set_xticks(x + width, mask_labels)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(-.4,.4)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45 )
        plt.savefig(figpath, format="pdf", bbox_inches="tight")
        plt.show()

        ## Right Claustrum: plot mean connectivity with each network
        rConn = np.array([clausNetCorrs[src][combo[0]]['R'],clausNetCorrs[src][combo[1]]['R']])
        figpath = os.path.join(directory,'figures',f'{src}_{combo[0]}_{combo[1]}_{tstamp}_rClaus_Yeo.pdf')

        fig, ax = plt.subplots(layout='constrained')

        multiplier = 0
        for i in range(2):
            offset = width * multiplier
            rects = ax.bar(x + offset, rConn[i,:], width, label=f'{sesids[i]}')
            #ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('correlation')
        ax.set_title(f'{src}: r(right claustrum,Yeo networks)')
        ax.set_xticks(x + width, mask_labels)
        ax.legend(loc='upper left', ncols=2)
        ax.set_ylim(-.4,.4)

        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45 )
        plt.savefig(figpath, format="pdf", bbox_inches="tight")
        plt.show()



### UNUSED CODE ###

'''
            if fn_atlas and fn_conf and fn_claus:
                print(f'Found all files for {subdir}')
                for file in subfiles:
                    #print(f'Processing {file}')
                    if os.path.getsize(fname) == 0:
                        print(f'{file} EMPTY, continuing...')
                        continue
                    if file.endswith('confounds_timeseries.tsv'):
                        #print(f'Found confounds file {file}')
                        fn_conf = os.path.join(root, file)
                    elif file.endswith('claustrum_TS_noDenosing_corrected.csv'):
                        #print(f'Found claustrum file {file}')
                        fn_claus = os.path.join(root, file)
                    else:
                        for atlas in atlases:
                            if atlas in file:
                                #print(f'Found atlas {atlas} in {file}')
                                fn_atlas[atlas] = os.path.join(root, file)
'''


'''     some plotting code
# Plot the corrected timeseries for both left and right claustrum
plt.figure(figsize=(10, 5))
plt.plot(csv_data['claustrumL_corrected'], label='ClaustrumL Corrected')
plt.plot(csv_data['claustrumR_corrected'], label='ClaustrumR Corrected')
plt.xlabel('Timepoints')
plt.ylabel('Residuals')
plt.title('Corrected Timeseries for Claustrum')
plt.legend()
plt.show()

# Plot the corrected timeseries for the left claustrum
plt.figure(figsize=(10, 5))
plt.plot(csv_data['claustrumL_corrected','clastrumR_corrected'], label=['ClaustrumL Corrected','ClaustrumR Corrected'])
plt.xlabel('Timepoints')
plt.ylabel('Residuals')
plt.title('Corrected Timeseries for Left Claustrum')
plt.legend()
plt.show()

# Plot the corrected timeseries for the right claustrum
plt.figure(figsize=(10, 5))
plt.plot(csv_data['claustrumR_corrected'], label='ClaustrumR Corrected')
plt.xlabel('Timepoints')
plt.ylabel('Residuals')
plt.title('Corrected Timeseries for Right Claustrum')
plt.legend()
plt.show()
'''

""" # Load the CSV file
fname = '/Users/fbarret2/Library/CloudStorage/GoogleDrive-frederick.barrett@gmail.com/.shortcut-targets-by-id/1FI_5QvPZSlcN2KM2vC9XB24d0tfEfg5U/Metapsycho Meta-analysis/ROI_timeseries_csvFiles/non-denoised_timeseries/Barrett_Psilocybin/sub-barrettpsil0/ses-Session4/sub-barrettpsil0_ses-Session4_claustrum_TS_noDenosing.csv'
csv_data = pd.read_csv(fname,header=0,names=['clausL','insL','putL','clausR','insR','putR'])

# Fit the GLM
glmL = sm.GLM(csv_data['clausL'], sm.add_constant(csv_data[['insL','putL']]))
glmR = sm.GLM(csv_data['clausR'], sm.add_constant(csv_data[['insR','putR']]))

resultsL = glmL.fit()
resultsR = glmR.fit()

# save the residuals
csv_data['claustrumL_corrected'] = resultsL.resid_response
csv_data['claustrumR_corrected'] = resultsR.resid_response

# Save the corrected timeseries as a new CSV file
csv_data.to_csv('claustrum_corrected_timeseries.csv', index=False)
 """