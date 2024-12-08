# clean behavior file for MSIT in 2104 study, create conditiions/regressors
#
# fbarrett@jhmi.edu 2024.12.07

import os
import json
import numpy as np
import pandas as pd
from statistics import mode, mean, median
import matplotlib.pyplot as plt

# set initial variables
rootpath = '/Users/fbarret2/Documents/_data/MSIT_Data_beh_preprocessed/subject_data'
savepath = '/Users/fbarret2/Documents/_data/2104_msit_claustrum'
search_strings = ['Keypress','Stims: text','Waiting_Text: autoDraw']
block_difficult = ['Stims: text = \'112\'','Stims: text = \'131\'',
                   'Stims: text = \'211\'','Stims: text = \'212\'','Stims: text = \'221\'','Stims: text = \'232\'','Stims: text = \'233\'',
                   'Stims: text = \'311\'','Stims: text = \'313\'','Stims: text = \'322\'','Stims: text = \'331\'','Stims: text = \'332\'']
block_easy = ['Stims: text = \'100\'','Stims: text = \'020\'','Stims: text = \'003\'']
block_limit = 3.5
events_dict = dict()

# iterate over all subjects/sessions, get/process behavioral data
for root, dirs, files in os.walk(rootpath,topdown=False):
    for file in files:
        if file.endswith('.log') and 'MSITbuilder' in file:
            fpath = os.path.join(root, file)
            fPathParts = fpath.split('/')
            fFileParts = file.split('_')

            if os.path.getsize(fpath) == 0:
                print(f'{file} EMPTY, continuing...')
                continue
            else:
                print(f'Processing {fpath}')
            
                # Load the CSV file
                data = pd.read_csv(fpath, sep='\t', header=None, names=['time', 'type', 'content'])
                filtered_data = pd.DataFrame(columns=['time', 'type', 'content'])
                for search_string in search_strings:
                    loc_filtered = data[data['content'].str.contains(search_string, na=False)]
                    filtered_data = pd.concat([filtered_data, loc_filtered])
                filtered_data = filtered_data.sort_values(by='time')
                scan_start = filtered_data['content'].str.contains('Keypress: 5', na=False)
                first_row = filtered_data[scan_start].index[0]
                filtered_data = filtered_data.iloc[first_row:]
                filtered_data = filtered_data.reset_index(drop=True)

                # correct the timing
                filtered_data['time'] = filtered_data['time'] - filtered_data['time'].min()

                # get the conditions
                easy_trials = pd.DataFrame(columns=['time', 'type', 'content'])
                for search_string in block_easy:
                    loc_filtered = filtered_data[filtered_data['content'].str.contains(search_string, na=False)]
                    easy_trials = pd.concat([easy_trials, loc_filtered])
                easy_trials = easy_trials.sort_values(by='time')
                easy_trials = easy_trials.reset_index(drop=True)

                diff_trials = pd.DataFrame(columns=['time', 'type', 'content'])
                for search_string in block_difficult:
                    loc_filtered = filtered_data[filtered_data['content'].str.contains(search_string, na=False)]
                    diff_trials = pd.concat([diff_trials, loc_filtered])
                diff_trials = diff_trials.sort_values(by='time')
                diff_trials = diff_trials.reset_index(drop=True)

                # find blocks of trials
                easy_diff = np.diff(easy_trials['time'].to_numpy())
                easy_block_idxs = np.sort(np.append(np.where(easy_diff > block_limit)[0],[0]))
                easy_conditions = pd.DataFrame.from_dict({'onset': easy_trials['time'][easy_block_idxs].to_numpy(),
                                   'duration': np.repeat(3,len(easy_block_idxs)),
                                   'trial_type': 'easy'})
                easy_fname = f'{fPathParts[-3]}_{fPathParts[-2]}_task-msit_run-{fFileParts[2]}_easy.tsv'
                easy_fpath = os.path.join(savepath, fPathParts[-3], fPathParts[-2],'func')
                if not os.path.exists(easy_fpath):
                    os.makedirs(easy_fpath, exist_ok=True)
                easy_conditions.to_csv(os.path.join(easy_fpath,easy_fname), sep='\t', index=False) 
                                
                diff_diff = np.diff(diff_trials['time'].to_numpy())
                diff_block_idxs = np.sort(np.append(np.where(diff_diff > block_limit)[0],[0]))
                diff_conditions = pd.DataFrame.from_dict({'onset': diff_trials['time'][diff_block_idxs].to_numpy(),
                                   'duration': np.repeat(3,len(diff_block_idxs)),
                                   'trial_type': 'difficult'})
                diff_fname = f'{fPathParts[-3]}_{fPathParts[-2]}_task-msit_run-{fFileParts[2]}_diff.tsv'
                diff_fpath = os.path.join(savepath, fPathParts[-3], fPathParts[-2],'func')
                diff_conditions.to_csv(os.path.join(diff_fpath, diff_fname), sep='\t', index=False) 







# plt.figure(figsize=(10,5)); plt.plot(filtered_data['time'],label='time'); plt.show()
