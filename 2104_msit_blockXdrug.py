# block-designed task-based analysis of MSIT in the 2104 study
#
# fbarrett@jhmi.edu 2024.12.08

import os
import json
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix

### VARIABLES ###
rootpath = '/Users/fbarret2/Documents/_data/2104_msit_claustrum'

### MAIN ###
# iterate over all subjects/sessions, get/process behavioral data
for root, dirs, files in os.walk(rootpath):
    for dir in dirs:
        if dir == 'func':
            subdir = os.path.join(root, dir)
            print(f'---- Searching in directory: {subdir} ----')
            for subroot, subdirs, subfiles in os.walk(os.path.join(root, dir)):
                for file in subfiles:
                    if os.path.getsize(os.path.join(subdir,file)) == 0:
                        #print(f'{file} EMPTY, continuing...')
                        continue
                    elif file.endswith('preproc_bold.nii'):
                        fparts = file.split('_')
                        subid = fparts[0]
                        sesid = fparts[1]
                        runid = fparts[3]
                        stem = os.path.splitext(os.path.basename(file))[0]

                        # get confounds
                        fn_conf = os.path.join(root, dir,f'{fparts[0]}_{fparts[1]}_{fparts[2]}_{fparts[3]}_desc-confounds_timeseries.tsv')

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



# Define your block design parameters
block_onsets = [0, 20, 40]
block_durations = [10, 10, 10]

# Create design matrix
design_matrix = make_first_level_design_matrix(
    frame_times,
    events=pd.DataFrame({'trial_type': ['block', 'block', 'block'], 
                         'onset': block_onsets, 
                         'duration': block_durations}),
    hrf_model='spm'
)
