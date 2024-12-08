# walk directory, create structure, populate structure
#
# fbarrett@jhmi.edu 2024.12.07

import os

def bidsify_flat_directory(directory):

    for root, dirs, files in os.walk(directory,topdown=False):
        for file in files:
            if os.path.getsize(os.path.join(root, file)) == 0:
                print(f'{file} EMPTY, continuing...')
            else:
                fparts = file.split('_')
                subid = fparts[0]
                sesid = fparts[1]

                subdir = os.path.join(directory, f'sub-{subid}', f'ses-{sesid}')
                if 'task' in file:
                    subdir = os.path.join(subdir, 'func')
                print(f'Processing {subdir}')
                if not os.path.exists(subdir):
                    print(f'Creating {subdir}')
                    os.makedirs(subdir)
                os.rename(os.path.join(root, file), os.path.join(subdir, file)) # move file to subdir

