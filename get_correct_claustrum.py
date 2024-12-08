# extract clastrum, insula, and putamen, then save these with corrected claustrum timeseries
#
# fbarrett@jhmi.edu 2024.12.07

import numpy as np
import pandas as pd
from nilearn.input_data import NiftiLabelsMasker
from sklearn.linear_model import LinearRegression
import os

croot = '/Users/fbarret2/git/napp_nii/claustrum/rois'
cLmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'mean--L-claustrum.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
iLmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'L_Ins12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
pLmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'L_Put12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
cRmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'mean--R-claustrum.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
iRmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'R_Ins12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
pRmask = NiftiLabelsMasker(labels_img=os.path.join(croot,'R_Put12Dil_Olap_Min6Dil_MeanBoxed.nii'), standardize=True, verbose=1,memory="nilearn_cache", memory_level=2)
cLmask.fit(); iLmask.fit(); pLmask.fit(); cRmask.fit(); iRmask.fit(); pRmask.fit()

header = ['claustrumL', 'insulaL','putamenL','claustrumR','insulaR','putamenR','residL','residR']

def correct_claustrum(fpath):
    cLdata = cLmask.transform(fpath)
    iLdata = iLmask.transform(fpath)
    pLdata = pLmask.transform(fpath)
    cRdata = cRmask.transform(fpath)
    iRdata = iRmask.transform(fpath)
    pRdata = pRmask.transform(fpath)

    Xl = np.hstack([iLdata, pLdata])
    Xr = np.hstack([iRdata, pRdata])

    regL = LinearRegression().fit(Xl, cLdata)
    regR = LinearRegression().fit(Xr, cRdata)

    residL = cLdata - regL.predict(Xl)
    residR = cRdata - regR.predict(Xr)

    if fpath.endswith('.nii'):
        outfile = fpath.replace('.nii', '_claustrumCorrect.csv') 
    elif fpath.endswith('.nii.gz'):
        outfile = fpath.replace('.nii.gz', '_claustrumCorrect.csv') 

    outdata = pd.DataFrame(np.hstack([cLdata, Xl, cRdata, Xr, residL, residR]),columns=header)
    outdata.to_csv(outfile, index=False)

    print(f'ROIs and corrected claustrum timecourse saved to {outfile}')

    return residL, residR    

