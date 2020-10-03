% hold variables to support claustrum timecourse extraction

function defs = claustrum_tc_defs(defs)

% fbarrett@jhmi.edu 2018.12.07

defs.paths.root = fullfile(fileparts(which('claustrum_tc_defs')),'rois');
if ~exist(defs.paths.root,'dir')
  error('claustrum ROIs not found!?\n');
end % if ~exist(defs.paths.root,'dir

defs.paths.insulaL = fullfile(defs.paths.root,'L_Ins12Dil_Olap_Min6Dil_MeanBoxed.nii');
defs.paths.insulaR = fullfile(defs.paths.root,'R_Ins12Dil_Olap_Min6Dil_MeanBoxed.nii');
defs.paths.putamenL = fullfile(defs.paths.root,'L_Put12Dil_Olap_Min6Dil_MeanBoxed.nii');
defs.paths.putamenR = fullfile(defs.paths.root,'R_Put12Dil_Olap_Min6Dil_MeanBoxed.nii');
defs.paths.white = fullfile(defs.paths.root,'eeewhite.nii');
defs.paths.claustrumL = fullfile(defs.paths.root,'mean--L-claustrum.nii');
defs.paths.claustrumR = fullfile(defs.paths.root,'mean--R-claustrum.nii');

defs.confounds = {defs.paths.insulaL,defs.paths.putamenL,...
    defs.paths.insulaR,defs.paths.putamenR};
