function [status] = parsave_matlabbatch(fpath,matlabbatch)

% supports saving matlabbatch variables to disk from within a parfor loop
% 
% fbarrett@jhmi.edu 2020.03.17

save(fpath,'matlabbatch')