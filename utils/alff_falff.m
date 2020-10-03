function [A,F] = alff_falff(t,fs,lc,hc)

% calculate ALFF and fALFF from a given vector
% 
%   [A,F] = alff_falff(t,fs,lc,hc)
% 
%   t - time course from which to calculate ALFF/fALFF
%   fs - sampling rate of t
%   lc - low-pass filter cutoff
%   hc - high-pass filter cutoff
% 
% adopted from github:Chaogan-Yan/DPABI/DPARSF/Subfunctions/y_alff_falff.m.
% The progenitor function required 4D data. The current script only
% requires a timecourse.
% 
% fbarrett@jhmi.edu 2019.06.20

% Get the frequency index
sampleLength = length(t);
paddedLength = 2^nextpow2(sampleLength);
if (lc >= fs/2) % All high included
    idx_LowCutoff = paddedLength/2 + 1;
else % high cut off, such as freq > 0.01 Hz
    idx_LowCutoff = ceil(lc * paddedLength * 1/fs + 1);
end
if (hc>=fs/2)||(hc==0) % All low pass
    idx_HighCutoff = paddedLength/2 + 1;
else % Low pass, such as freq < 0.08 Hz
    idx_HighCutoff = fix(hc *paddedLength * 1/fs + 1);
end


% Detrend, zero padding, fft
fprintf('\n\t Performing FFT ...');
t = detrend(t);
t = [t;zeros(paddedLength - sampleLength,size(t,2))];
t = 2*abs(fft(t))/sampleLength;

% ALFF AND fALFF
A = mean(t(idx_LowCutoff:idx_HighCutoff,:));
F = sum(t(idx_LowCutoff:idx_HighCutoff,:),1)./sum(t(2:(paddedLength/2+1),:),1);
F(~isfinite(F))=0;
