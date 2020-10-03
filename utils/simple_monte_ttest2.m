function [t,p,null] = simple_monte_ttest2(x,y,varargin)

% simple monte carlo analysis for paired t-test
% 
%   [t,p,null] = simple_monte(x,y,v,varargin)
% 
%   INPUTS
%       x - Nx1 vector of values paired to y
%       y - Nx1 vector of values paired to x
% 
%   OPTIONAL - key/value pairs
%       'niter' - [default: 5000] number of iterations for monte carlo sim
% 
%   OUTPUTS
%       t    - veridical t-value comparing means of x and y
%       p    - p-value of veridical value in the null distribution
%       null - null distribution of "t"s generated from monte carlo sim
% 
% fbarrett@jhmi.edu 2019.04.16

[t,p,null] = deal([]);

if nargin > 3
  for k=1:2:length(varargin)
    switch varargin{k}
      case 'niter'
        niter = varargin{k+1};
      otherwise
        warning('variable not recognized (%s)\n',varargin{k});
    end % switch
  end % for k
end % if nargin > 3

if ~exist('niter','var'), niter = 5000; end

xl = length(x);
if xl~=length(y), error('input vectors are of unequal length'); end

x(isinf(x)) = NaN; y(isinf(y)) = NaN; % remove INFs

for k=1:niter
  swap = rand(xl,1) < .5;
  tmpx = x; tmpx(swap) = y(swap);
  tmpy = y; tmpy(swap) = x(swap);
  
  [~,~,~,stats] = ttest2(tmpx,tmpy);
  
  null(k) = stats.tstat;
end % for k=1:niter

null = sort(null);

[~,~,~,stats] = ttest(x,y);
t = stats.tstat;

if isnan(t), p = NaN; return, end

pos = find(t < null,1,'first');
if t > 0
  if all(t > null)
    p = 1/niter;
  else
    p = 1-pos/niter+eps;
  end % if all(t > null
elseif t < 0
  if all(t < null)
    p = 1/niter;
  else
    p = pos/niter+eps;
  end % if all(t < null
end % if t > 0
