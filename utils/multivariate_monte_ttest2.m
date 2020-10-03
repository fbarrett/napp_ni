function [t,p,null] = multivariate_monte_ttest2(x,y,varargin)

% monte carlo analysis for groups of paired t-tests
% 
%   [null,pos,p] = multivariate_monte_ttest2(x,y,v,varargin)
% 
%   INPUTS
%       x - N (observation) x M (permutation block) matrix of values paired to y
%       y - N x M matrix of values paired to x
% 
%   OPTIONAL - key/value pairs
%       'niter' - [default: 5000] number of iterations for monte carlo sim
% 
%   OUTPUTS
%       null - null distribution of "t"s generated from monte carlo sim
%       t    - veridical t-value comparing means of x and y
%       p    - p-value of veridical value in the null distribution
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

xs = size(x);
if ~all(xs==size(y)), error('inputs are of unequal size'); end

x(isinf(x)) = NaN; y(isinf(y)) = NaN; % remove INFs

for k=1:niter
  swap = rand(size(x,2),1) < .5;
  tmpx = x; tmpx(:,swap) = y(:,swap);
  tmpy = y; tmpy(:,swap) = x(:,swap);
  
  [~,~,~,stats] = ttest2(tmpx',tmpy');
  
  null(k) = max(abs([stats.tstat]));
end % for k=1:niter

null = sort(null);

[~,~,~,stats] = ttest(x',y');
t = stats.tstat;

for k=1:length(t)
  if isnan(t(k)), p(k) = NaN; continue, end
  
  pos = find(abs(t(k)) < null,1,'first');
  if isempty(pos) || pos == niter
    p(k) = 1/niter;
  else
    p(k) = 1-pos/niter+eps;
  end % if all(t > null
end % for k=1:length(t
