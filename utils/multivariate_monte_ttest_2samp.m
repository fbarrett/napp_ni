function [t,p,null] = multivariate_monte_ttest2(x,y,varargin)

% monte carlo analysis for groups of two-sample t-tests
% 
%   [null,pos,p] = multivariate_monte_ttest2(x,y,v,varargin)
% 
%   INPUTS
%       x - Nx2 matrix of dummy coded variables indicating group assignment
%       y - NxM matrix of values paired to x
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

if size(x,2)~=2, error('''x'' must be Nx2'); end

% remove rows with no membership
rmrows = ~sum(x,2);
x(rmrows,:) = [];
y(rmrows,:) = [];

if ~all(sum(x,2) == 1)
  error('''x'' mis-specified: each row must belong to only one group');
end % if ~all(sum(x,2

if ~(size(x,1) == size(y,1)), error('inputs must have same number of rows'); end

y(isinf(y)) = NaN; % remove INFs

for k=1:niter
  swap = rand(size(x,1),1) < .5;
  tmpx = x; tmpx(swap,:) = fliplr(tmpx(swap,:));
  while ~all(sum(tmpx)>1) % make sure more than 1 vol per tmp group=;-e
    swap = rand(size(x,1),1) < .5;
    tmpx = x; tmpx(swap,:) = fliplr(tmpx(swap,:));
  end % while ~all(sum(tmpx
  
  [~,~,~,stats] = ttest2(y(find(tmpx(:,1)),:),y(find(tmpx(:,2)),:));
  
  null(k) = max(abs([stats.tstat]));
end % for k=1:niter

null = sort(null);

[~,~,~,stats] = ttest2(y(find(x(:,1)),:),y(find(x(:,2)),:));
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