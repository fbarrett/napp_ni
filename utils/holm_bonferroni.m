function H = holm_bonferroni(pvals,alpha)

% apply Holm-Bonferroni correction to a set of p-values
% 
%   H = holm_bonferroni(pvals,alpha)
% 
%   pvals - vector of pvals (doesn't need to be sorted)
%   alpha - critical P (e.g. 0.05)
% 
%   H - binary vector indicating significance of values of pvals
% 
% fbarrett@jhmi.edu 2019.05.30

if nargin < 2, alpha = 0.05; end

n = length(pvals);
H = zeros(1,n);
[i,j] = sort(pvals);

for k=1:n
  if i(k) < alpha/(n-(k-1))
    H(j(k)) = 1;
  else
    return
  end
end % for k