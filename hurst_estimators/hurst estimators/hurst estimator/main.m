M = csvread('data.csv');

H_aggva     = aggvar(M, 1);
H_peng      = peng(M, 1);
H_higuchi   = higuchi(M, 1);
H_abs       = absval(M, 1);

% You need alot of data for an accurate hurst exponent estimate.
% Finds correlation between different lengths within the data.
% Doesn't add anything by splitting the data    .

% 

% Most agree on 0.66
