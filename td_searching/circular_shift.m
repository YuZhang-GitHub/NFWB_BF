function [arr_shifted] = circular_shift(arr,scope)
%CIRCULAR_SHIFT Summary of this function goes here
%   Detailed explanation goes here
%   scope should be centered around zero, for example: scope = [-pi, pi];

T = (scope(2)-scope(1))/2;
POS = abs(arr) > T;
arr_sub = arr(POS);
arr_sub_sign = sign(arr_sub);

K = ceil(abs( (arr_sub-arr_sub_sign*T) / (2*T) ));

arr_sub_ = arr_sub - arr_sub_sign .* (K * 2 * T);

arr_shifted = arr;
arr_shifted(POS) = arr_sub_;

end