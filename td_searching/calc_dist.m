function [dist_vec] = calc_dist(single_point,ant_pos)
%CALC_DIST Summary of this function goes here
%   Detailed explanation goes here

num_of_pts = size(ant_pos, 1);

dist_vec = zeros(num_of_pts, 1);

for ii = 1:num_of_pts
    dist_vec(ii) = sqrt((single_point(1)-ant_pos(ii, 1))^2 + (single_point(2)-ant_pos(ii, 2))^2);
end

end

