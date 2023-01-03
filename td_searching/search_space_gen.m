function [search_tuple] = search_space_gen(D)
%SEARCH_SPACE_GEN Summary of this function goes here
%   Detailed explanation goes here

a_x_set = linspace(0, 2, 10);
a_y_set = cell(1, length(a_x_set));

num_of_a_y = 0;
for i = 1:length(a_x_set)
    a_y_set{i} = unique(linspace(-D/2*a_x_set(i), D/2*a_x_set(i), 2*i));
    num_of_a_y = num_of_a_y + length(a_y_set{i});
end

for i = 1:length(a_x_set)
    for j = 1:length(a_y_set{i})
%         scatter(a_x_set(i), a_y_set{i}(j))
%         hold on
    end
end

b_set = linspace(-D, D, 5);

search_tuple = zeros(num_of_a_y*length(b_set), 3);

count = 1;
for i = 1:length(a_x_set)
    for j = 1:length(a_y_set{i})
        for k = 1:length(b_set)
            search_tuple(count, :) = [a_x_set(i), a_y_set{i}(j), b_set(k)];
            count = count + 1;
        end
    end
end

end

