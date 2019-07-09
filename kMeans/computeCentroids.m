function centroids = computeCentroids(X, prev_centroids, memberships, k)
%COMPUTECENTROIDS Computes the centroids for the k clusters by taking the 
%average value of the data points in the cluster.
%   centroids = COMPUTECENTROIDS(X, memberships, k) returns the new centroids 
%   by computing the means of the data points assigned to each centroid. 
%
%   Parameters
%     X           - The dataset, with one sample per row.
%     memberships - The index of the centroid that the corresponding data point
%                   in X belongs to (a value in the range 1 - k).
%     k           - The number of clusters.
%
%   Returns
%     A matrix of centroids, with k rows where each row contains a centroid.

% $Author: ChrisMcCormick $    $Date: 2013/08/30 22:00:00 $    $Revision: 1.1 $

% X contains 'm' samples with 'n' dimensions each.
[m n] = size(X);

centroids = zeros(k, n);

% For each centroid...
for (i = 1 : k)
    % If no points are assigned to the centroid, don't move it.
    if (~any(memberships == i))
        centroids(i, :) = prev_centroids(i, :);
    % Otherwise, compute the cluster's new centroid.
    else
        % Select the data points assigned to centroid k.
        points = X((memberships == i), :);

        % Compute the new centroid as the mean of the data points.
        centroids(i, :) = mean(points, 1);    
    end
end

end

