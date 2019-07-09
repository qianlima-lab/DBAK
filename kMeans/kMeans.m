function [memberships,centroids] = kMeans(X, initial_centroids, max_iters)
%KMEANS Run the k-means clustering algorithm on the data set X.
%   [centroids, memberships] = KMEANS(X, initial_centroids, max_iters)
%   Runs the k-means algorithm on the dataset X where 'k' is given by the
%   number of initial centroids in 'initial_centroids'
%
%   This function will test for convergence and stop when the centroids don't
%   change from one iteration to the next. It will also break after 'max_iters'
%   iterations.  
%
%   The initial centroids should all be unique. They are typically taken 
%   randomly from the data set. See the 'kMeansInitiCentroids' function for
%   selecting random, unique points from X as your initial centroids. Note that
%   the choice of initial centroids will affect the final clusters. To get
%   repeatable results from k-means, you need to use the same initial 
%   centroids.
%
%   Parameters
%     X                 - The dataset, with one example per row.
%     initial_centroids - The initial centroids to use, one per row (there
%                         should be 'k' rows).
%     max_iters         - The maximum number of iterations to run (k-means will
%                         stop sooner if it converges).
%   Returns
%     centroids    -  A k x n matrix of centroids, where n is the number of 
%                    dimensions in the data points in X.
%     memberships  - A column vector containing the index of the assigned 
%                    cluster (a value between 1 - k) for each corresponding 
%                    data point in X.

% $Author: ChrisMcCormick $    $Date: 2014/04/08 22:00:00 $    $Revision: 1.2 $

% Get 'k' from the size of 'initial_centroids'.
k = size(initial_centroids, 1);

centroids = initial_centroids;
prevCentroids = centroids;

% Run K-Means
for (i = 1 : max_iters)
    
    % Output progress
    %fprintf('K-Means iteration %d / %d...\n', i, max_iters);
    %fflush(stdout);
    
    % For each example in X, assign it to the closest centroid
    memberships = findClosestCentroids(X, centroids);
        
    % Given the memberships, compute new centroids
    centroids = computeCentroids(X, centroids, memberships, k);
    
    % Check for convergence. If the centroids haven't changed since
    % last iteration, we've converged.
    if (prevCentroids == centroids)
        %fprintf("  Stopping after %d iterations.\n", i);
        %if exist('OCTAVE_VERSION') fflush(stdout); end;
        break;
    end

    % Update the 'previous' centroids.
    prevCentroids = centroids;
end

end

