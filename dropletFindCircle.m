function circlesResults = dropletFindCircle(folder)
% dropletFindCircle Detect droplets using tuned multi-scale circle search.
%   circlesResults = dropletFindCircle(folder) scans the specified FOLDER
%   for .tif/.tiff images, runs imfindcircles at two radius bands, applies
%   non-maximum suppression, and returns a struct array with fields:
%   filename, centers, radii, and metric. Results are also saved to
%   circlesResults.mat in the same folder.
%
%   This implementation follows the provided detectEachDroplet routine with
%   small/large radius passes, contrast enhancement, and overlap filtering.
%
%   See also IMFINDCIRCLES.

    if nargin < 1 || isempty(folder)
        folder = pwd;
    end

    % ---------- PARAMETERS TO TUNE ----------
    rSmall      = [25 55];     % pixel range small droplets
    rLarge      = [55 110];    % pixel range large droplets
    sensSmall   = 0.85;        % lower -> fewer but cleaner
    sensLarge   = 0.90;
    edgeThr     = 0.15;        % higher -> fewer false circles
    metricThr   = 0.25;        % remove weak detections
    overlapFrac = 0.6;         % >60 % overlap = same droplet
    radiusDiff  = 0.3;         % radii within 30 % = same droplet
    % ----------------------------------------

    imageFiles = dir(fullfile(folder,'*.tif*'));
    N = numel(imageFiles);

    circlesResults = repmat(struct('filename','','centers',[], ...
                                   'radii',[],'metric',[]), N, 1);

    for i = 1:N
        img = imread(fullfile(folder, imageFiles(i).name));

        % grayscale
        if ndims(img) == 3
            grayImg = rgb2gray(img);
        else
            grayImg = img;
        end
        I = im2double(grayImg);

        % light smoothing and contrast
        I = imgaussfilt(I, 2);
        I = adapthisteq(I, 'ClipLimit', 0.01);

        % droplets have dark rims on brighter background
        [c1, r1, m1] = imfindcircles(I, rSmall, ...
            'Sensitivity',   sensSmall, ...
            'EdgeThreshold', edgeThr, ...
            'ObjectPolarity','dark', ...
            'Method','TwoStage');

        [c2, r2, m2] = imfindcircles(I, rLarge, ...
            'Sensitivity',   sensLarge, ...
            'EdgeThreshold', edgeThr, ...
            'ObjectPolarity','dark', ...
            'Method','TwoStage');

        centers = [c1; c2];
        radii   = [r1; r2];
        metric  = [m1; m2];

        % basic filtering
        keep = metric > metricThr;
        centers = centers(keep,:);
        radii   = radii(keep);
        metric  = metric(keep);

        % remove circles too close to image border
        [h,w] = size(I);
        margin = 3;
        inFrame = centers(:,1) - radii > margin & ...
                  centers(:,1) + radii < w - margin & ...
                  centers(:,2) - radii > margin & ...
                  centers(:,2) + radii < h - margin;
        centers = centers(inFrame,:);
        radii   = radii(inFrame);
        metric  = metric(inFrame);

        % non-max suppression: keep one circle per droplet
        if ~isempty(radii)
            keep = suppressOverlaps(centers, radii, metric, ...
                                    overlapFrac, radiusDiff);
            centers = centers(keep,:);
            radii   = radii(keep);
            metric  = metric(keep);
        end

        % store
        circlesResults(i).filename = imageFiles(i).name;
        circlesResults(i).centers  = centers;
        circlesResults(i).radii    = radii;
        circlesResults(i).metric   = metric;

        % quick check plot (optional)
        figure
        imshow(img,[]);
        hold on
        viscircles(centers, radii, 'EdgeColor','b');
        title(sprintf('%s  (%d droplets)', imageFiles(i).name, numel(radii)));
        hold off
    end

    % save for reuse
    save(fullfile(folder, 'circlesResults.mat'), 'circlesResults');
end

% --------- helper: non-maximum suppression for circles ----------
function keep = suppressOverlaps(centers, radii, metric, overlapFrac, radiusDiff)

    n = numel(radii);
    keep    = false(n,1);
    discard = false(n,1);

    % process from strongest metric to weakest
    [~, order] = sort(metric, 'descend');

    for k = 1:n
        i = order(k);
        if discard(i)
            continue
        end

        keep(i) = true;

        % distance from current circle to all others
        dx = centers(:,1) - centers(i,1);
        dy = centers(:,2) - centers(i,2);
        d  = hypot(dx, dy);

        % "same droplet" if centers close and radii similar
        similarR = abs(radii - radii(i)) < radiusDiff * radii(i);
        overlap  = d < overlapFrac * radii(i) & similarR;

        overlap(i) = false;
        discard(overlap) = true;
    end
end
