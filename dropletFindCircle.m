function circlesResults = dropletFindCircle(imagePath, showFigures)
% dropletFindCircle Detect droplets for a single image using tuned multi-scale search.
%   RESULT = dropletFindCircle(IMAGE_PATH) runs imfindcircles on the specified
%   image (full path or current-folder name), applies non-maximum suppression,
%   and returns a 1x1 struct with fields filename, centers, radii, and metric.
%
%   RESULT = dropletFindCircle(IMAGE_PATH, true) also opens a figure showing
%   the detected circles on top of the image (useful while tuning parameters).
%
%   This implementation follows the provided detectEachDroplet routine with
%   small/large radius passes, contrast enhancement, and overlap filtering.
%
%   See also IMFINDCIRCLES.

    if nargin < 1 || isempty(imagePath)
        error('dropletFindCircle requires an image path.');
    end
    if nargin < 2 || isempty(showFigures)
        showFigures = false;
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

    if ~isfile(imagePath)
        error('Image not found: %s', imagePath);
    end

    [~, name, ext] = fileparts(imagePath);
    fileName = strcat(name, ext);

    img = imread(imagePath);

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
    circlesResults = struct('filename', fileName, ...
                            'centers',  centers, ...
                            'radii',    radii, ...
                            'metric',   metric);

    % quick check plot (optional)
    if showFigures
        figure
        imshow(img,[]);
        hold on
        viscircles(centers, radii, 'EdgeColor','b');
        title(sprintf('%s  (%d droplets)', fileName, numel(radii)));
        hold off
    end
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