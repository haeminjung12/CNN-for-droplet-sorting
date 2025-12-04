function droplet_visualizer(imageDir)
%DROPLET_VISUALIZER Display labeled droplets over each image with color coding.
%
%   DROPLET_VISUALIZER(IMAGE_DIR) loads droplet_labels.mat from IMAGE_DIR
%   (produced by droplet_labeler) and shows each image with all detected
%   droplets drawn as colored bounding boxes. Use Prev/Next to move between
%   images; a dropdown lets you jump directly to a file. A legend on the
%   figure explains the colors for Empty, Single, >2, IgnoreImage, and
%   unlabeled droplets.
%
%   If IMAGE_DIR is omitted, you will be prompted to choose a folder.
%
%   Example
%   -------
%       droplet_visualizer('path/to/images');
%
%   Requires: Image Processing Toolbox.

    if nargin < 1 || isempty(imageDir)
        imageDir = uigetdir(pwd, 'Select image folder');
        if imageDir == 0
            return;
        end
    end

    labelPath = fullfile(imageDir, 'droplet_labels.mat');
    if ~exist(labelPath, 'file')
        error('droplet_labels.mat not found in %s. Run droplet_labeler first.', imageDir);
    end

    S = load(labelPath, 'dropletTable');
    if ~isfield(S, 'dropletTable') || ~istable(S.dropletTable)
        error('droplet_labels.mat does not contain dropletTable.');
    end

    tbl = S.dropletTable;
    if isempty(tbl)
        error('dropletTable is empty.');
    end

    % Normalize label types
    if isstring(tbl.Label)
        tbl.Label = cellstr(tbl.Label);
    elseif iscategorical(tbl.Label)
        tbl.Label = cellstr(string(tbl.Label));
    end
    if isstring(tbl.ImageName)
        tbl.ImageName = cellstr(tbl.ImageName);
    end

    % Unique images in table (respect stored order)
    [imageNames, ~, imageOrder] = unique(tbl.ImageName, 'stable');
    numImages = numel(imageNames);
    currentImageIdx = 1;

    % Color map for labels
    palette = struct('Empty', [0.2 0.6 0.9], ...
                     'Single', [0.3 0.7 0.3], ...
                     'gt2', [0.9 0.3 0.3], ...
                     'IgnoreImage', [0.5 0.5 0.5], ...
                     'Unlabeled', [0.95 0.75 0.1]);

    fig = figure('Name', 'Droplet Visualizer', ...
                 'NumberTitle', 'off', ...
                 'MenuBar', 'none', ...
                 'ToolBar', 'none', ...
                 'Units', 'normalized', ...
                 'Position', [0.1, 0.1, 0.8, 0.8]);
    ax = axes('Parent', fig, 'Position', [0.05, 0.15, 0.9, 0.8]);
    title(ax, '');

    ui.prevBtn = uicontrol(fig, 'Style', 'pushbutton', ...
                           'String', 'Prev Image', ...
                           'Units', 'normalized', ...
                           'Position', [0.05, 0.05, 0.12, 0.05], ...
                           'Callback', @(~,~) move_image(-1));
    ui.nextBtn = uicontrol(fig, 'Style', 'pushbutton', ...
                           'String', 'Next Image', ...
                           'Units', 'normalized', ...
                           'Position', [0.18, 0.05, 0.12, 0.05], ...
                           'Callback', @(~,~) move_image(1));
    ui.popup = uicontrol(fig, 'Style', 'popupmenu', ...
                         'String', imageNames, ...
                         'Units', 'normalized', ...
                         'Position', [0.32, 0.05, 0.30, 0.05], ...
                         'Callback', @(src, ~) jump_to(src.Value));

    % Legend axes (text-based legend that honors colors)
    ui.legendAx = axes('Parent', fig, 'Position', [0.65, 0.01, 0.30, 0.10]);
    axis(ui.legendAx, 'off');

    update_display();

    %---------------- nested callbacks ----------------%
    function move_image(delta)
        currentImageIdx = currentImageIdx + delta;
        if currentImageIdx < 1
            currentImageIdx = 1;
        elseif currentImageIdx > numImages
            currentImageIdx = numImages;
        end
        set(ui.popup, 'Value', currentImageIdx);
        update_display();
    end

    function jump_to(idx)
        currentImageIdx = idx;
        update_display();
    end

    function update_display()
        imageName = imageNames{currentImageIdx};
        fullPath = fullfile(imageDir, imageName);
        if ~isfile(fullPath)
            cla(ax);
            title(ax, sprintf('Missing image file: %s', imageName), 'Parent', ax);
            return;
        end
        img = imread(fullPath);

        mask = imageOrder == currentImageIdx;
        rows = tbl(mask, :);

        cla(ax);
        imshow(img, 'Parent', ax);
        hold(ax, 'on');

        for k = 1:height(rows)
            bbox = rows.BoundingBox(k, :);
            lbl = rows.Label{k};
            if isempty(lbl)
                lbl = 'Unlabeled';
            end
            if strcmpi(lbl, '>2')
                color = palette.gt2;
            elseif isfield(palette, lbl)
                color = palette.(lbl);
            else
                color = palette.Unlabeled;
            end

            rectangle(ax, 'Position', bbox, 'EdgeColor', color, 'LineWidth', 1.5);
            text(ax, bbox(1), bbox(2) - 6, sprintf('#%d %s', rows.DropletID(k), lbl), ...
                 'Color', color, 'FontWeight', 'bold', 'BackgroundColor', 'k', ...
                 'Margin', 1, 'Parent', ax, 'FontSize',3);
        end
        hold(ax, 'off');

        status = sprintf('Image %d / %d: %s  |  Droplets: %d', ...
                         currentImageIdx, numImages, imageName, height(rows));
        title(ax, status);

        draw_legend();
    end

    function draw_legend()
        if ~isgraphics(ui.legendAx)
            return;
        end
        cla(ui.legendAx);
        axis(ui.legendAx, 'off');
        labels = {'Empty', 'Single', '>2', 'IgnoreImage', 'Unlabeled'};
        colors = {palette.Empty, palette.Single, palette.gt2, palette.IgnoreImage, palette.Unlabeled};
        y = linspace(0.8, 0.1, numel(labels));
        for i = 1:numel(labels)
            text(ui.legendAx, 0.05, y(i), labels{i}, 'Color', colors{i}, ...
                'FontWeight', 'bold', 'FontSize', 10, 'Units', 'normalized');
        end
        title(ui.legendAx, 'Legend', 'FontWeight', 'normal', 'FontSize', 10);
    end
end