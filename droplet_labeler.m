function droplet_labeler(imageDir)
%DROPLET_LABELER Interactive GUI to label droplets in microscopy images.
%
%   DROPLET_LABELER starts a GUI that loads all images in a folder, detects
%   droplets in each image with dropletFindCircle (imfindcircles), crops them,
%   and lets the user annotate each droplet as "Empty", "Single", or ">2".
%   Results are saved to a CSV file (droplet_labels.csv) in the selected
%   folder and to a MAT file so you can resume the session later.
%
%   DROPLET_LABELER(IMAGE_DIR) uses the specified directory instead of
%   prompting the user.
%
%   Notes
%   -----
%   * Droplet detection delegates to dropletFindCircle so you can reuse the
%     same parameters you tune there (radius range, sensitivity, etc.).
%   * The GUI highlights the current droplet on the full image and shows a
%     cropped zoom for easier labeling.
%   * Use the "Ignore Image" button when a frame should be excluded entirely.
%   * Use the "Save & Quit" button to write labels to disk and close the GUI.
%
%   Example
%   -------
%       droplet_labeler('path/to/images');
%
%   Requires: Image Processing Toolbox.

    if nargin < 1 || isempty(imageDir)
        imageDir = uigetdir(pwd, 'Select image folder');
        if imageDir == 0
            return;
        end
    end

    circlesPath = fullfile(imageDir, 'circlesResults.mat');
    if exist(circlesPath, 'file')
        S = load(circlesPath, 'circlesResults');
        circlesResults = S.circlesResults;
    else
        circlesResults = dropletFindCircle(imageDir);
    end
    if isempty(circlesResults)
        error('No droplets detected in %s. Adjust dropletFindCircle parameters and retry.', imageDir);
    end

    % Map filenames to detections
    fileNames = {circlesResults.filename}';
    uniqueFiles = unique(fileNames, 'stable');
    images = cell(numel(uniqueFiles), 1);
    dropletTable = table();
    for k = 1:numel(uniqueFiles)
        filePath = fullfile(imageDir, uniqueFiles{k});
        images{k} = imread(filePath);

        % Find detection record for this file
        recIdx = find(strcmp(fileNames, uniqueFiles{k}), 1, 'first');
        centers = circlesResults(recIdx).centers;
        radii = circlesResults(recIdx).radii;
        n = numel(radii);

        if n == 0
            continue;
        end

        bboxes = [centers(:, 1) - radii, centers(:, 2) - radii, 2 * radii, 2 * radii];
        dropletIds = (1:n)';
        dropletTable = [dropletTable; table(repmat(k, n, 1), ...
                                            dropletIds, ...
                                            repmat({uniqueFiles{k}}, n, 1), ...
                                            bboxes, ...
                                            repmat(false, n, 1), ...
                                            repmat({''}, n, 1), ...
                                            'VariableNames', {'ImageIndex', 'DropletID', 'ImageName', 'BoundingBox', 'ImageIgnored', 'Label'})]; %#ok<AGROW>
    end

    if isempty(dropletTable)
        error('Droplet detection produced no bounding boxes.');
    end

    totalDroplets = height(dropletTable);
    currentIdx = 1;
    ui = build_gui();
    update_display();

    %---------------- Nested utility functions ----------------%
    function ui = build_gui()
        scr = get(0, 'ScreenSize');
        figW = min(scr(3) * 0.9, 1200);
        figH = min(scr(4) * 0.9, 750);
        ui.fig = figure('Name', 'Droplet Labeler', ...
                        'NumberTitle', 'off', ...
                        'MenuBar', 'none', ...
                        'ToolBar', 'none', ...
                        'Position', [50, scr(4) - figH - 50, figW, figH]);

        % Layout: left shows full image, right shows crop + controls
        leftW = 0.58; rightW = 0.42;
        ui.axFull = axes('Parent', ui.fig, ...
                          'Position', [0.05, 0.1, leftW - 0.08, 0.82]);
        ui.axCrop = axes('Parent', ui.fig, ...
                          'Position', [leftW, 0.45, rightW - 0.08, 0.5]);
        title(ui.axCrop, 'Cropped droplet');

        % Status text
        ui.status = uicontrol(ui.fig, 'Style', 'text', ...
                              'Units', 'normalized', ...
                              'Position', [leftW, 0.36, rightW - 0.08, 0.05], ...
                              'HorizontalAlignment', 'left', ...
                              'FontSize', 11, ...
                              'String', '');

        % Buttons
        btnNames = {'Empty', 'Single', '>2'};
        colors = {[0.85, 0.93, 0.98], [0.86, 0.95, 0.86], [0.98, 0.88, 0.88]};
        for b = 1:numel(btnNames)
            ui.labelBtn(b) = uicontrol(ui.fig, ...
                'Style', 'pushbutton', ...
                'String', btnNames{b}, ...
                'Units', 'normalized', ...
                'Position', [leftW + 0.02, 0.26 - 0.07 * (b-1), 0.15, 0.06], ...
                'BackgroundColor', colors{b}, ...
                'FontSize', 12, ...
                'Callback', @(~,~) set_label(btnNames{b})); %#ok<AGROW>
        end

        ui.prevBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Prev', ...
                               'Units', 'normalized', ...
                               'Position', [leftW + 0.2, 0.05, 0.1, 0.06], ...
                               'FontSize', 11, ...
                               'Callback', @(~,~) move_idx(-1));
        ui.nextBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Next', ...
                               'Units', 'normalized', ...
                               'Position', [leftW + 0.32, 0.05, 0.1, 0.06], ...
                               'FontSize', 11, ...
                               'Callback', @(~,~) move_idx(1));
        ui.skipBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Skip (no label)', ...
                               'Units', 'normalized', ...
                               'Position', [leftW + 0.02, 0.05, 0.15, 0.06], ...
                               'FontSize', 11, ...
                               'Callback', @(~,~) move_idx(1));
        ui.ignoreBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                                 'String', 'Ignore Image', ...
                                 'Units', 'normalized', ...
                                 'Position', [leftW + 0.14, 0.14, 0.14, 0.06], ...
                                 'FontSize', 11, ...
                                 'BackgroundColor', [0.97, 0.92, 0.85], ...
                                 'Callback', @(~,~) ignore_image());

        ui.saveBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Save & Quit', ...
                               'Units', 'normalized', ...
                               'Position', [leftW + 0.45, 0.05, 0.15, 0.06], ...
                               'FontSize', 11, ...
                               'BackgroundColor', [0.9, 0.9, 1], ...
                               'Callback', @(~,~) save_and_close());
    end

    function update_display()
        if currentIdx < 1, currentIdx = 1; end
        if currentIdx > totalDroplets, currentIdx = totalDroplets; end

        row = dropletTable(currentIdx, :);
        img = images{row.ImageIndex};
        bbox = row.BoundingBox;
        cropPad = 6;
        cropRect = expand_rect(bbox, size(img), cropPad);
        crop = imcrop(img, cropRect);

        % Full image with overlay
        axes(ui.axFull); %#ok<LAXES>
        imshow(img, 'Parent', ui.axFull);
        hold(ui.axFull, 'on');
        rectangle(ui.axFull, 'Position', bbox, 'EdgeColor', 'y', 'LineWidth', 1.5);
        text(ui.axFull, bbox(1), bbox(2) - 5, sprintf('#%d', row.DropletID), ...
             'Color', 'y', 'FontWeight', 'bold', 'BackgroundColor', 'k', 'Margin', 1);
        hold(ui.axFull, 'off');

        % Crop display
        imshow(crop, 'Parent', ui.axCrop);

        % Status
        statusStr = sprintf('Droplet %d / %d  |  Image: %s  |  Label: %s%s', ...
                             currentIdx, totalDroplets, row.ImageName, label_or_blank(row.Label), image_ignore_suffix(row));
        set(ui.status, 'String', statusStr);
    end

    function move_idx(delta)
        currentIdx = currentIdx + delta;
        if currentIdx < 1, currentIdx = 1; end
        if currentIdx > totalDroplets, currentIdx = totalDroplets; end
        update_display();
    end

    function set_label(lbl)
        imgIdx = dropletTable.ImageIndex(currentIdx);
        mask = dropletTable.ImageIndex == imgIdx;
        if any(dropletTable.ImageIgnored(mask))
            dropletTable.ImageIgnored(mask) = false;
            ignoreMask = mask & strcmp(dropletTable.Label, 'IgnoreImage');
            dropletTable.Label(ignoreMask) = {''};
        end
        dropletTable.Label{currentIdx} = lbl;
        move_idx(1);
    end

    function ignore_image()
        imgIdx = dropletTable.ImageIndex(currentIdx);
        mask = dropletTable.ImageIndex == imgIdx;
        dropletTable.ImageIgnored(mask) = true;
        dropletTable.Label(mask) = {'IgnoreImage'};

        % jump to first droplet of next image if available
        nextIdx = find(dropletTable.ImageIndex > imgIdx, 1, 'first');
        if ~isempty(nextIdx)
            currentIdx = nextIdx;
        else
            currentIdx = totalDroplets; % stay on last droplet if no next image
        end
        update_display();
    end

    function save_and_close()
        csvPath = fullfile(imageDir, 'droplet_labels.csv');
        matPath = fullfile(imageDir, 'droplet_labels.mat');
        writetable(dropletTable, csvPath);
        save(matPath, 'dropletTable');
        msgbox(sprintf('Saved labels to:\n%s\n%s', csvPath, matPath), 'Saved');
        if isvalid(ui.fig)
            close(ui.fig);
        end
    end
end

function rect = expand_rect(bbox, imgSize, pad)
%EXPAND_RECT Add padding to a bounding box while staying inside the image.
    x1 = max(bbox(1) - pad, 1);
    y1 = max(bbox(2) - pad, 1);
    x2 = min(bbox(1) + bbox(3) + pad, imgSize(2));
    y2 = min(bbox(2) + bbox(4) + pad, imgSize(1));
    rect = [x1, y1, x2 - x1, y2 - y1];
end

function out = label_or_blank(labelCell)
    if isempty(labelCell) || isempty(labelCell{1})
        out = '(none)';
    else
        out = labelCell{1};
    end
end

function suffix = image_ignore_suffix(row)
    if row.ImageIgnored
        suffix = '  |  IGNORED';
    else
        suffix = '';
    end
end
