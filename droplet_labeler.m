function droplet_labeler(imageDir)
%DROPLET_LABELER Interactive GUI to label droplets in microscopy images.
%
%   DROPLET_LABELER starts a GUI that opens images one at a time, detects
%   droplets in the current image with dropletFindCircle (imfindcircles),
%   crops them, and lets the user annotate each droplet as "Empty",
%   "Single", or ">2" before moving on to the next image. Detections are
%   cached per image to circlesResults.mat so you can pause and resume without
%   reprocessing the whole folder.
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
    cachedResults = struct('filename', {}, 'centers', {}, 'radii', {}, 'metric', {});
    if exist(circlesPath, 'file')
        S = load(circlesPath, 'circlesResults');
        cachedResults = S.circlesResults;
    end

    imageFiles = dir(fullfile(imageDir, '*.tif*'));
    if isempty(imageFiles)
        error('No .tif/.tiff images found in %s.', imageDir);
    end

    % Track which images have already been expanded into dropletTable
    processedMask = false(numel(imageFiles), 1);

    labelStore = fullfile(imageDir, 'droplet_labels.mat');
    dropletTable = load_existing_table(labelStore, imageFiles);
    if ~isempty(dropletTable)
        processedMask(unique(dropletTable.ImageIndex)) = true;
    end

    circlesResults = cachedResults;
    nextImageIdx = find(~processedMask, 1, 'first');

    % Ensure at least one image is processed before launching the GUI
    while height(dropletTable) == 0 && ~isempty(nextImageIdx)
        add_image_detections(nextImageIdx);
        nextImageIdx = find(~processedMask, 1, 'first');
    end

    % Identify the next image to process after initialization
    nextImageIdx = find(~processedMask, 1, 'first');

    if isempty(dropletTable)
        error('Droplet detection produced no bounding boxes in the available images.');
    end

    totalDroplets = height(dropletTable);
    firstUnlabeled = find(cellfun(@isempty, dropletTable.Label), 1, 'first');
    if isempty(firstUnlabeled)
        currentIdx = 1;
    else
        currentIdx = firstUnlabeled;
    end
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
                          'Position', [leftW, 0.52, rightW - 0.08, 0.42]);
        title(ui.axCrop, 'Cropped droplet');

        % Status text
        ui.status = uicontrol(ui.fig, 'Style', 'text', ...
                              'Units', 'normalized', ...
                              'Position', [leftW, 0.44, rightW - 0.08, 0.08], ...
                              'HorizontalAlignment', 'left', ...
                              'FontSize', 11, ...
                              'String', '');

        % Common button layout helpers
        col1 = leftW + 0.02;
        colSpacing = 0.10;
        col2 = col1 + colSpacing;
        col3 = col2 + colSpacing;
        col4 = col3 + colSpacing;
        wideBtn = 0.18;
        narrowBtn = 0.09;
        imageNavBtn = 0.10;
        rowNav = 0.03;
        rowSave = 0.10;
        rowIgnore = 0.17;
        rowLabelTop = 0.35;

        % Label buttons
        btnNames = {'Empty', 'Single', '>2'};
        colors = {[0.85, 0.93, 0.98], [0.86, 0.95, 0.86], [0.98, 0.88, 0.88]};
        for b = 1:numel(btnNames)
            ui.labelBtn(b) = uicontrol(ui.fig, ...
                'Style', 'pushbutton', ...
                'String', btnNames{b}, ...
                'Units', 'normalized', ...
                'Position', [col1, rowLabelTop - 0.08 * (b-1), wideBtn, 0.07], ...
                'BackgroundColor', colors{b}, ...
                'FontSize', 12, ...
                'Callback', @(~,~) set_label(btnNames{b})); %#ok<AGROW>
        end

        % Image-level controls
        ui.ignoreBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                                 'String', 'Ignore Image', ...
                                 'Units', 'normalized', ...
                                 'Position', [col1, rowIgnore, wideBtn, 0.06], ...
                                 'FontSize', 11, ...
                                 'BackgroundColor', [0.97, 0.92, 0.85], ...
                                 'Callback', @(~,~) ignore_image());
        ui.reloadBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                                  'String', 'Load Saved', ...
                                  'Units', 'normalized', ...
                                  'Position', [col2, rowIgnore, narrowBtn, 0.06], ...
                                  'FontSize', 11, ...
                                  'BackgroundColor', [0.93, 0.93, 0.93], ...
                                  'Callback', @(~,~) reload_saved());

        % Save controls
        ui.saveProgressBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                                       'String', 'Save Progress', ...
                                       'Units', 'normalized', ...
                                       'Position', [col1, rowSave, wideBtn, 0.06], ...
                                       'FontSize', 11, ...
                                       'BackgroundColor', [0.88, 0.95, 1], ...
                                       'Callback', @(~,~) save_results(true));
        ui.saveBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Save & Quit', ...
                               'Units', 'normalized', ...
                               'Position', [col2, rowSave, wideBtn, 0.06], ...
                               'FontSize', 11, ...
                               'BackgroundColor', [0.9, 0.9, 1], ...
                               'Callback', @(~,~) save_and_close());

        % Navigation
        ui.skipBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Skip (no label)', ...
                               'Units', 'normalized', ...
                               'Position', [col1, rowNav, wideBtn, 0.06], ...
                               'FontSize', 11, ...
                               'Callback', @(~,~) move_idx(1));
        ui.prevBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Prev', ...
                               'Units', 'normalized', ...
                               'Position', [col2, rowNav, narrowBtn, 0.06], ...
                               'FontSize', 11, ...
                               'Callback', @(~,~) move_idx(-1));
        ui.nextBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                               'String', 'Next', ...
                               'Units', 'normalized', ...
                               'Position', [col3, rowNav, narrowBtn, 0.06], ...
                               'FontSize', 11, ...
                               'Callback', @(~,~) move_idx(1));
        ui.nextImageBtn = uicontrol(ui.fig, 'Style', 'pushbutton', ...
                                    'String', 'Next Image', ...
                                    'Units', 'normalized', ...
                                    'Position', [col4, rowNav, imageNavBtn, 0.06], ...
                                    'FontSize', 11, ...
                                    'Callback', @(~,~) move_image(1));
    end

    function update_display()
        if currentIdx < 1, currentIdx = 1; end
        if currentIdx > totalDroplets, currentIdx = totalDroplets; end

        row = dropletTable(currentIdx, :);
        img = imread(fullfile(imageDir, row.ImageName{1}));
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
        imgMask = dropletTable.ImageIndex == row.ImageIndex;
        imgDropletCount = sum(imgMask);
        imgDropletIdx = sum(find(imgMask) <= currentIdx);
        statusStr = sprintf(['Droplet %d / %d (image %d / %d; %d of %d in this image)\n' ...
                             'File: %s  |  Label: %s%s'], ...
                             currentIdx, totalDroplets, row.ImageIndex, numel(imageFiles), ...
                             imgDropletIdx, imgDropletCount, row.ImageName{1}, ...
                             label_or_blank(row.Label), image_ignore_suffix(row));
        set(ui.status, 'String', statusStr);
    end

    function move_idx(delta)
        currentIdx = currentIdx + delta;
        if currentIdx < 1
            currentIdx = 1;
        end

        % If we ran off the end of existing droplets, process the next image(s)
        while currentIdx > totalDroplets && ~isempty(nextImageIdx)
            add_image_detections(nextImageIdx);
            nextImageIdx = find(~processedMask, 1, 'first');
            totalDroplets = height(dropletTable);
        end

        if currentIdx > totalDroplets
            currentIdx = totalDroplets;
        end
        update_display();
    end

    function move_image(deltaImage)
        if deltaImage == 0
            return;
        end

        currentImage = dropletTable.ImageIndex(currentIdx);
        target = currentImage + sign(deltaImage);

        while target >= 1 && target <= numel(imageFiles)
            ensure_image_loaded(target);
            mask = dropletTable.ImageIndex == target;
            if any(mask)
                if deltaImage > 0
                    currentIdx = find(mask, 1, 'first');
                else
                    currentIdx = find(mask, 1, 'last');
                end
                update_display();
                return;
            end
            target = target + sign(deltaImage);
        end
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
        save_results(true);
        if isvalid(ui.fig)
            close(ui.fig);
        end
    end

    function save_results(showDialog)
        csvPath = fullfile(imageDir, 'droplet_labels.csv');
        matPath = labelStore;
        writetable(dropletTable, csvPath);
        save(matPath, 'dropletTable');
        if showDialog
            msgbox(sprintf('Saved labels to:\n%s\n%s', csvPath, matPath), 'Saved');
        end
    end

    function reload_saved()
        tbl = load_existing_table(labelStore, imageFiles);
        if isempty(tbl)
            warndlg('No compatible saved labels found.', 'Load Saved');
            return;
        end
        dropletTable = tbl;
        processedMask(:) = false;
        processedMask(unique(dropletTable.ImageIndex)) = true;
        nextImageIdx = find(~processedMask, 1, 'first');
        totalDroplets = height(dropletTable);
        firstUnlabeled = find(cellfun(@isempty, dropletTable.Label), 1, 'first');
        if isempty(firstUnlabeled)
            currentIdx = 1;
        else
            currentIdx = firstUnlabeled;
        end
        update_display();
    end

    function added = add_image_detections(imgIdx)
        filePath = fullfile(imageDir, imageFiles(imgIdx).name);

        recIdx = find(strcmp({circlesResults.filename}, imageFiles(imgIdx).name), 1, 'first');
        if ~isempty(recIdx)
            det = circlesResults(recIdx);
        else
            det = dropletFindCircle(filePath);
            circlesResults(end + 1) = det; %#ok<AGROW>
            save(circlesPath, 'circlesResults');
        end

        centers = det.centers;
        radii   = det.radii;
        n = numel(radii);

        processedMask(imgIdx) = true;

        if n == 0
            added = 0;
            return;
        end

        bboxes = [centers(:, 1) - radii, centers(:, 2) - radii, 2 * radii, 2 * radii];
        dropletIds = (1:n)';
        newRows = table(repmat(imgIdx, n, 1), ...
                        dropletIds, ...
                        repmat({imageFiles(imgIdx).name}, n, 1), ...
                        bboxes, ...
                        repmat(false, n, 1), ...
                        repmat({''}, n, 1), ...
                        'VariableNames', {'ImageIndex', 'DropletID', 'ImageName', 'BoundingBox', 'ImageIgnored', 'Label'});

        dropletTable = [dropletTable; newRows]; %#ok<AGROW>
        dropletTable = sortrows(dropletTable, {'ImageIndex', 'DropletID'});
        totalDroplets = height(dropletTable);
        added = n;
    end

    function ensure_image_loaded(imgIdx)
        if processedMask(imgIdx)
            return;
        end

        add_image_detections(imgIdx);
        nextImageIdx = find(~processedMask, 1, 'first');
        totalDroplets = height(dropletTable);
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

function tbl = load_existing_table(labelStore, imageFiles)
%LOAD_EXISTING_TABLE Restore prior labels and detections if compatible with current files.
    tbl = table('Size', [0, 6], ...
                'VariableTypes', {'double', 'double', 'cell', 'double', 'logical', 'cell'}, ...
                'VariableNames', {'ImageIndex', 'DropletID', 'ImageName', 'BoundingBox', 'ImageIgnored', 'Label'});

    if ~exist(labelStore, 'file')
        return;
    end

    S = load(labelStore);
    if ~isfield(S, 'dropletTable') || ~istable(S.dropletTable)
        return;
    end

    prev = S.dropletTable;
    needed = {'ImageName', 'DropletID', 'BoundingBox', 'ImageIgnored', 'Label'};
    if ~all(ismember(needed, prev.Properties.VariableNames))
        return;
    end

    if isstring(prev.Label)
        prev.Label = cellstr(prev.Label);
    elseif iscategorical(prev.Label)
        prev.Label = cellstr(string(prev.Label));
    end
    if isstring(prev.ImageName)
        prev.ImageName = cellstr(prev.ImageName);
    end

    [found, idx] = ismember(prev.ImageName, {imageFiles.name});
    prev = prev(found, :);
    idx  = idx(found);
    if isempty(prev)
        return;
    end

    prev.ImageIndex = idx;
    tbl = prev(:, {'ImageIndex', 'DropletID', 'ImageName', 'BoundingBox', 'ImageIgnored', 'Label'});
    tbl = sortrows(tbl, {'ImageIndex', 'DropletID'});
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