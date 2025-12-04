function droplet_dataset_builder(imageDir, outputDir, targetSize, padding)
%DROPLET_DATASET_BUILDER Export cropped droplet images for CNN training.
%
%   DROPLET_DATASET_BUILDER(IMAGE_DIR) reads droplet labels produced by
%   droplet_labeler in IMAGE_DIR, crops each labeled droplet, resizes the
%   crop to 64x64, and writes it into per-class subfolders within a
%   "cropped_dataset" directory in IMAGE_DIR. A manifest CSV is also
%   emitted so you can trace each crop back to its source.
%
%   DROPLET_DATASET_BUILDER(IMAGE_DIR, OUTPUT_DIR, TARGET_SIZE, PADDING)
%   lets you override the output directory (default: IMAGE_DIR/cropped_dataset),
%   target size (default: [64 64]), and padding added around each bounding box
%   before cropping (default: 6 pixels). TARGET_SIZE must be a two-element
%   vector [rows cols].
%
%   Examples
%   --------
%       droplet_dataset_builder('path/to/images');
%       droplet_dataset_builder('path/to/images', 'my_dataset', [64 64], 4);
%
%   Requires: Image Processing Toolbox
%
%   See also: droplet_labeler, droplet_visualizer

    if nargin < 1 || isempty(imageDir)
        imageDir = uigetdir(pwd, 'Select image folder');
        if imageDir == 0
            return;
        end
    end
    if nargin < 2 || isempty(outputDir)
        outputDir = fullfile(imageDir, 'cropped_dataset');
    end
    if nargin < 3 || isempty(targetSize)
        targetSize = [64 64];
    end
    if numel(targetSize) ~= 2
        error('TARGET_SIZE must be a 2-element vector [rows cols].');
    end
    if nargin < 4 || isempty(padding)
        padding = 6; % mirror labeler default crop padding
    end

    labels = load_labels(imageDir);
    labels = normalize_labels(labels);
    labels = labels(~labels.ImageIgnored & ~strcmp(labels.Label, ''), :);
    if isempty(labels)
        error('No labeled droplets found (labels empty or all ignored).');
    end

    classes = unique(labels.Label);
    mkdir_if_needed(outputDir);
    outFolders = containers.Map;
    for i = 1:numel(classes)
        folderName = sanitize_label(classes{i});
        folderPath = fullfile(outputDir, folderName);
        mkdir_if_needed(folderPath);
        outFolders(classes{i}) = folderPath;
    end

    manifest = table('Size', [0 5], ...
                     'VariableTypes', {'cell','cell','double','double','double'}, ...
                     'VariableNames', {'CropPath','Label','ImageIndex','DropletID','Padding'});

    for r = 1:height(labels)
        row = labels(r, :);
        rowLabel = row.Label{1};
        if isstring(rowLabel)
            rowLabel = char(rowLabel);
        end
        rowImageName = row.ImageName{1};
        imgPath = fullfile(imageDir, rowImageName);
        if ~isfile(imgPath)
            warning('Skipping missing image: %s', imgPath);
            continue;
        end

        img = imread(imgPath);
        rect = expand_rect(row.BoundingBox, size(img), padding);
        crop = imcrop(img, rect);
        if isempty(crop)
            warning('Skipping droplet %d in %s (empty crop).', row.DropletID, row.ImageName{1});
            continue;
        end

        crop = imresize(crop, targetSize); %#ok<IMRESIZE>
        sanitized = sanitize_label(rowLabel);
        outName = sprintf('%s_img%03d_drop%03d.png', sanitized, row.ImageIndex, row.DropletID);
        outPath = fullfile(outFolders(rowLabel), outName);
        imwrite(crop, outPath);
        outPath = cellstr(outPath);

        manifest(end+1,:) = {outPath, rowLabel, row.ImageIndex, row.DropletID, padding};
    end

    manifestPath = fullfile(outputDir, 'dataset_index.csv');
    writetable(manifest, manifestPath);
    fprintf('Wrote %d crops to %s\nManifest: %s\n', height(manifest), outputDir, manifestPath);
end

function labels = load_labels(imageDir)
    matPath = fullfile(imageDir, 'droplet_labels.mat');
    csvPath = fullfile(imageDir, 'droplet_labels.csv');

    if isfile(matPath)
        S = load(matPath);
        if isfield(S, 'dropletTable') && istable(S.dropletTable)
            labels = S.dropletTable;
            return;
        end
    end

    if isfile(csvPath)
        labels = readtable(csvPath);
        return;
    end

    error('Could not find droplet_labels.mat or droplet_labels.csv in %s', imageDir);
end

function tbl = normalize_labels(tbl)
    % ensure expected variables and convert strings to cellstr
    % Rebuild BoundingBox from CSV-expanded columns when needed
    if ~ismember('BoundingBox', tbl.Properties.VariableNames)
        bbCols = startsWith(tbl.Properties.VariableNames, 'BoundingBox');
        if nnz(bbCols) ~= 4
            error('Labels table is missing required columns. Found: %s', strjoin(tbl.Properties.VariableNames, ', '));
        end
        bbNames = tbl.Properties.VariableNames(bbCols);
        % enforce deterministic order
        [~, order] = sort(bbNames);
        bbData = tbl{:, bbCols};
        bbData = bbData(:, order);
        tbl.BoundingBox = bbData;
    end

    required = {'ImageIndex','DropletID','ImageName','BoundingBox','ImageIgnored','Label'};
    if ~all(ismember(required, tbl.Properties.VariableNames))
        error('Labels table is missing required columns. Found: %s', strjoin(tbl.Properties.VariableNames, ', '));
    end

    if isstring(tbl.Label)
        tbl.Label = cellstr(tbl.Label);
    elseif iscategorical(tbl.Label)
        tbl.Label = cellstr(string(tbl.Label));
    end
    tbl.Label = cellfun(@strtrim, tbl.Label, 'UniformOutput', false);
    if isstring(tbl.ImageName)
        tbl.ImageName = cellstr(tbl.ImageName);
    end
    if ~islogical(tbl.ImageIgnored)
        tbl.ImageIgnored = logical(tbl.ImageIgnored);
    end

    % enforce numeric types for manifest compatibility
    if iscell(tbl.ImageIndex)
        tbl.ImageIndex = cellfun(@double, tbl.ImageIndex);
    elseif isstring(tbl.ImageIndex)
        tbl.ImageIndex = str2double(tbl.ImageIndex);
    elseif iscategorical(tbl.ImageIndex)
        tbl.ImageIndex = double(tbl.ImageIndex);
    end

    if iscell(tbl.DropletID)
        tbl.DropletID = cellfun(@double, tbl.DropletID);
    elseif isstring(tbl.DropletID)
        tbl.DropletID = str2double(tbl.DropletID);
    elseif iscategorical(tbl.DropletID)
        tbl.DropletID = double(tbl.DropletID);
    end
    % replace empty cells with ''
    emptyMask = cellfun(@(c) isempty(c) || (isstring(c) && strlength(c)==0), tbl.Label);
    tbl.Label(emptyMask) = {''};
end

function rect = expand_rect(bbox, imgSize, pad)
    x1 = max(bbox(1) - pad, 1);
    y1 = max(bbox(2) - pad, 1);
    x2 = min(bbox(1) + bbox(3) + pad, imgSize(2));
    y2 = min(bbox(2) + bbox(4) + pad, imgSize(1));
    rect = [x1, y1, x2 - x1, y2 - y1];
end

function name = sanitize_label(lbl)
    name = regexprep(lbl, '[^A-Za-z0-9]+', '_');
    name = regexprep(name, '^_+|_+$', '');
    if isempty(name)
        name = 'label';
    end
    if isstring(name)
        name = char(name);
    end
end

function mkdir_if_needed(pathStr)
    if ~exist(pathStr, 'dir')
        mkdir(pathStr);
    end
end