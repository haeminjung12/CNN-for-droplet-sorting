# Droplet labeling GUI (MATLAB)

This repository contains a pair of MATLAB scripts: `dropletFindCircle.m` for detection and `droplet_labeler.m` for labeling. The labeler loads a folder of microscopy images, detects droplets automatically with `dropletFindCircle`, and presents an interactive GUI so you can label each droplet as **Empty**, **Single**, **>2**, or ignore the entire image when it is not usable.

## Requirements
- MATLAB R2021b or later (older versions may work but were not tested)
- Image Processing Toolbox

## Usage
1. Start MATLAB and add this folder to your path.
2. Run the script with the directory of your images. If you omit the argument, a folder picker appears. You can re-open a
   folder later to continue labeling; existing `droplet_labels.mat` entries are restored when the bounding boxes match.

   ```matlab
   droplet_labeler('path/to/your/images');
   ```

3. For each detected droplet:
   - The full image is shown on the left with the current droplet highlighted.
   - A zoomed crop of the droplet is shown on the right.
   - Click **Empty**, **Single**, or **>2** to assign a label. Use **Prev**/**Next** to navigate or **Skip** to leave unlabeled. Use **Ignore Image** to mark all droplets in the current frame as unusable.
4. Click **Save & Quit** to write two files into the image directory:
   - `droplet_labels.csv` (spreadsheet-friendly)
   - `droplet_labels.mat` (MATLAB table for downstream scripts; includes image filename, bounding box, ignore flag, and label)

### Tweaking detection (dropletFindCircle)
`droplet_labeler` calls `dropletFindCircle`, which searches for `.tif/.tiff` files in the folder, enhances contrast, and runs two imfindcircles passes (small and large radii). Edit the parameter block at the top of `dropletFindCircle.m` to retune radius bands, sensitivity, and suppression thresholds for your images. Pass a second argument (`true`) to display per-image detection overlays when you want to visually inspect the circles.

`dropletFindCircle` saves the detections to `circlesResults.mat` in your image folder so you can reuse or inspect them between sessions. Launching `droplet_labeler` will reuse that file when present.
