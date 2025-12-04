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