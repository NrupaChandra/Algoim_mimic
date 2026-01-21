clear; clc;

%% Paths 
ref_file  = "1kTestBernstein_p1_output_8_filtered64_sorted.txt";
pred_file = "Weight_scalling_from_ML.txt";

out_dir = "plots_simple";
if ~exist(out_dir, "dir")
    mkdir(out_dir);
end

%%  Read reference table 
opts = detectImportOptions(ref_file, "Delimiter", ";");
opts = setvartype(opts, "string");
T_ref = readtable(ref_file, opts);

%% Read prediction table
opts = detectImportOptions(pred_file, "Delimiter", ";");
opts = setvartype(opts, "string");
T_pred = readtable(pred_file, opts);

%% Helper: parse comma-separated vector 
parseVec = @(s) str2double(split(s, ","))';

%% Loop over reference entries 
for i = 1:height(T_ref)

    id = T_ref.id(i);

    % Find matching prediction
    idx = find(T_pred.id == id, 1);
    if isempty(idx)
        continue
    end

    % ---- True (reference) data ----
    xr = parseVec(T_ref.nodes_x(i));
    yr = parseVec(T_ref.nodes_y(i));
    wr = parseVec(T_ref.weights(i));

    % ---- ML-predicted data ----
    xp = parseVec(T_pred.nodes_x(idx));
    yp = parseVec(T_pred.nodes_y(idx));
    wp = parseVec(T_pred.weights(idx));

    % ---- Area via quadrature ----
    A_true = sum(wr);
    A_pred = sum(wp);

    % ---- Absolute percentage error ----
    A_err_pct = 100 * abs(A_pred - A_true) / A_true;

    % ---- Plot ----
    fig = figure("Visible","off","Position",[100 100 600 600]);
    hold on;

    plot(xr, yr, '.', 'MarkerSize', 18);   % true nodes
    plot(xp, yp, 'x', 'MarkerSize', 8);    % ML nodes

    axis equal;
    axis([-1 1 -1 1]);
    grid on;

    title("Node comparison");
    xlabel("x");
    ylabel("y");
    legend("True", "ML", "Location","best");

    % ---- Print area info on plot ----
    txt = sprintf( ...
        "Area (true) = %.8f\nArea (ML)   = %.8f\nAbs err     = %.4f %%", ...
        A_true, A_pred, A_err_pct );

    text(0.02, 0.98, txt, ...
        "Units","normalized", ...
        "VerticalAlignment","top", ...
        "BackgroundColor","w", ...
        "FontSize",10);

    % ---- Save ----
    saveas(fig, fullfile(out_dir, id + ".png"));
    close(fig);
end
