clear; clc;

%% ------------------------------------------------------------
% 1. Load Scale+Center DATA (ground truth)
%% ------------------------------------------------------------
inFile = "predicted_scales_centers.txt";
%%inFile = "1kTestBernstein_p1_ScaleCenter.txt";

opts = detectImportOptions(inFile, "Delimiter",";");
opts = setvartype(opts, "string");

T = readtable(inFile, opts);
vars = T.Properties.VariableNames;

assert(any(strcmp(vars,"id")), ...
    "Input file must contain a column named 'id'");

ids = string(T.id);
N   = height(T);

%% ------------------------------------------------------------
% 2. Detect column names robustly
%% ------------------------------------------------------------
% scales
sx_candidates = vars(startsWith(vars,"xscale","IgnoreCase",true));
sy_candidates = vars(startsWith(vars,"yscale","IgnoreCase",true));

assert(~isempty(sx_candidates), "No xscales column found");
assert(~isempty(sy_candidates), "No yscales column found");

sx_name = sx_candidates{1};
sy_name = sy_candidates{1};

% centers (singular OR plural)
cx_candidates = vars(startsWith(vars,"xcenter","IgnoreCase",true));
cy_candidates = vars(startsWith(vars,"ycenter","IgnoreCase",true));

assert(~isempty(cx_candidates), "No xcenter/xcenters column found");
assert(~isempty(cy_candidates), "No ycenter/ycenters column found");

cx_name = cx_candidates{1};
cy_name = cy_candidates{1};

fprintf("Detected columns:\n");
fprintf("  %s  %s  %s  %s\n", sx_name, sy_name, cx_name, cy_name);

%% ------------------------------------------------------------
% 3. Parse string vectors safely
%% ------------------------------------------------------------
parseVec = @(s) single(str2double(split(s, ","))).';

scale_x  = zeros(N,8,"single");
scale_y  = zeros(N,8,"single");
center_x = zeros(N,8,"single");
center_y = zeros(N,8,"single");

for i = 1:N
    scale_x(i,:)  = parseVec(T.(sx_name)(i));
    scale_y(i,:)  = parseVec(T.(sy_name)(i));
    center_x(i,:) = parseVec(T.(cx_name)(i));
    center_y(i,:) = parseVec(T.(cy_name)(i));
end

%% ------------------------------------------------------------
% 4. Fixed reference constants (IDENTICAL to Python)
%% ------------------------------------------------------------
weightRefs = single([ ...
    0.1012285362903763
    0.2223810344533745
    0.3137066458778873
    0.3626837833783620
    0.3626837833783620
    0.3137066458778873
    0.2223810344533745
    0.1012285362903763 ]);

Wref = weightRefs * weightRefs.';   % 8x8

%% ------------------------------------------------------------
% 5. Reference Gauss–Legendre nodes
%% ------------------------------------------------------------
g = single([ ...
   -0.9602898564975363
   -0.7966664774136267
   -0.5255324099163290
   -0.1834346424956498
    0.1834346424956498
    0.5255324099163290
    0.7966664774136267
    0.9602898564975363 ]);

%% ------------------------------------------------------------
% 6. Open output file
%% ------------------------------------------------------------
outFile = "Weight_scalling_from_ML.txt";
%%outFile = "Weight_scalling_from_data.txt";
fid = fopen(outFile,"w");

fprintf(fid,"number;id;nodes_x;nodes_y;weights\n");

%% ------------------------------------------------------------
% 7. Reconstruction (EXACT inverse + centers)
%% ------------------------------------------------------------
for k = 1:N

    sx = scale_x(k,:).';     % 8x1
    sy = scale_y(k,:).';     % 8x1
    cx = center_x(k,:).';   % 8x1
    cy = center_y(k,:).';   % 8x1

    % Nodes (affine, nodewise centers)
    X = cx * ones(1,8) + sx * g.';   % 8x8
    Y = ones(8,1) * cy.' + g * sy.'; % 8x8

    % Weights (unchanged)
    W = (sx * sy.') .* Wref;

    % Flatten row-major (Python-compatible)
    x_flat = reshape(X.', 1, []);
    y_flat = reshape(Y.', 1, []);
    w_flat = reshape(W.', 1, []);

    % Format
    xs = strjoin(compose("%.16g", x_flat), ",");
    ys = strjoin(compose("%.16g", y_flat), ",");
    ws = strjoin(compose("%.16g", w_flat), ",");

    fprintf(fid,"%d;%s;%s;%s;%s\n", ...
        k, ids(k), xs, ys, ws);
end

fclose(fid);

fprintf(" Reconstruction finished: %s\n", outFile);
