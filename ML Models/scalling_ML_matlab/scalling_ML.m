%% Load ONNX model
onnxPath = "model_ML/scaling_fnn_v2.onnx";  

net = importNetworkFromONNX( ...
    onnxPath, ...
    "InputDataFormats",  ["BC","BC","BC"], ...
    "OutputDataFormats", ["BC","BC","BC","BC"] ... 
);

% analyzeNetwork(net)

%% Load TXT data
filePath = "C:\Git\Algoim_mimic\ML Models\scalling_ML_matlab\data\1kTestBernstein_p1_data_filtered64.txt";

T = readtable(filePath, ...
    "Delimiter",";", ...
    "TextType","string");

%% Extract ID column
ids = T.id;   % string array, one per row

%% Helper: parse "0,1,0,1" → 1×4 single
parseVec = @(s) single(str2double(split(s, ","))).';

%% Allocate inputs
N = height(T);   % batch size
C = 4;           % feature dimension (must match training)

exp_x  = zeros(N, C, "single");
exp_y  = zeros(N, C, "single");
coeff  = zeros(N, C, "single");

%% Fill input arrays
for i = 1:N
    exp_x(i,:) = parseVec(T.exp_x(i));
    exp_y(i,:) = parseVec(T.exp_y(i));
    coeff(i,:) = parseVec(T.coeff(i));
end

%% Run inference (NOW RETURNS 4 OUTPUTS)
[scale_x, scale_y, center_x, center_y] = predict(net, exp_x, exp_y, coeff);

%% Inspect results
disp("Input size (exp_x):");
disp(size(exp_x))

disp("Output size (scale_x):");
disp(size(scale_x))

disp("Output size (scale_y):");
disp(size(scale_y))

disp("Output size (center_x):");
disp(size(center_x))

disp("Output size (center_y):");
disp(size(center_y))

%% Save results as custom TXT (extended format with centers)
outPath = "predicted_scales_centers.txt";
fid = fopen(outPath, "w");

% Header (UPDATED)
fprintf(fid, "number;id;xscales;yscales;xcenter;ycenter\n");

for i = 1:N
    % Convert scale vectors to comma-separated strings
    xstr = sprintf('%.16g,', scale_x(i,:));
    ystr = sprintf('%.16g,', scale_y(i,:));
    xstr(end) = [];
    ystr(end) = [];

    % Centers might be scalar (1 value) or vector. Handle both safely.
    cx = center_x(i,:);
    cy = center_y(i,:);

    if numel(cx) == 1
        cxstr = sprintf('%.16g', cx);
    else
        cxstr = sprintf('%.16g,', cx);
        cxstr(end) = [];
    end

    if numel(cy) == 1
        cystr = sprintf('%.16g', cy);
    else
        cystr = sprintf('%.16g,', cy);
        cystr(end) = [];
    end

    % Write line
    fprintf(fid, "%d;%s;%s;%s;%s;%s\n", i, ids(i), xstr, ystr, cxstr, cystr);
end

fclose(fid);

disp("Saved predicted_scales_centers.txt");
