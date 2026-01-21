classdef Unsqueeze_To_DivLayer1006 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    
    %#codegen
    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    
    properties (Learnable)
        onnx__Pow_50
        onnx__Pow_52
        onnx__ReduceSum_33
        x_nodal_preproces_1
        x_nodal_preproces_2
        x_nodal_preproces_4
    end
    
    properties
        ONNXParams         % An ONNXParameters object containing parameters used by this layer.
    end
    
    methods
        function this = Unsqueeze_To_DivLayer1006(name, onnxParams)
            this.Name = name;
            this.NumInputs = 3;
            this.OutputNames = {'x_nodal_preproces_5'};
            this.ONNXParams = onnxParams;
            this.onnx__Pow_50 = onnxParams.Learnables.onnx__Pow_50;
            this.onnx__Pow_52 = onnxParams.Learnables.onnx__Pow_52;
            this.onnx__ReduceSum_33 = onnxParams.Learnables.onnx__ReduceSum_33;
            this.x_nodal_preproces_1 = onnxParams.Learnables.x_nodal_preproces_1;
            this.x_nodal_preproces_2 = onnxParams.Learnables.x_nodal_preproces_2;
            this.x_nodal_preproces_4 = onnxParams.Learnables.x_nodal_preproces_4;
        end
        
        function [x_nodal_preproces_5] = predict(this, exp_x, exp_y, coeff)
            if isdlarray(exp_x)
                exp_x = stripdims(exp_x);
            end
            if isdlarray(exp_y)
                exp_y = stripdims(exp_y);
            end
            if isdlarray(coeff)
                coeff = stripdims(coeff);
            end
            exp_xNumDims = 2;
            exp_yNumDims = 2;
            coeffNumDims = 2;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.onnx__Pow_50 = this.onnx__Pow_50;
            onnxParams.Learnables.onnx__Pow_52 = this.onnx__Pow_52;
            onnxParams.Learnables.onnx__ReduceSum_33 = this.onnx__ReduceSum_33;
            onnxParams.Learnables.x_nodal_preproces_1 = this.x_nodal_preproces_1;
            onnxParams.Learnables.x_nodal_preproces_2 = this.x_nodal_preproces_2;
            onnxParams.Learnables.x_nodal_preproces_4 = this.x_nodal_preproces_4;
            [x_nodal_preproces_5, x_nodal_preproces_5NumDims] = Unsqueeze_To_DivFcn(exp_x, exp_y, coeff, exp_xNumDims, exp_yNumDims, coeffNumDims, onnxParams, 'Training', false, ...
                'InputDataPermutation', {[2 1], [2 1], [2 1], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 1], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {x_nodal_preproces_5}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Unsqueeze_To_DivLayer1006');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Unsqueeze_To_DivLayer1006'));
            end
            x_nodal_preproces_5 = dlarray(single(x_nodal_preproces_5), 'CB');
            if ~coder.target('MATLAB')
                x_nodal_preproces_5 = extractdata(x_nodal_preproces_5);
            end
        end
        
        function [x_nodal_preproces_5] = forward(this, exp_x, exp_y, coeff)
            if isdlarray(exp_x)
                exp_x = stripdims(exp_x);
            end
            if isdlarray(exp_y)
                exp_y = stripdims(exp_y);
            end
            if isdlarray(coeff)
                coeff = stripdims(coeff);
            end
            exp_xNumDims = 2;
            exp_yNumDims = 2;
            coeffNumDims = 2;
            onnxParams = this.ONNXParams;
            onnxParams.Learnables.onnx__Pow_50 = this.onnx__Pow_50;
            onnxParams.Learnables.onnx__Pow_52 = this.onnx__Pow_52;
            onnxParams.Learnables.onnx__ReduceSum_33 = this.onnx__ReduceSum_33;
            onnxParams.Learnables.x_nodal_preproces_1 = this.x_nodal_preproces_1;
            onnxParams.Learnables.x_nodal_preproces_2 = this.x_nodal_preproces_2;
            onnxParams.Learnables.x_nodal_preproces_4 = this.x_nodal_preproces_4;
            [x_nodal_preproces_5, x_nodal_preproces_5NumDims] = Unsqueeze_To_DivFcn(exp_x, exp_y, coeff, exp_xNumDims, exp_yNumDims, coeffNumDims, onnxParams, 'Training', true, ...
                'InputDataPermutation', {[2 1], [2 1], [2 1], ['as-is'], ['as-is'], ['as-is']}, ...
                'OutputDataPermutation', {[2 1], ['as-is']});
            if any(cellfun(@(A)~isnumeric(A), {x_nodal_preproces_5}))
                fprintf('Runtime error in network. The custom layer ''%s'' output a non-numeric value.\n', 'Unsqueeze_To_DivLayer1006');
                error(message('nnet_cnn_onnx:onnx:BadCustomLayerRuntimeOutput', 'Unsqueeze_To_DivLayer1006'));
            end
            x_nodal_preproces_5 = dlarray(single(x_nodal_preproces_5), 'CB');
            if ~coder.target('MATLAB')
                x_nodal_preproces_5 = extractdata(x_nodal_preproces_5);
            end
        end
    end
end

function [x_nodal_preproces_5, x_nodal_preproces_5NumDims, state] = Unsqueeze_To_DivFcn(exp_x, exp_y, coeff, exp_xNumDims, exp_yNumDims, coeffNumDims, params, varargin)
%UNSQUEEZE_TO_DIVFCN Function implementing an imported ONNX network.
%
% THIS FILE WAS AUTO-GENERATED BY importONNXFunction.
% ONNX Operator Set Version: 14
%
% Variable names in this function are taken from the original ONNX file.
%
% [X_NODAL_PREPROCES_5] = Unsqueeze_To_DivFcn(EXP_X, EXP_Y, COEFF, PARAMS)
%			- Evaluates the imported ONNX network UNSQUEEZE_TO_DIVFCN with input(s)
%			EXP_X, EXP_Y, COEFF and the imported network parameters in PARAMS. Returns
%			network output(s) in X_NODAL_PREPROCES_5.
%
% [X_NODAL_PREPROCES_5, STATE] = Unsqueeze_To_DivFcn(EXP_X, EXP_Y, COEFF, PARAMS)
%			- Additionally returns state variables in STATE. When training,
%			use this form and set TRAINING to true.
%
% [__] = Unsqueeze_To_DivFcn(EXP_X, EXP_Y, COEFF, PARAMS, 'NAME1', VAL1, 'NAME2', VAL2, ...)
%			- Specifies additional name-value pairs described below:
%
% 'Training'
% 			Boolean indicating whether the network is being evaluated for
%			prediction or training. If TRAINING is true, state variables
%			will be updated.
%
% 'InputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			 between the dimensions of the input data and the dimensions of
%			the ONNX model input. For example, the permutation from HWCN
%			(MATLAB standard) to NCHW (ONNX standard) uses the vector
%			[4 3 1 2]. See the documentation for IMPORTONNXFUNCTION for
%			more information about automatic permutation.
%
%			'none' - Input(s) are passed in the ONNX model format. See 'Inputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between input data dimensions and the expected
%			ONNX input dimensions.%
%			cell array - If the network has multiple inputs, each cell
%			contains 'auto', 'none', or a numeric vector.
%
% 'OutputDataPermutation'
%			'auto' - Automatically attempt to determine the permutation
%			between the dimensions of the output and a conventional MATLAB
%			dimension ordering. For example, the permutation from NC (ONNX
%			standard) to CN (MATLAB standard) uses the vector [2 1]. See
%			the documentation for IMPORTONNXFUNCTION for more information
%			about automatic permutation.
%
%			'none' - Return output(s) as given by the ONNX model. See 'Outputs'.
%
%			numeric vector - The permutation vector describing the
%			transformation between the ONNX output dimensions and the
%			desired output dimensions.%
%			cell array - If the network has multiple outputs, each cell
%			contains 'auto', 'none' or a numeric vector.
%
% Inputs:
% -------
% EXP_X, EXP_Y, COEFF
%			- Input(s) to the ONNX network.
%			  The input size(s) expected by the ONNX file are:
%				  EXP_X:		[batch, 4]				Type: FLOAT
%				  EXP_Y:		[batch, 4]				Type: FLOAT
%				  COEFF:		[batch, 4]				Type: FLOAT
%			  By default, the function will try to permute the input(s)
%			  into this dimension ordering. If the default is incorrect,
%			  use the 'InputDataPermutation' argument to control the
%			  permutation.
%
%
% PARAMS	- Network parameters returned by 'importONNXFunction'.
%
%
% Outputs:
% --------
% X_NODAL_PREPROCES_5
%			- Output(s) of the ONNX network.
%			  Without permutation, the size(s) of the outputs are:
%				  X_NODAL_PREPROCES_5:		[Unknown, Unknown]				Type: FLOAT
%			  By default, the function will try to permute the output(s)
%			  from this dimension ordering into a conventional MATLAB
%			  ordering. If the default is incorrect, use the
%			  'OutputDataPermutation' argument to control the permutation.
%
% STATE		- (Optional) State variables. When TRAINING is true, these will
% 			  have been updated from the original values in PARAMS.State.
%
%
%  See also importONNXFunction

% Preprocess the input data and arguments:
[exp_x, exp_y, coeff, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(exp_x, exp_y, coeff, params, varargin{:});
% Put all variables into a single struct to implement dynamic scoping:
[Vars, NumDims] = packageVariables(params, {'exp_x', 'exp_y', 'coeff'}, {exp_x, exp_y, coeff}, [exp_xNumDims exp_yNumDims coeffNumDims]);
% Call the top-level graph function:
[x_nodal_preproces_5, x_nodal_preproces_5NumDims, state] = Unsqueeze_To_DivGraph1000(exp_x, exp_y, coeff, NumDims.exp_x, NumDims.exp_y, NumDims.coeff, Vars, NumDims, Training, params.State);
% Postprocess the output data
[x_nodal_preproces_5] = postprocessOutput(x_nodal_preproces_5, outputDataPerms, anyDlarrayInputs, Training, varargin{:});
end

function [x_nodal_preproces_5, x_nodal_preproces_5NumDims1005, state] = Unsqueeze_To_DivGraph1000(exp_x, exp_y, coeff, exp_xNumDims1002, exp_yNumDims1003, coeffNumDims1004, Vars, NumDims, Training, state)
% Function implementing the graph 'Unsqueeze_To_DivGraph1000'
% Update Vars and NumDims from the graph's formal input parameters. Note that state variables are already in Vars.
Vars.exp_x = exp_x;
NumDims.exp_x = exp_xNumDims1002;
Vars.exp_y = exp_y;
NumDims.exp_y = exp_yNumDims1003;
Vars.coeff = coeff;
NumDims.coeff = coeffNumDims1004;

% Execute the operators:
% Unsqueeze:
[shape, NumDims.x_nodal_preproces_14] = prepareUnsqueezeArgs(Vars.exp_x, Vars.x_nodal_preproces_4, NumDims.exp_x);
Vars.x_nodal_preproces_14 = reshape(Vars.exp_x, shape);

% Unsqueeze:
[shape, NumDims.x_nodal_preproces_12] = prepareUnsqueezeArgs(Vars.exp_y, Vars.x_nodal_preproces_1, NumDims.exp_y);
Vars.x_nodal_preproces_12 = reshape(Vars.exp_y, shape);

% Unsqueeze:
[shape, NumDims.x_nodal_preproces_13] = prepareUnsqueezeArgs(Vars.coeff, Vars.x_nodal_preproces_2, NumDims.coeff);
Vars.x_nodal_preproces_13 = reshape(Vars.coeff, shape);

% Pow:
Vars.x_nodal_preproces_9 = power(Vars.onnx__Pow_50, Vars.x_nodal_preproces_14);
NumDims.x_nodal_preproces_9 = max(NumDims.onnx__Pow_50, NumDims.x_nodal_preproces_14);

% Mul:
Vars.x_nodal_preproces_7 = Vars.x_nodal_preproces_13 .* Vars.x_nodal_preproces_9;
NumDims.x_nodal_preproces_7 = max(NumDims.x_nodal_preproces_13, NumDims.x_nodal_preproces_9);

% Pow:
Vars.x_nodal_preproces_8 = power(Vars.onnx__Pow_52, Vars.x_nodal_preproces_12);
NumDims.x_nodal_preproces_8 = max(NumDims.onnx__Pow_52, NumDims.x_nodal_preproces_12);

% Mul:
Vars.x_nodal_preproces_6 = Vars.x_nodal_preproces_7 .* Vars.x_nodal_preproces_8;
NumDims.x_nodal_preproces_6 = max(NumDims.x_nodal_preproces_7, NumDims.x_nodal_preproces_8);

% ReduceSum:
dims = prepareReduceArgs(Vars.onnx__ReduceSum_33, NumDims.x_nodal_preproces_6);
Vars.x_nodal_preproces_11 = sum(Vars.x_nodal_preproces_6, dims);
[Vars.x_nodal_preproces_11, NumDims.x_nodal_preproces_11] = onnxSqueeze(Vars.x_nodal_preproces_11, Vars.onnx__ReduceSum_33, NumDims.x_nodal_preproces_6);

% ReduceMax:
dims = prepareReduceArgs(Vars.ReduceMaxAxes1001, NumDims.x_nodal_preproces_11);
Vars.x_nodal_preproces_10 = max(Vars.x_nodal_preproces_11, [], dims);
NumDims.x_nodal_preproces_10 = NumDims.x_nodal_preproces_11;

% Add:
Vars.x_nodal_preprocessor = Vars.x_nodal_preproces_10 + Vars.x_nodal_preproces_3;
NumDims.x_nodal_preprocessor = max(NumDims.x_nodal_preproces_10, NumDims.x_nodal_preproces_3);

% Div:
Vars.x_nodal_preproces_5 = Vars.x_nodal_preproces_11 ./ Vars.x_nodal_preprocessor;
NumDims.x_nodal_preproces_5 = max(NumDims.x_nodal_preproces_11, NumDims.x_nodal_preprocessor);

% Set graph output arguments from Vars and NumDims:
x_nodal_preproces_5 = Vars.x_nodal_preproces_5;
x_nodal_preproces_5NumDims1005 = NumDims.x_nodal_preproces_5;
% Set output state from Vars:
state = updateStruct(state, Vars);
end

function [inputDataPerms, outputDataPerms, Training] = parseInputs(exp_x, exp_y, coeff, numDataOutputs, params, varargin)
% Function to validate inputs to Unsqueeze_To_DivFcn:
p = inputParser;
isValidArrayInput = @(x)isnumeric(x) || isstring(x);
isValidONNXParameters = @(x)isa(x, 'ONNXParameters');
addRequired(p, 'exp_x', isValidArrayInput);
addRequired(p, 'exp_y', isValidArrayInput);
addRequired(p, 'coeff', isValidArrayInput);
addRequired(p, 'params', isValidONNXParameters);
addParameter(p, 'InputDataPermutation', 'auto');
addParameter(p, 'OutputDataPermutation', 'auto');
addParameter(p, 'Training', false);
parse(p, exp_x, exp_y, coeff, params, varargin{:});
inputDataPerms = p.Results.InputDataPermutation;
outputDataPerms = p.Results.OutputDataPermutation;
Training = p.Results.Training;
if isnumeric(inputDataPerms)
    inputDataPerms = {inputDataPerms};
end
if isstring(inputDataPerms) && isscalar(inputDataPerms) || ischar(inputDataPerms)
    inputDataPerms = repmat({inputDataPerms},1,3);
end
if isnumeric(outputDataPerms)
    outputDataPerms = {outputDataPerms};
end
if isstring(outputDataPerms) && isscalar(outputDataPerms) || ischar(outputDataPerms)
    outputDataPerms = repmat({outputDataPerms},1,numDataOutputs);
end
end

function [exp_x, exp_y, coeff, Training, outputDataPerms, anyDlarrayInputs] = preprocessInput(exp_x, exp_y, coeff, params, varargin)
% Parse input arguments
[inputDataPerms, outputDataPerms, Training] = parseInputs(exp_x, exp_y, coeff, 1, params, varargin{:});
anyDlarrayInputs = any(cellfun(@(x)isa(x, 'dlarray'), {exp_x, exp_y, coeff}));
% Make the input variables into unlabelled dlarrays:
exp_x = makeUnlabeledDlarray(exp_x);
exp_y = makeUnlabeledDlarray(exp_y);
coeff = makeUnlabeledDlarray(coeff);
% Permute inputs if requested:
exp_x = permuteInputVar(exp_x, inputDataPerms{1}, 2);
exp_y = permuteInputVar(exp_y, inputDataPerms{2}, 2);
coeff = permuteInputVar(coeff, inputDataPerms{3}, 2);
end

function [x_nodal_preproces_5] = postprocessOutput(x_nodal_preproces_5, outputDataPerms, anyDlarrayInputs, Training, varargin)
% Set output type:
if ~anyDlarrayInputs && ~Training
    if isdlarray(x_nodal_preproces_5)
        x_nodal_preproces_5 = extractdata(x_nodal_preproces_5);
    end
end
% Permute outputs if requested:
x_nodal_preproces_5 = permuteOutputVar(x_nodal_preproces_5, outputDataPerms{1}, 2);
end


%% dlarray functions implementing ONNX operators:

function [Y, numDimsY] = onnxSqueeze(X, ONNXAxes, numDimsX)
% Implements the ONNX Squeeze operator
if numDimsX == 0
    Y = X;
    numDimsY = numDimsX;
else
    % Find the new ONNX shape
    curOShape = size(X, numDimsX:-1:1);
    if isempty(ONNXAxes)
        newOShape = curOShape(curOShape ~= 1);
    else
        ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsX;
        newOShape = curOShape;
        newOShape(ONNXAxes+1) = [];
    end
    % Get numDimsY from ONNX shape
    numDimsY  = numel(newOShape);
    newMShape = [fliplr(newOShape) ones(1, 2-numDimsY)];    % Append 1's to shape if numDims<2
    Y         = reshape(X, newMShape);
end
end

function dims = prepareReduceArgs(ONNXAxes, numDimsX)
% Prepares arguments for implementing the ONNX Reduce operator
if isempty(ONNXAxes)
    ONNXAxes = 0:numDimsX-1;   % All axes
end
ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsX;
dims = numDimsX - ONNXAxes;
end

function [newShape, numDimsY] = prepareUnsqueezeArgs(X, ONNXAxes, numDimsX)
% Prepares arguments for implementing the ONNX Unsqueeze operator
numDimsY = numDimsX + numel(ONNXAxes);
ONNXAxes = extractdata(ONNXAxes);
ONNXAxes(ONNXAxes<0) = ONNXAxes(ONNXAxes<0) + numDimsY;
ONNXAxes = sort(ONNXAxes);                                              % increasing order
if numDimsY == 1
    newShape = size(X);
else
    DLTAxes  = flip(numDimsY - ONNXAxes);                                  % increasing order
    newShape = ones(1, numDimsY);
    posToSet = setdiff(1:numDimsY, DLTAxes, 'stable');
    newShape(posToSet) = size(X, 1:numel(posToSet));
end
end

%% Utility functions:

function s = appendStructs(varargin)
% s = appendStructs(s1, s2,...). Assign all fields in s1, s2,... into s.
if isempty(varargin)
    s = struct;
else
    s = varargin{1};
    for i = 2:numel(varargin)
        fromstr = varargin{i};
        fs = fieldnames(fromstr);
        for j = 1:numel(fs)
            s.(fs{j}) = fromstr.(fs{j});
        end
    end
end
end

function checkInputSize(inputShape, expectedShape, inputName)

if numel(expectedShape)==0
    % The input is a scalar
    if ~isequal(inputShape, [1 1])
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, "[1,1]", inputSizeStr));
    end
elseif numel(expectedShape)==1
    % The input is a vector
    if ~shapeIsColumnVector(inputShape) || ~iSizesMatch({inputShape(1)}, expectedShape)
        expectedShape{2} = 1;
        expectedSizeStr = makeSizeString(expectedShape);
        inputSizeStr = makeSizeString(inputShape);
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
else
    % The input has 2 dimensions or more
    
    % The input dimensions have been reversed; flip them back to compare to the
    % expected ONNX shape.
    inputShape = fliplr(inputShape);
    
    % If the expected shape has fewer dims than the input shape, error.
    if numel(expectedShape) < numel(inputShape)
        expectedSizeStr = strjoin(["[", strjoin(string(expectedShape), ","), "]"], "");
        error(message('nnet_cnn_onnx:onnx:InputHasGreaterNDims', inputName, expectedSizeStr));
    end
    
    % Prepad the input shape with trailing ones up to the number of elements in
    % expectedShape
    inputShape = num2cell([ones(1, numel(expectedShape) - length(inputShape)) inputShape]);
    
    % Find the number of variable size dimensions in the expected shape
    numVariableInputs = sum(cellfun(@(x) isa(x, 'char') || isa(x, 'string'), expectedShape));
    
    % Find the number of input dimensions that are not in the expected shape
    % and cannot be represented by a variable dimension
    nonMatchingInputDims = setdiff(string(inputShape), string(expectedShape));
    numNonMatchingInputDims  = numel(nonMatchingInputDims) - numVariableInputs;
    
    expectedSizeStr = makeSizeString(expectedShape);
    inputSizeStr = makeSizeString(inputShape);
    if numNonMatchingInputDims == 0 && ~iSizesMatch(inputShape, expectedShape)
        % The actual and expected input dimensions match, but in
        % a different order. The input needs to be permuted.
        error(message('nnet_cnn_onnx:onnx:InputNeedsPermute',inputName, expectedSizeStr, inputSizeStr));
    elseif numNonMatchingInputDims > 0
        % The actual and expected input sizes do not match.
        error(message('nnet_cnn_onnx:onnx:InputNeedsResize',inputName, expectedSizeStr, inputSizeStr));
    end
end
end

function doesMatch = iSizesMatch(inputShape, expectedShape)
% Check whether the input and expected shapes match, in order.
% Size elements match if (1) the elements are equal, or (2) the expected
% size element is a variable (represented by a character vector or string)
doesMatch = true;
for i=1:numel(inputShape)
    if ~(isequal(inputShape{i},expectedShape{i}) || ischar(expectedShape{i}) || isstring(expectedShape{i}))
        doesMatch = false;
        return
    end
end
end

function sizeStr = makeSizeString(shape)
sizeStr = strjoin(["[", strjoin(string(shape), ","), "]"], "");
end

function isVec = shapeIsColumnVector(shape)
if numel(shape) == 2 && shape(2) == 1
    isVec = true;
else
    isVec = false;
end
end
function X = makeUnlabeledDlarray(X)
% Make numeric X into an unlabelled dlarray
if isa(X, 'dlarray')
    X = stripdims(X);
elseif isnumeric(X)
    if isinteger(X)
        % Make ints double so they can combine with anything without
        % reducing precision
        X = double(X);
    end
    X = dlarray(X);
end
end

function [Vars, NumDims] = packageVariables(params, inputNames, inputValues, inputNumDims)
% inputNames, inputValues are cell arrays. inputRanks is a numeric vector.
Vars = appendStructs(params.Learnables, params.Nonlearnables, params.State);
NumDims = params.NumDimensions;
% Add graph inputs
for i = 1:numel(inputNames)
    Vars.(inputNames{i}) = inputValues{i};
    NumDims.(inputNames{i}) = inputNumDims(i);
end
end

function X = permuteInputVar(X, userDataPerm, onnxNDims)
% Returns reverse-ONNX ordering
if onnxNDims == 0
    return;
elseif onnxNDims == 1 && isvector(X)
    X = X(:);
    return;
elseif isnumeric(userDataPerm)
    % Permute into reverse ONNX ordering
    if numel(userDataPerm) ~= onnxNDims
        error(message('nnet_cnn_onnx:onnx:InputPermutationSize', numel(userDataPerm), onnxNDims));
    end
    perm = fliplr(userDataPerm);
elseif isequal(userDataPerm, 'auto') && onnxNDims == 4
    % Permute MATLAB HWCN to reverse onnx (WHCN)
    perm = [2 1 3 4];
elseif isequal(userDataPerm, 'as-is')
    % Do not permute the input
    perm = 1:ndims(X);
else
    % userDataPerm is either 'none' or 'auto' with no default, which means
    % it's already in onnx ordering, so just make it reverse onnx
    perm = max(2,onnxNDims):-1:1;
end
X = permute(X, perm);
end

function Y = permuteOutputVar(Y, userDataPerm, onnxNDims)
switch onnxNDims
    case 0
        perm = [];
    case 1
        if isnumeric(userDataPerm)
            % Use the user's permutation because Y is a column vector which
            % already matches ONNX.
            perm = userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            % Treat the 1D onnx vector as a 2D column and transpose it
            perm = [2 1];
        else
            % userDataPerm is 'none'. Leave Y alone because it already
            % matches onnx.
            perm = [];
        end
    otherwise
        % ndims >= 2
        if isnumeric(userDataPerm)
            % Use the inverse of the user's permutation. This is not just the
            % flip of the permutation vector.
            perm = onnxNDims + 1 - userDataPerm;
        elseif isequal(userDataPerm, 'auto')
            if onnxNDims == 2
                % Permute reverse ONNX CN to DLT CN (do nothing)
                perm = [];
            elseif onnxNDims == 4
                % Permute reverse onnx (WHCN) to MATLAB HWCN
                perm = [2 1 3 4];
            else
                % User wants the output in ONNX ordering, so just reverse it from
                % reverse onnx
                perm = onnxNDims:-1:1;
            end
        elseif isequal(userDataPerm, 'as-is')
            % Do not permute the input
            perm = 1:ndims(Y);
        else
            % userDataPerm is 'none', so just make it reverse onnx
            perm = onnxNDims:-1:1;
        end
end
if ~isempty(perm)
    Y = permute(Y, perm);
end
end

function s = updateStruct(s, t)
% Set all existing fields in s from fields in t, ignoring extra fields in t.
for name = transpose(fieldnames(s))
    s.(name{1}) = t.(name{1});
end
end
