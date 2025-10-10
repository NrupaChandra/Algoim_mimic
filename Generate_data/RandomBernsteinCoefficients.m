clc
clear

apply_monotone_filter = true; 
monotone_mode = 'both+';
monotone_tol  = 1e-9;
time = false;
export = false;
k = 1;
N = 1000000;
filebase = '1MTestBernstein_';

T = BaseTransformers2D;

%% generate random coefficients
Coeffs = cell(k+1,N);
rng(3, 'twister');
if time 
    tic 
end
for i=1:k
    for j=1:N
        % if j <= 10
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(ceil((i+1)/2),ceil((i+1)/2)) = -10 * rand(1,1);
        % elseif j <= 20
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(1:i+1+1:end) = -2 * diag(rand(i+1,i+1));
        % elseif j <= 30
        %     Coeffs{i+1,j} = tril(-rand(i+1,i+1),-1)+triu(rand(i+1,i+1),0);
        % elseif j <= 40
        %     Coeffs{i+1,j} = tril(-rand(i+1,i+1),-2)+triu(rand(i+1,i+1),-1);
        % elseif j <= 50
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(:,1) = -rand(i+1,1);
        % elseif j <= 60
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(1,1) = -rand(1,1);
        % elseif j <= 70
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(1,1) = -rand(1,1);
        %     Coeffs{i+1,j}(end,1) = -rand(1,1);
        % elseif j <= 80
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(1,1) = -rand(1,1);
        %     Coeffs{i+1,j}(end,1) = -rand(1,1);
        %     Coeffs{i+1,j}(end,end) = -rand(1,1);
        % elseif j <= 90
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(1,1) = -rand(1,1);
        %     Coeffs{i+1,j}(end,1) = -rand(1,1);
        %     Coeffs{i+1,j}(end,end) = -rand(1,1);
        %     Coeffs{i+1,j}(1,end) = -rand(1,1);
        % elseif j <= 100
        %     Coeffs{i+1,j} = rand(i+1,i+1);
        %     Coeffs{i+1,j}(end,1) = -rand(1,1);
        %     Coeffs{i+1,j}(ceil((i+1)/2),ceil((i+1)/2)) = -rand(1,1);
        %     Coeffs{i+1,j}(end,end) = -rand(1,1);
        % end
        % Coeffs{i+1,j} = flip(Coeffs{i+1,j}); % just so the point assigned to the lower left corner corresponds to that corner in the graphical representation
        Coeffs{i+1,j} = 2 * rand(i+1) - 1;
    end
end
if time
    toc
end
%% filter coefficients to exclude those most likely not containing an interface
if time
    tic
end
filteredCoeffs = cell(k+1,1);
for i=1:size(Coeffs,1)
    if ~isempty(Coeffs(i,:))
        filteredCoeffs{i} = FilterCutCells(Coeffs(i,:), 'Bernstein');
    end
end
%%  filter by monotonicity in Bernstein space
if apply_monotone_filter
    for i = 1:numel(filteredCoeffs)
        if ~isempty(filteredCoeffs{i})
            % Keep only monotone ones
            filteredCoeffs{i} = FilterMonotoneBernstein(filteredCoeffs{i}, monotone_mode, monotone_tol);
        end
    end
end

if time
    toc
end

if time 
    tic 
end

%% write coefficients to file(s)
for i=1:numel(filteredCoeffs)
    p = i-1;

    ids = [];
    if ~exist([filebase,'p',num2str(p),'_data.txt'],'file')
        % write table data
        file = fopen([filebase,'p',num2str(p),'_data.txt'],'wt');
        fprintf(file,"number;id;exp_x;exp_y;coeff\n");
    else
        file = fopen([filebase,'p',num2str(p),'_data.txt'],'at');
        data = readtable([filebase,'p',num2str(p),'_data.txt'], 'Delimiter', ';', 'ReadVariableNames', true);        
        ids = data.id;
    end

    count = 1 + numel(ids);
    for j=1:numel(filteredCoeffs{i}) 
        [N,M] = size(filteredCoeffs{i}{j});
        poly = zeros(3,N*M);
        C = T.Bernstein2Power(filteredCoeffs{i}{j});
        for n=1:N
            for m=1:M
                poly(1,(m-1)*N+n) = n-1;
                poly(2,(m-1)*N+n) = m-1;
                poly(3,(m-1)*N+n) = C(n,m);
            end
        end
        id = generateUniqueId(poly);
        if isempty(ids) || ~ismember(id,ids)
            fprintf(file,[num2str(count),';',id,';',regexprep(num2str(poly(1,:)),'\s+',','),';',regexprep(num2str(poly(2,:)),'\s+',','),';',regexprep(num2str(poly(3,:)),'\s+',','),'\n']);
            count = count + 1;
        end
        poly(3,:) = -poly(3,:);
        id = generateUniqueId(poly);
        if isempty(ids) || ~ismember(id,ids)
            fprintf(file,[num2str(count),';',id,';',regexprep(num2str(poly(1,:)),'\s+',','),';',regexprep(num2str(poly(2,:)),'\s+',','),';',regexprep(num2str(poly(3,:)),'\s+',','),'\n']);
            count = count + 1;
        end
    end        
    
    fclose(file);

    if export
        if ~exist([filebase,'p',num2str(p)], 'dir')
           mkdir([filebase,'p',num2str(p)]);
        end
        data = readtable([filebase,'p',num2str(p),'_data.txt'], 'Delimiter', ';', 'ReadVariableNames', true);
        for j=1:size(data,1)       
            row = data(j,:);
            expX = str2num(row.exp_x{:});
            expY = str2num(row.exp_y{:});
            coeff = str2num(row.coeff{:});
            N = max(expX)+1;
            M = max(expY)+1;
            C = zeros(N,M);
            for k=1:numel(expX)
                n = expX(k)+1;
                m = expY(k)+1;
                C(n,m) = C(n,m) + coeff(k);
            end
            PolyExport2D([pwd '/' filebase 'p',num2str(p),'/',row.id{:}], C, false);
        end
    end
end
if time
    toc
end