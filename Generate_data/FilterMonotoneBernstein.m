function out = FilterMonotoneBernstein(inCells, mode, tol)

if nargin < 2 || isempty(mode), mode = 'both+'; end
if nargin < 3 || isempty(tol),  tol = 1e-12; end

out = {};
for t = 1:numel(inCells)
    C = inCells{t};
    if isempty(C), continue; end
    [N,M] = size(C); n = N-1; m = M-1;

    Dx = n * diff(C,1,1);   
    Dy = m * diff(C,1,2);   

    switch mode
        case {'both+','nondecreasing'}
            ok = all(Dx(:) >= -tol) && all(Dy(:) >= -tol);
        case {'both-','nonincreasing'}
            ok = all(Dx(:) <=  tol) && all(Dy(:) <=  tol);
        case 'x+'
            ok = all(Dx(:) >= -tol);
        case 'x-'
            ok = all(Dx(:) <=  tol);
        case 'y+'
            ok = all(Dy(:) >= -tol);
        case 'y-'
            ok = all(Dy(:) <=  tol);
        otherwise
            error('Unknown mode: %s', mode);
    end

    if ok
        out{end+1} = C; 
    end
end
end
