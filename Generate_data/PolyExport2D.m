function [] = PolyExport2D(filename, p, overwrite)
%EXPORT png data for the level set in a cell, given the polynomial
%coefficients, p(i,j) are the coefficients for x^(i-1)*y^(j-1), nodes the
%number of export nodes
%Additionally csv files are saved of the coefficients arranged in an array corresponding to
% [x^0y^0, x^0y^1, ...; x^1y^0; ...; ...x^(n-1)y^(n-1)]

if nargin < 3
    overwrite = true;
end

T = BaseTransformers2D;
if overwrite || ~exist([filename,'_Power','.csv'], 'file')
    q = p;
    writematrix(q,[filename,'_Power','.csv']);
end

if overwrite || ~exist([filename,'_Bernstein','.csv'], 'file')
    q = T.Power2Bernstein(p);
    writematrix(q,[filename,'_Bernstein','.csv']);
end

if overwrite || ~exist([filename,'_Legendre','.csv'], 'file')
    q = T.Power2Legendre(p);
    writematrix(q,[filename,'_Legendre','.csv']);
end

if overwrite || ~exist([filename,'.png'], 'file')
    nodes = 50;
    [X, Y] = meshgrid(linspace(-1,1,nodes),linspace(-1,1,nodes));
    Z = zeros(size(X));
    for i = 1:size(p,1)
        for j=1:size(p,2)
            Z = Z + p(i,j)*X.^(i-1).*Y.^(j-1);
        end
    end
    
    % Result = zeros(numel(Z),3);
    % for i=1:numel(Z)
    %     Result(i,1) = X(i);
    %     Result(i,2) = Y(i);
    %     Result(i,3) = Z(i);
    % end
    % writematrix(Result, filename);
    
    h = figure('Visible','off'); hold on;
    contourf(X, Y, Z, [-Inf, 0], 'FaceColor', 'white', 'FaceAlpha', 0.5);
    contourf(X, Y, Z, [0, Inf], 'FaceColor', 'blue', 'FaceAlpha', 0.5);
    contour(X,Y,Z,[0 0], 'LineColor', 'r', 'LineWidth', 1.5);
    
    saveas(h, filename, 'png');
end

end