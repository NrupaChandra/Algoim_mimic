function [h] = PlotPolynomial2D(p)
%PlotPolynomial2D creates a surface plot of the provided polynomial in
%power basis

    nodes = 100;
    [X, Y] = meshgrid(linspace(-1,1,nodes),linspace(-1,1,nodes));
    Z = zeros(size(X));
    for i=1:size(p,1)
        for j=1:size(p,2)
            Z = Z + p(i,j)*X.^(i-1).*Y.^(j-1);
        end
    end    
    
    h = figure('Visible','on'); hold on;
    surf(X,Y,Z);

end