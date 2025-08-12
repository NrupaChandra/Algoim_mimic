function [h] = PlotBernsteinBox2D(p)
%PlotBernsteinBox2D Plots the control points of the polynomial provided in
%Bernstein basis

    nodes = size(p,1);
    [X, Y] = meshgrid(linspace(-1,1,nodes),linspace(-1,1,nodes)); 
    
    h = figure('Visible','on'); hold on;
    scatter3(X,Y,p);

end