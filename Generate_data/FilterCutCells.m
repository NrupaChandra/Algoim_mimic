function [FilteredCoeffs] = FilterCutCells(Coeffs, BasisType, numnodes)
%FilterCutCells Filters coefficients for cut cells. Only coefficient sets that have a zero (detected by the signs of bernstein coefficients) set in the reference domain remain
%Coeffs is a cell array, with each cell defining a coefficients set
%BasisType specifies wether a 'Bernstein', 'Legendre' or 'Power' basis is
%supplied

if nargin ~= 3
    nodes = 50;
else 
    nodes = numnodes;
end

T = BaseTransformers2D;
filter = zeros(numel(Coeffs),1);
for i=1:numel(Coeffs)

    % coeff = Coeffs{i};
    % 
    % if BasisType=="Legendre" 
    %     coeff = T.Legendre2Power(coeff);
    %     coeff = T.Power2Bernstein(coeff);
    % elseif BasisType=="Power"
    %     coeff = T.Power2Bernstein(coeff);
    % end
    % 
    % if any(coeff > 1e-4, [1,2]) && any(coeff < -1e-4, [1,2])
    %     filter(i) = 1;
    % end

    coeff = Coeffs{i};

    if BasisType=="Legendre" 
        coeff = T.Legendre2Power(coeff);
    elseif BasisType=="Bernstein"
        coeff = T.Bernstein2Power(coeff);
    end
    
    [X, Y] = meshgrid(linspace(-1,1,nodes),linspace(-1,1,nodes));
    Z = zeros(size(X));
    for k=1:size(coeff,1)
        for j=1:size(coeff,2)
            Z = Z + coeff(k,j)*X.^(k-1).*Y.^(j-1);
        end
    end   
    if any(Z > 0, [1,2]) && any(Z < 0, [1,2])
        filter(i) = 1;
    end
end
FilteredCoeffs = Coeffs(filter==1);

end