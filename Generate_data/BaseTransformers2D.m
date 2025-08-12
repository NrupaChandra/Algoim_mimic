classdef BaseTransformers2D
    %contains functions to transform back and forth between different
    %polynomial bases 

    properties
        
    end

    methods
        function q = Legendre2Power(obj, p)
            % Legendre2Power converts 2D polynomial coefficients from orthonormal Legendre
            % basis to power basis using Legendre polynomials with orthonormality.
            %
            % INPUT:
            %   p - Matrix of coefficients in the orthonormal Legendre basis.
            %                    legendreCoeffs(m+1, n+1) corresponds to the coefficient
            %                    of P_m(x) P_n(y).
            %
            % OUTPUT:
            %   q - Matrix of coefficients in the power basis.
            %                 powerCoeffs(m+1, n+1) corresponds to the coefficient of x^m y^n.
            % assert(istril(flip(p)));

            [numRows, numCols] = size(p);
            q = zeros(size(p));
            maxDegree = max(numRows, numCols) - 1;
            
            % Precompute the transformation matrix for orthonormal Legendre to power
            T_inv = zeros(maxDegree+1, maxDegree+1);
            for m = 0:maxDegree                
                for k = 0:floor(m/2)
                    normFactor = sqrt((2*m + 1) / 2); % Orthonormal scaling
                    T_inv(m+1, (m-2*k)+1) = (normFactor * (1/2^m) * (-1)^k * ...
                        nchoosek(m,k) * nchoosek(2*m-2*k,m));
                end
            end
            
            % Transform each term (P_m(x) P_n(y)) to power basis
            for m = 0:numRows-1
                for n = 0:numCols-1
                    % if m + n <= maxDegree
                        % Coefficients for orthonormal P_m(x) in terms of x^n
                        Tx_inv = T_inv(m+1, 1:m+1);
                        % Coefficients for orthonormal P_n(y) in terms of y^n
                        Ty_inv = T_inv(n+1, 1:n+1);
                        % Combine contributions to form power basis
                        q(1:m+1, 1:n+1) = q(1:m+1, 1:n+1) + ...
                            p(m+1, n+1) * (Tx_inv' * Ty_inv);
                    % end
                end
            end
        end        

        function q = Power2Legendre(obj, p)
            % Power2Legendre converts 2D polynomial coefficients from power to orthonormal Legendre basis.
            %
            % INPUT:
            %   p - Matrix of coefficients in the power basis.
            %                 powerCoeffs(m+1, n+1) corresponds to the coefficient of x^m y^n.
            %
            % OUTPUT:
            %   q - Matrix of coefficients in the Legendre basis.
            %                    legendreCoeffs(m+1, n+1) corresponds to the coefficient
            %                    of P_m(x) P_n(y).
            % assert(istril(flip(p)));
        
            % Initialize Legendre coefficients
            [numRows, numCols] = size(p);
            q = zeros(size(p));
            maxDegree = max(numRows, numCols) - 1;
            
            % Precompute the transformation matrix for power to orthonormal Legendre
            T = zeros(maxDegree+1, maxDegree+1);
            for m = 0:maxDegree                
                for k = 0:floor(m/2)
                    normFactor = sqrt((2*m + 1) / 2); % Orthonormal scaling
                    T(m+1, (m-2*k)+1) = (normFactor * (1/2^m) * (-1)^k * ...
                        nchoosek(m,k) * nchoosek(2*m-2*k,m)); % obtained by expanding the Legendre polynomials in terms of a monomial basis
                end
            end
            T = T^(-1);

            % Transform each term (x^m y^n) to Legendre basis
            for m = 0:numRows-1
                for n = 0:numCols-1
                    % if m + n <= maxDegree
                        % Coefficients for x^m in terms of P_m(x)
                        Tx = T(m+1, 1:m+1);
                        % Coefficients for y^n in terms of P_n(y)
                        Ty = T(n+1, 1:n+1);
                        % Combine contributions to form Legendre basis
                        q(1:m+1, 1:n+1) = q(1:m+1, 1:n+1) + ...
                            p(m+1, n+1) * (Tx' * Ty);
                    % end
                end
            end
        end

        function q = Bernstein2Power(obj, p)
            % Bernstein2Power converts 2D polynomial coefficients from
            % Bernstein basis to power basis.
            %
            % INPUT:
            %   p - Matrix of coefficients in the orthonormal Legendre basis.
            %                    legendreCoeffs(m+1, n+1) corresponds to the coefficient
            %                    of P_m(x) P_n(y).
            %
            % OUTPUT:
            %   q - Matrix of coefficients in the power basis.
            %                 powerCoeffs(m+1, n+1) corresponds to the coefficient of x^m y^n.
            % assert(istril(flip(p)));

            [numRows, numCols] = size(p);
            q = zeros(size(p));
            maxDegree = max(numRows, numCols) - 1;
            
            % Precompute the transformation matrix for orthonormal Legendre to power
            T_inv = zeros(maxDegree+1, maxDegree+1);
            for k = 0:maxDegree                
                for j = k:maxDegree
                    for n = 0:j
                        T_inv(k+1, n+1) = T_inv(k+1, n+1) + nchoosek(maxDegree,j) * nchoosek(j,k) * ...
                            (-1)^(j-k) * 2^(-j) * nchoosek(j,n);
                    end
                end
            end
            
            % Transform each term (P_m(x) P_n(y)) to power basis
            for m = 0:numRows-1
                for n = 0:numCols-1
                    % if m + n <= maxDegree
                        % Coefficients for orthonormal P_m(x) in terms of x^n
                        Tx_inv = T_inv(m+1, 1:end);
                        % Coefficients for orthonormal P_n(y) in terms of y^n
                        Ty_inv = T_inv(n+1, 1:end);
                        % Combine contributions to form power basis
                        q(1:end, 1:end) = q(1:end, 1:end) + ...
                            p(m+1, n+1) * (Tx_inv' * Ty_inv);
                    % end
                end
            end
        end

        function q = Power2Bernstein(obj, p)
            % Power2Bernstein converts 2D polynomial coefficients from
            % power basis to bernstein basis.
            %
            % INPUT:
            %   p - Matrix of coefficients in the power basis.
            %                    legendreCoeffs(m+1, n+1) corresponds to the coefficient
            %                    of P_m(x) P_n(y).
            %
            % OUTPUT:
            %   q - Matrix of coefficients in the bernstein basis.
            %                 powerCoeffs(m+1, n+1) corresponds to the coefficient of x^m y^n.
            % assert(istril(flip(p)));

            [numRows, numCols] = size(p);
            q = zeros(size(p));
            maxDegree = max(numRows, numCols) - 1;
            
            % Precompute the transformation matrix for orthonormal Legendre to power
            T = zeros(maxDegree+1, maxDegree+1);
            for k = 0:maxDegree                
                for j = k:maxDegree
                    for n = 0:j
                        T(k+1, n+1) = T(k+1, n+1) + nchoosek(maxDegree,j) * nchoosek(j,k) * ...
                            (-1)^(j-k) * 2^(-j) * nchoosek(j,n);
                    end
                end
            end
            T = T^(-1);
            
            % Transform each term (P_m(x) P_n(y)) to power basis
            for m = 0:numRows-1
                for n = 0:numCols-1
                    % if m + n <= maxDegree
                        % Coefficients for orthonormal P_m(x) in terms of x^n
                        Tx = T(m+1, 1:end);
                        % Coefficients for orthonormal P_n(y) in terms of y^n
                        Ty = T(n+1, 1:end);
                        % Combine contributions to form power basis
                        q(1:end, 1:end) = q(1:end, 1:end) + ...
                            p(m+1, n+1) * (Tx' * Ty);
                    % end
                end
            end
        end
    end
end