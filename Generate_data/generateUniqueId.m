% Function to generate a unique identifier from the defining data values
function id = generateUniqueId(poly)
    % Ensure poly is numeric
    if ~isnumeric(poly)
        error('Input "poly" must be a numeric array.');
    end
    
    % Flatten the poly array into a single vector
    polyData = poly(:)';
    
    % Convert to uint8 for hashing
    byteData = typecast(polyData, 'uint8');
    
    % Create SHA256 hasher
    hasher = System.Security.Cryptography.MD5.Create();
    
    % Compute hash
    hashBytes = hasher.ComputeHash(byteData);
    hashBytes = double(hashBytes);
    % hashBytes = hashBytes(1:16);

    % Convert hash to hexadecimal string
    id = sprintf('%02x', hashBytes);
end