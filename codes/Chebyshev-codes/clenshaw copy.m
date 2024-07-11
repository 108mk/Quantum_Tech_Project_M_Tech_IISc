function result = clenshaw(a, x)
    % Get the initial values of c0 and c1
    c0 = a(end - 1);  
    c1 = a(end);      

    % Iterate through the coefficients in reverse order
    for i = 3:numel(a)  
        temp = c0;   % Store c0 temporarily
        c0 = a(end - i + 1) - c1;   % Update c0
        c1 = temp + c1 * (2 * x);  % Update c1
    end

    % Calculate the final result
    result = c0 + c1 * x; 
end