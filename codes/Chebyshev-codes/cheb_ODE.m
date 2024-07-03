% Problem definition
f = @(x) x;
%analytical_solution = @(x) (10 / (9 * pi^2)) * sin(3 * pi * x);
N = 100; % Number of collocation points

% Chebyshev collocation points
x = cos(pi * (0:N) / N)';
% Chebyshev differentiation matrix
T = chebyshev_differentiation_matrix(N);
% Second-order differentiation matrix
%T2 = T^2;

% Boundary conditions
T = T(2:end-1, 2:end-1);

% Evaluate the forcing term at the collocation points
F = f(x);
F = F(2:end-1);

% Solve the linear system
u_inner = T \ F;
% Add boundary conditions back
u = [0; u_inner; 0];

% Plot the numerical and analytical solutions
xx = linspace(-1, 1, 200);
figure;
%plot(x, u, 'r-', xx, analytical_solution(xx), 'b-');
plot(x, u, 'ro');
legend('Numerical solution');
xlabel('x'); ylabel('u(x)');
title('Chebyshev collocation method for Poisson equation');