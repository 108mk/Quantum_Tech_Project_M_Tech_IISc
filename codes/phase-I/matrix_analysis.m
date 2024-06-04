%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution of the homogeneous, Fisher-KPP equation, using a direct application of the Carleman method
% combined with Euler's method. The result is compared with solutions from inbuilt MATLAB solvers.
%
% Code written by Manish Kumar in 2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Plotting configuration
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
fontsize = 14;

%% Simulation parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nx = 4; % Spatial discretization for Euler's method for ODE
nt = 10000; % Temporal discretization for Euler's method for ODE

% for inbuilt PDE solver (Proxy for exact answer)
nx_pde = 100; % Spatial discretization for the 'pdepe' solver
nt_pde = 40000; % Temporal discretization for the 'pdepe' solver


Re0 = (20); % Desired Reynolds number
L0 = (1); % Domain length

D = (0.2); %diffusion
aa = (0.2); % f= a*u + b*u^2
bb = -1; % f= a*u + b*u^2
U0 = (0.1); % u(x,0)=0.1*(1-cos(2*pi*x))

f = 1; % Number of oscillations of the sinusoidal initial condition inside the domain

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_max = 5; % Maximum Carleman truncation level
ode_deg = 2; % Degree of the Carleman ODE, should not be changed

%% Initialize

Ns = 1:N_max; % Truncation levels

nu = D; %U0*L0/Re0; % Viscosity/Diffusion cefficient
beta = aa; %linear damping term

Tnl = L0/U0; % Nonlinear time
t_plot = 2*Tnl/5; % Time to plot solution

% Spatial domain edges
x0 = -L0/2;
x1 = L0/2;

% Temporal domain edges
t0 = (0);
T = (3); % Simulation time
t1 = T; %


% Euler's method discretization interval sizes and domains
dx = (x1-x0)/(nx-1);
dt = (t1-t0)/(nt-1);
xs = linspace(x0,x1,nx);
ts = linspace(t0,t1,nt);

% ode45 discretization: proxy for exact solver 
nt_ode = nt*10; % Make it more accurate than the Euler solution
dt_ode = (t1-t0)/(nt_ode-1);
ts_ode = linspace(t0,t1,nt_ode)';

% pdepe discretization interval sizes
dx_pde = (x1-x0)/(nx_pde-1); % Spatial discretization interval size for pdepe solver
dt_pde = (t1-t0)/(nt_pde-1);
xs_pde = linspace(x0,x1,nx_pde);
ts_pde = linspace(t0,t1,nt_pde);

%% Discretize Burger's equation

F0 = zeros(nt,nx);
%for it = 1:nt
 %   F0(it,:) = F0_fun(ts(it),xs); %time wise populate the F0 matrix
%end

% linear term matrix (LAPLACIAN)
F1 = zeros(nx,nx);
F1(1+1:nx+1:end) = D/dx^2;
F1(1+nx:nx+1:end) = D/dx^2;
F1(1:nx+1:end) = -(2)*D/dx^2;    %%%%%%%%%%%%%%%%%%%% edited
F1 = F1 + aa*eye(nx); % Add linear damping if present

% quadratic term matrix
F2 = zeros(nx,nx^2);
%F2((nx^2+nx+1):(nx^2+nx+1):end) = -1/(4*dx);  %%%%%%%%%%%%%%  edited
%F2(1+1:(nx^2+nx+1):end) = +1/(4*dx);           %%%%%%%%%%%%%  edited
F2(1:nx^2+nx+1:end)= bb;   %%%%%%%%%%%%%%%%%%  added
F2 = reshape(F2,nx,nx^2);

% Enforce the Dirichlet boundaries within the domain.
% F0(1) = 0;
% F0(end) = 0;
%F1(1,:) = 0; %first row
%F1(end,:) = 0; %last row
%F2(1,:) = 0; %first row
%F2(end,:) = 0; %first row

% Initial condition
u0 = @(x) U0*(sin((pi)*x/L0).^2);
u0s = u0(xs);

% ODE for ode45 solver
%F0_interp = @(t) interp1(ts,F0,t)'; %linear interpolation 
burgers_odefun = @(t,u) F1*u + F2*kron(u,u);

% PDE, initial condition and boundary condition for pdepe solver
burger_pde = @(x,t,u,dudx) deal(1, nu*dudx, -beta*u);    %%%%%%%%
burger_ic = u0;
burger_bc = @(xl, ul, xr, ur, t) deal(ul, 0, ur, 0);

%% Check CFL condition


% Dissipative and advective CFL numbers for Euler's method and pdepe
C1_e = U0*dt/dx;
C2_e = 2*nu*dt/dx^2;
C1_ode = U0*dt_ode/dx;
C2_ode = 2*nu*dt_ode/dx^2;
C1_pde = U0*dt_pde/dx_pde;
C2_pde = 2*nu*dt_pde/dx_pde^2;
if C1_e > 1
    error(sprintf("C1_e = %.2f\n",C1_e));
end
if C2_e > 1
    error(sprintf("C2_e = %.2f\n",C2_e));
end
if C1_ode > 1
    error(sprintf("C1_ode = %.2f\n",C1_ode));
end
if C2_ode > 1
    error(sprintf("C2_ode = %.2f\n",C2_ode));
end
if C1_pde > 1
    error(sprintf("C1_pde = %.2f\n",C1_pde));
end
if C2_pde > 1
    error(sprintf("C2_pde = %.2f\n",C2_pde));
end


%% Calculate the Carleman convergence number

lambdas = eig(F1);
%fprintf('printing eigenvalues of F1\n');
%disp(lambdas);

% Since we included the Dirichlet boundaries explicitly, the matrix F1 has
% a 2D nullspace corresponding to eigenvectors with non-zero boundary
% values. We remove these two zero-eigenvalues on the next line of
% code. We could also have avoided this by not explicitly including the
% zeroed boundaries in the integration domain, which corresponds to reducing
% nx by 2. This would increase lambda_1 according to the formula
%
% lambda_j = -nu*4/dx^2*sin(j*pi/(2*(nx+1-2)))^2.
%
% Instead we choose to keep the boundaries explicitly in the domain and just
% discard the zero eigenvalues. This will result in a larger R, so
% the claims in the paper are conservative relative to this. Note that the
% spectral norm of F2 is only marginally changed by the zeroing of
% boundaries.

lambdas = lambdas(lambdas ~= 0);
lambda = max(lambdas);

f2 = norm(F2);
f1 = norm(F1);
f0 = 0;

for it = 1:nt
    f0 = max(norm(F0(it,:)),f0);
end

R = ( norm(u0s)*f2 + f0/norm(u0s) )/abs(lambda);

r1 = (abs(lambda)-sqrt(lambda^2-4*f2*f0))/(2*f2);
r2 = (abs(lambda)+sqrt(lambda^2-4*f2*f0))/(2*f2);

if dt > 1/(N_max*f1)
    error('Time step too large');
end

if f0 + f2 > abs(lambda)
    fprintf('Perturbation too large\n');
end

%% Prepare Carleman matrix

fprintf('Preparing Carleman matrix\n');

% Calculate matrix block sizes
dNs = zeros(N_max,1);

for N = Ns
    dNs(N) = (nx^(N+1)-nx)/(nx-1);
end

% First prepare the Carleman system with just the source term at t=0
A = spalloc(dNs(end),dNs(end),dNs(end)*nx);

Fs = [F1 F2];

for i = Ns
    for j = 1:min(ode_deg, N_max-i+1)
        if i == 1 && j == 0
            continue;
        end
        a0 = 1+(nx^i-nx)/(nx-1);
        a1 = a0 + nx^i-1;
        b0 = 1+(nx^(j+i-1)-nx)/(nx-1);
        b1 = b0 + nx^(j+i-1)-1;

        Aij = spalloc(nx^i,nx^(i+j-1),nx^(i+j-1+1));
        f0 = 1+(nx^j-nx)/(nx-1);
        f1 = f0+nx^j-1;
        Fj = Fs(:,f0:f1);

        for p = 1:i
            Ia = kronp(sparse(eye(nx)), p-1);
            Ib = kronp(sparse(eye(nx)), i-p);
            Aij = Aij + kron(kron(Ia, Fj), Ib);
        end
        A(a0:a1, b0:b1) = Aij;
    end
end



%eigensA = eig(A);
fprintf('matrix A size\n');
disp(size(A));

[V, D] = eig(full(A));

%%% Conjugate equation
% A = transpose(A);

%% Checking no resonance condition

A = full(A);
% sub diagonal block matrices
A11 = A(1:nx, 1:nx);
A22 = A(1+nx:nx+nx^2, 1+nx:nx+nx^2);
A33 = A(1+nx+nx^2:nx+nx^2+nx^3, 1+nx+nx^2:nx+nx^2+nx^3);
A44 = A(1+nx+nx^2+nx^3:nx+nx^2+nx^3+nx^4, 1+nx+nx^2+nx^3:nx+nx^2+nx^3+nx^4);
A55 = A(1+nx+nx^2+nx^3+nx^4:nx+nx^2+nx^3+nx^4+nx^5, 1+nx+nx^2+nx^3+nx^4:nx+nx^2+nx^3+nx^4+nx^5);


% respective eigenvalue of the above
l1 = eig(A11);
l2 = eig(A22);
l3 = eig(A33);
l4 = eig(A44);
l5 = eig(A55);

fprintf('result for A11 and A22 \n')
[val,pos]=intersect(l1, l2)

fprintf('result for A11 and A33 \n')
[val,pos]=intersect(l1, l3)

fprintf('result for A11 and A44 \n')
[val,pos]=intersect(l1, l4)

fprintf('result for A11 and A55 \n')
[val,pos]=intersect(l1, l5)






