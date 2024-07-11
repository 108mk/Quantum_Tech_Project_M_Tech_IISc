%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solution of the homogeneous, Fisher-KPP equation, using a direct application of the Carleman method
% combined with Chebyshev Spectral method. The result is compared with solutions from inbuilt MATLAB solvers.
%
% Code written by Manish Kumar in 2024.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clear global variables at the beginning
clear;

%% Plotting configuration
set(groot, 'defaultAxesTickLabelInterpreter','latex'); set(groot, 'defaultLegendInterpreter','latex');
fontsize = 14;

%% Simulation parameters: SPACE TIME DISCRETIZATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nx = 8; % Spatial discretization for PDE
nt_cheb = 64; % Temporal CHEBYSHEV intepolation points

% Spatial domain edges
L0 = (1); % Domain length
x0 = 0;
x1 = L0;

dx = (x1-x0)/(nx-1);
xs = linspace(x0, x1, nx);

% Temporal domain edges
T = (0.1);  % TOTAL Simulation time
t0 = (0); % START TIME
t1 = T;   % END TIME

dt = (t1-t0)/(nt_cheb-1);
ts = linspace(t0, t1, nt_cheb);

%% REACTION DIFFUSION PARAMETERS
%Re0 = (20); % Desired Reynolds number
D = (0.2); % DIFFUSION
aa = (0.4); % LINEAR TERM in f= aa*u + bb*u^2
bb = (-1); % QUADRATIC TERM in f= aa*u + bb*u^2

%% CARLEMAN TRUNCATION ORDER
N_max = 2; % Maximum Carleman truncation level
ode_deg = 2; % Degree of the Carleman ODE, should not be changed

Ns = 1:N_max; % Truncation levels
nu = D; %VISCOCITY/DIFFUSION cefficient
beta = aa; %linear damping term


%% CHEBYSHEV-method: Order and Discretization of Space and Time
K_max = 3; % CHEBYSHEV truncation order
Ks = 1:K_max;

%% RUNGE-KUTTA ode45 discretization: proxy for exact solver 
nt_ode = nt_cheb*400; % Make it more accurate than the Euler solution
dt_ode = (t1-t0)/(nt_ode-1);
ts_ode = linspace(t0, t1, nt_ode)';
dt = dt_ode;


%% Initial condition
U0 = (0.1); % u(x,0)=0.1*(1-cos(2*pi*x))
f = 1; % Number of oscillations in the initial distribution (below)
u0 = @(x) U0*(sin((f*pi)*x/L0).^2);
u0s = u0(xs);

Tnl = L0/U0; %Nonlinear time
t_plot = 2*Tnl/5; %Time to plot solution

%% Linear term matrix (Shifted and Scaled LAPLACIAN)
F1 = zeros(nx, nx);
F1(1+1:nx+1:end) = D/dx^2;
F1(1+nx:nx+1:end) = D/dx^2;
F1(1:nx+1:end) = -(2)*D/dx^2;
F1 = F1 + aa*eye(nx); % Add linear damping if present

%% Quadratic term matrix
F2 = zeros(nx ,nx^2);
F2(1:nx^2+nx+1:end)= bb;
F2 = reshape(F2,nx, nx^2);

% Weired boundary condition
% Enforce the Dirichlet boundaries within the domain.
% F0(1) = 0;
% F0(end) = 0;
%F1(1,:) = 0; %first row
%F1(end,:) = 0; %last row
%F2(1,:) = 0; %first row
%F2(end,:) = 0; %first row


%% ODE for ode45 solver
fischers_odefun = @(t,u) F1*u + F2*kron(u,u);

%% Check CFL condition
C1_ode = U0*dt_ode/dx;
C2_ode = 2*nu*dt_ode/dx^2;

if C1_ode > 1
    error(sprintf("C1_ode = %.2f\n",C1_ode));
end
if C2_ode > 1
    error(sprintf("C2_ode = %.2f\n",C2_ode));
end


%% Calculate the Carleman convergence number
lambdas = eig(F1);
lambdas = lambdas(lambdas ~= 0);
lambda = max(lambdas);

f2 = norm(F2);
f1 = norm(F1);
f0 = 0; % Homogenous equation 

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

%Calculate matrix block sizes for carleman matrix
dNs = zeros(N_max, 1);
for N = Ns
    dNs(N) = (nx^(N+1)-nx)/(nx-1);
end

%% First prepare the Carleman system with just the source term at t=0
A = spalloc(dNs(end),dNs(end),dNs(end)*nx);
Fs = [F1 F2];

for i = Ns %Carleman matrix block wise preparation
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


%% Scaling matrix to bring eigenvalue in range [-1, 1]
fprintf('matrix A size\n'); disp(size(A));

A_full = full(A); %Sparse matrix to full matrix
A_size = size(A_full);

% Eigenvalues of carleman matrix 
% real() to avoid a+0i data format used by matlab to store real numbers.
A_l = real(eig(A_full)); 

alpha = max(A_l)+min(A_l);
beta= max(A_l)-min(A_l);

%Scaling the matrix
A_scaled = (A_full - (0.5*(max(A_l)+min(A_l))*eye(A_size)))/((max(A_l)-min(A_l))*0.5); %eigenvalue scaled matrix into [-1, 1]

%Store similarity transformation matrix T for A = TDinv(T)
[V_scaled, D_scaled] = eig(A_scaled); %eigenvectors and diagonalized matrix
D_scaled = real(D_scaled);
inv_V_scaled = real(inv(V_scaled));

A_scaled_l = real(eig(A_scaled)); %list of eigenvalues of scaled matrix

% Check if matrix eigenvalue is scaled to [-1, 1]
if min(A_scaled_l) < -1.0001
    error(sprintf("scaling failure: min_eig = %.2f\n", min(A_scaled_l) ));
end

if max(A_scaled_l) > 1.0001
    error(sprintf("scaling failure: max_eig = %.2f\n", max(A_scaled_l) ));
end
    
t_in = 0; % initail time in scaled time domain
T_scaled = T*((max(A_l)-min(A_l))*0.5); % end time in scaled time domain
ts_scaled = linspace(t_in, T_scaled, nt_cheb ); % scaled time steps

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%SOLVING CARLEMAN ODE%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Solve Carleman system using CHEBYSHEV method
ys_c_N = zeros(K_max, nt_cheb, dNs(N_max));

%rescaling matrix in time
factor = exp(ts*alpha);
disp(size(factor));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for N = Ks
    A_N = A_scaled; % Truncated carleman matrix to N_max
    
    %% preparing Chebyshev matrix
    fprintf('Chebyshev matrix buildup start time\n');
    disp(datetime);
    CT = zeros(dNs(N_max), dNs(N_max), nt_cheb); % chebyshev product matrix initialize%
    
    for i = 1:nt_cheb
        temp = (besseli(0:N, ts_scaled(i)));
        
        for j = 1:dNs(N_max)
            cheb_arr = (chebyshevT(0:N, D_scaled(j,j)));
            CT(j, j, i) = CT(j, j, i) + dot( cheb_arr, (temp) ); %Clenshaw calulator
            %CT(j, j, i) = CT(j, j, i) + clenshaw(temp, D_scaled(j,j));
        end
    end
    
    fprintf('Chebyshev matrix buildup end time\n');
    disp(datetime);
    %----------- timestamping
    
    %% initial solution vector |y(0)>
    y0s = [];
    for i = 1:N_max
        y0s = [y0s, kronp(u0s, i)];
    end
    
    
    %% Time INTERPOLATION/INTEGRATION
    fprintf('Solving Chebyshev truncation for K=%d\n',N);

    ys = zeros(nt_cheb ,dNs(N_max));
    ys(1,:) = y0s;
    
    for k = 1:(nt_cheb-1)
        ys(k+1,:) = (( V_scaled*CT(:, :, k+1)*inv_V_scaled )*ys(1,:)')';
        ys(k+1,:) = ys(k+1,:).*factor(k+1);
    end
    
    %---------timestamping for numerical integration
    
    fprintf('Grand loop Done----------------------\n');
    
    ys_c_N(N, : ,1:dNs(N_max)) = real(ys(:,:));
end

us_c_N = ys_c_N(:, :, 1:nx); % extracting solution from carleman Solution vector

%%%%%%%%%%%%%% SAVE ARRAY for Future access %%%%%%%%%%%%%%%
%array_us_d = us_c_N;
%save('12_june_nx_16_nt_64_K_max_2.mat', 'array_us_d');

%for i = 1:K_max
%    us_c_N(i, :, : )=fliplr(us_c_N(i, :, :));
%end

%normalizer = max(us_c_N(1, 1, :))*(1/U0);
%us_c_N = us_c_N./normalizer;

%putting boundary condition explicitly
%for i = 1:K_max
%    us_c_N(i, :, 1) = zeros(1, nt_cheb, 1);
%    us_c_N(i, :, nx) = zeros(1, nt_cheb, 1);
%end


%% ODE for ''ode45'' solver
fischers_odefun = @(t,u) F1*u + F2*kron(u,u);

% Solve "exact" ODE
nt_ode = nt_cheb*400; % Make it more accurate as much as possible
t_in = 0; % scaled initail time
T_scaled = T*((max(A_l)-min(A_l))*0.5); % scaled end time
ts = linspace(t_in, T_scaled, nt_cheb ); % chebyshev interpolation interval

ts_ode = linspace(t_in, T, nt_ode ); %finer time interval for ODE45 solver
dt_ode = (T_scaled - t_in)/(nt_ode - 1);

fprintf('Solving "exact" ODE\n');
opts = odeset('RelTol', 1e-10, 'AbsTol', 1e-10);
[ts_ode, us_ode] = ode45(fischers_odefun, ts_ode, u0s, opts);

% explicit boundary condition imposing
us_ode(:, 1) =zeros(nt_ode, 1);
us_ode(:, nx) =zeros(nt_ode, 1);

% Interpolate so we can compare with other solutions
us_d = interp1(ts_ode, us_ode, ts);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% ERROR Estimation %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Calculate errors
% We will now calculate the error between all three solutions. First find
% their differences, and then take their norms over space. The variables
% are named according to:
%
% dus_ = difference
% eps_ = l_2 error
% eps_rel_ = l_inf error
%
% _c_ = Carleman solution
% _e_ = Direct Euler solution
% _d_ = ode45 solution
% _pde_ = pdepe solution
% _N = per Carleman level N
%nt_cheb=16;
dus_c_d_N = zeros(K_max,nt_cheb,nx);
dus_rel_c_d_N = zeros(K_max,nt_cheb,nx);
eps_c_d_N = zeros(K_max,nt_cheb);
eps_rel_c_d_N = zeros(K_max,nt_cheb);

lp =2; %two norm

for N = 1:K_max
    dus_c_d_N(N,:,:) = reshape(us_c_N(N,:,:),nt_cheb,nx) - us_d(:,:);
    dus_rel_c_d = reshape(dus_c_d_N(N,:,:),nt_cheb,nx)./us_d(:,:);
    dus_rel_c_d(isnan(dus_rel_c_d)) = 0;
    dus_rel_c_d_N(N,:,:) = dus_rel_c_d;
    
    for k = 1:nt_cheb
        eps_c_d_N(N,k) = norm(reshape(dus_c_d_N(N,k,:),nx,1), lp); % two norm
        eps_rel_c_d_N(N,k) = norm(reshape(dus_rel_c_d_N(N,k,:),nx,1), inf); % infinity norm
    end
end


%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%%%%%% PLOT %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%


% Find indices for which we will plot the solution
i_plot = find(ts_scaled>=t_plot, 1);
i_plot_pde = find(ts_ode>=t_plot,1);
i_start = ceil(i_plot*3/4);


%%%%%%%%
%%%%%%%%
figure(1)
ylim([-max(abs(us_ode(1,:))), max(abs(us_ode(1,:)))]);

%% Plot absolute l_2 error between Carleman and direct solution
for N = Ks
    ax = subplot(1,1,1);
    ax.ColorOrderIndex = N;
    semilogy(ts_scaled, eps_c_d_N(N,:), 'DisplayName', sprintf('Chebyshev truncation, $K=%d$', N));
    hold on;
end

% Format absolute l_2 error plot
subplot(1,1,1);
title(sprintf('Absolute error'), 'interpreter','latex');
xlabel('$t$', 'interpreter','latex');
ylabel('$\|\varepsilon_{\mathrm{abs}}\|_{\infty}$', 'interpreter','latex');
xline(t_plot,':','DisplayName','T_{nl}/3', 'HandleVisibility', 'Off');
%5ylim([min([min(eps_c_d_N(:,i_start:end)),eps_d_pde(i_start:end)])*0.1 max(eps_c_d_N(1,:))*10]);
lgd = legend();
set(lgd,'fontsize',fontsize-4);
set(gca,'fontsize',fontsize);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Format error convergence plot
% Plot time-maximum absolute l_2 error between Carleman and pdepe
figure(2)
ax = subplot(1,1,1);
semilogy(Ks, ((max(eps_c_d_N, [], 2)) ),'-o','DisplayName',sprintf('Time-maximum error')); %%%%####&&&&@@@@@@@@

subplot(1,1,1);
title(sprintf('Error convergence'), 'interpreter','latex');
xlabel('$K$', 'interpreter','latex');
ylabel('$\max_t \|\varepsilon_{\mathrm{abs}}\|_{\infty}$', 'interpreter','latex');
ax = gca;
lgd = legend();
set(gca,'fontsize',fontsize);
set(lgd,'fontsize',fontsize-4);

% Finalize and save
%sgtitle(sprintf('Reaction-Diffusion-solution with $$n_x=%d$, $n_t=%d$, $\\mathrm{R}=%.2f$',nx, nt_cheb, R), 'interpreter','latex', 'fontsize', fontsize+2);
%savefig(sprintf('rde_re0_%.2f_N_%d_nx_%d_nt_%d_rev2.fig',Re0,N_max,nx,nt));


%% 3D-PLOT for solution u(x,t) vs x vs t
temp = ys_c_N(K_max, 1:nt_cheb, 1:nx);
temp1 = squeeze(temp);
figure(3)
mesh(temp1);

title(sprintf('Solution 3D plot'), 'interpreter','latex');
xlabel('$n_x$', 'interpreter','latex');
ylabel('$n_t$', 'interpreter','latex');
zlabel('$u(n_x,n_t)$', 'interpreter','latex');

