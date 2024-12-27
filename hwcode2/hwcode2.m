% 主程序，计算误差和阶数
N_values = [10, 20, 40, 80];
exact_u = @(x) (x - 1).*sin(x);
exact_u_prime = @(x) (x - 1).*cos(x) + sin(x);
f = @(x) -(2*cos(x) - (x - 1).*sin(x));

L2_errors = zeros(length(N_values), 1);
H1_errors = zeros(length(N_values), 1);

for i = 1:length(N_values)
    N = N_values(i);
    
    % 求解数值解和导数
    [u_h, u_h_derivative] = solve_fem(N, f);

    % 计算 L2 和 H1 误差
    L2_errors(i) = compute_L2_error(exact_u, u_h, N);
    H1_errors(i) = compute_H1_error(exact_u, u_h, exact_u_prime, u_h_derivative, N);

    % 计算误差的阶数
    if i > 1
        L2_order = log(L2_errors(i-1) / L2_errors(i)) / log(N_values(i) / N_values(i-1));
        H1_order = log(H1_errors(i-1) / H1_errors(i)) / log(N_values(i) / N_values(i-1));
        fprintf('N = %d, L2_error = %.5e, L2_order = %.2f, H1_error = %.5e, H1_order = %.2f\n', ...
            N, L2_errors(i), L2_order, H1_errors(i), H1_order);
    else
        fprintf('N = %d, L2_error = %.5e, H1_error = %.5e\n', N, L2_errors(i), H1_errors(i));
    end
end

function [x, mid_x] = generate_grid(N)
    % 生成网格节点
    x = linspace(0, 1, N+1); % N+1 个点，包括端点
    mid_x = (x(1:end-1) + x(2:end)) / 2; % 生成每个区间的中点
end

function [N_local, B_local] = local_basis_functions(xi, mid_xi, xi1)
    % 局部形函数
    syms x
    N0_sym = (x - mid_xi)*(x - xi1) / ((xi - mid_xi)*(xi - xi1));
    N1_2_sym = (x - xi)*(x - xi1) / ((mid_xi - xi)*(mid_xi - xi1));
    N1_sym = (x - xi)*(x - mid_xi) / ((xi1 - xi)*(xi1 - mid_xi));
    
    % 形函数导数
    B0_sym = diff(N0_sym, x);
    B1_2_sym = diff(N1_2_sym, x);
    B1_sym = diff(N1_sym, x);

    % 将符号表达式转换为数值函数
    N_local = {
        matlabFunction(N0_sym, 'Vars', x), ...
        matlabFunction(N1_2_sym, 'Vars', x), ...
        matlabFunction(N1_sym, 'Vars', x)
    };
    B_local = {
        matlabFunction(B0_sym, 'Vars', x), ...
        matlabFunction(B1_2_sym, 'Vars', x), ...
        matlabFunction(B1_sym, 'Vars', x)
    };
end


function [A, F] = assemble_matrices(N, f)
    % 生成网格
    [x, mid_x] = generate_grid(N);
    
    % 初始化全局矩阵
    A = zeros(2*N + 1, 2*N + 1);
    F = zeros(2*N + 1, 1);
    
    % 循环计算局部刚度矩阵和右端项
    for i = 1:N
        xi = x(i);
        mid_xi = mid_x(i);
        xi1 = x(i+1);
        
        % 获取局部形函数及导数的数值函数
        [N_local, B_local] = local_basis_functions(xi, mid_xi, xi1);
        
        % 定义局部刚度矩阵 Ai 和局部右端项 Fi
        A_local = zeros(3, 3);
        F_local = zeros(3, 1);
        
        % 计算局部刚度矩阵
        for j = 1:3
            for k = 1:3
                A_local(j, k) = integral(@(x) B_local{j}(x) .* B_local{k}(x), xi, xi1);
            end
            % 计算局部右端项 Fi
            F_local(j) = integral(@(x) N_local{j}(x) .* f(x), xi, xi1);
        end
        
        % 组装到全局矩阵
        global_indices = [2*i-1, 2*i, 2*i+1];
        A(global_indices, global_indices) = A(global_indices, global_indices) + A_local;
        F(global_indices) = F(global_indices) + F_local;
    end
end

function [u_h, u_h_derivative] = solve_fem(N, f)
    % 组装刚度矩阵和右端项
    [A, F] = assemble_matrices(N, f);
    
    % 解线性方程组
    u_h = solve_system(A, F);
    
    % 计算导数
    [x, mid_x] = generate_grid(N);
    u_h_derivative = zeros(size(u_h));
    
    for i = 1:N
        xi = x(i);
        mid_xi = mid_x(i);
        xi1 = x(i+1);
        
        % 局部导数
        [~, B_local] = local_basis_functions(xi, mid_xi, xi1);
        
        % 计算数值导数
        u_h_derivative(2*i-1:2*i+1) = [
            B_local{1}(xi), B_local{2}(mid_xi), B_local{3}(xi1)
        ];
    end
end

function u_h = solve_system(A, F)
    % 删去边界条件对应的行和列 (第1行/列和最后1行/列)
    A_reduced = A(2:end-1, 2:end-1);
    F_reduced = F(2:end-1);
    
    % 求解线性方程组
    u_reduced = A_reduced \ F_reduced;
    
    % 插入边界条件
    u_h = [0; u_reduced; 0]; % u(0) = u(1) = 0
end

function [L2_error, H1_error] = compute_errors(N, u_h, exact_u, exact_u_prime)
    % 生成网格
    [x, mid_x] = generate_grid(N);
    
    % 定义误差项
    L2_error = 0;
    H1_error = 0;
    
    for i = 1:N
        xi = x(i);
        mid_xi = mid_x(i);
        xi1 = x(i+1);
        
        % 获取数值解的局部形函数
        [N_local, B_local] = local_basis_functions(xi, mid_xi, xi1);
        
        % 计算数值解
        u_num = @(x) u_h(2*i-1)*N_local{1}(x) + u_h(2*i)*N_local{2}(x) + u_h(2*i+1)*N_local{3}(x);
        u_prime_num = @(x) u_h(2*i-1)*B_local{1}(x) + u_h(2*i)*B_local{2}(x) + u_h(2*i+1)*B_local{3}(x);
        
        % L2 误差
        L2_error = L2_error + integral(@(x) (u_num(x) - exact_u(x)).^2, xi, xi1);
        
        % H1 误差
        H1_error = H1_error + integral(@(x) (u_prime_num(x) - exact_u_prime(x)).^2, xi, xi1);
    end
    
    L2_error = sqrt(L2_error);
    H1_error = sqrt(H1_error);
end

function L2_error = compute_L2_error(exact_u, u_h, N)
    % 生成网格
    [x, mid_x] = generate_grid(N);
    
    % 定义误差
    L2_error = 0;
    
    % 逐个区间计算 L2 误差
    for i = 1:N
        xi = x(i);
        mid_xi = mid_x(i);
        xi1 = x(i+1);
        
        % 数值解
        [N_local, ~] = local_basis_functions(xi, mid_xi, xi1);
        u_num = @(x) u_h(2*i-1)*N_local{1}(x) + u_h(2*i)*N_local{2}(x) + u_h(2*i+1)*N_local{3}(x);
        
        % L2 误差计算
        L2_error = L2_error + integral(@(x) (u_num(x) - exact_u(x)).^2, xi, xi1);
    end
    
    % 最终 L2 误差
    L2_error = sqrt(L2_error);
end

function H1_error = compute_H1_error(exact_u, u_h, exact_u_prime, u_h_prime, N)
    % 生成网格
    [x, mid_x] = generate_grid(N);
    
    % 定义误差
    H1_error = 0;
    
    % 逐个区间计算 H1 误差
    for i = 1:N
        xi = x(i);
        mid_xi = mid_x(i);
        xi1 = x(i+1);
        
        % 数值解导数
        [~, B_local] = local_basis_functions(xi, mid_xi, xi1);
        u_prime_num = @(x) u_h(2*i-1)*B_local{1}(x) + u_h(2*i)*B_local{2}(x) + u_h(2*i+1)*B_local{3}(x);
        
        % H1 误差计算
        H1_error = H1_error + integral(@(x) (u_prime_num(x) - exact_u_prime(x)).^2, xi, xi1);
    end
    
    % 最终 H1 误差
    H1_error = sqrt(H1_error);
end

