% 主程序
N_values = [10, 20, 40, 80];
u_exact = @(x) x.^2 .* (x - 1).^2;
u_exact_prime = @(x) 2 * x .* (x - 1) .* (2 * x - 1);
f = @(x) 2 * (6 * x.^2 - 6 * x + 1);

L2_errors = zeros(length(N_values), 1);
H1_errors = zeros(length(N_values), 1);

% 绘制u_exact
x = linspace(0, 1, 80);
figure;
plot(x, u_exact(x), 'r-');
hold on;
title('Exact solution');

for i = 1:length(N_values)
    N = N_values(i);
    
    % 组装刚度矩阵和右端项
    [A, F] = assemble_matrices(N, f);
    
    % 求解线性方程组
    u_h = -A \ F;

    % 绘制u_h
    x = generate_grid(N);
    figure;
    plot(x, u_h, 'o-');
    hold on;
    title(['N = ', num2str(N)]);
    
    % 计算u_h的导数
    x = generate_grid(N);
    u_h_prime = zeros(N+2, 1);
    
    % 逐个区间计算导数
    for j = 2:N+2
        xj = x(j-1);
        xj1 = x(j);
        
        % 计算数值解在每个区间上的导数
        u_h_prime(j-1) = (u_h(j) - u_h(j-1)) / (xj1 - xj);
    end
    
    % 计算 L2 和 H1 误差
    L2_errors(i) = compute_L2_error(u_exact, u_h, N);
    H1_errors(i) = compute_H1_error(u_exact, u_h, u_exact_prime, u_h_prime, N);
    
    % 计算误差的阶数
    if i > 1
        L2_order = log(L2_errors(i-1) / L2_errors(i)) / log(N_values(i) / N_values(i-1));
        H1_order = log(H1_errors(i-1) / H1_errors(i)) / log(N_values(i) / N_values(i-1));
        fprintf('N = %d, L2_error = %.5e, L2_order = %.2f, H1_error = %.5e, H1_order = %.2f\n', ...
            N, L2_errors(i), L2_order, H1_errors(i), H1_order);
    else
        fprintf('N = %d, L2_error = %.5e, H1_error = %.5e\n', N, L2_errors(i), H1_errors(i));
    end
    
    % 保存旧的误差值
    L2_error_old = L2_error;
    H1_error_old = H1_error;
    N_old = N;
end

function [x] = generate_grid(N)
    % 生成等距网格 [0, 1]，包括端点
    x = linspace(0, 1, N + 2);
end

function [N_local, B_local] = local_basis_functions(xi, xi1)
    syms x;
    
    % 线性形函数 N1 和 N2
    N1 = (xi1 - x) / (xi1 - xi);  % 在 xi 处为 1，在 xi1 处为 0
    N2 = (x - xi) / (xi1 - xi);   % 在 xi1 处为 1，在 xi 处为 0

    % 导数
    B1 = diff(N1, x);
    B2 = diff(N2, x);
    
    % 将符号函数转化为数值函数
    N_local = {matlabFunction(N1, 'Vars', x), matlabFunction(N2, 'Vars', x)};
    B_local = {matlabFunction(B1, 'Vars', x), matlabFunction(B2, 'Vars', x)};
end

function [A, F] = assemble_matrices(N, f)
    % 生成网格节点
    x = generate_grid(N);
    
    % 初始化刚度矩阵和右端项向量
    A = zeros(N+2, N+2);
    F = zeros(N+2, 1);
    
    % 循环遍历每个区间，计算局部刚度矩阵和右端项
    for i = 2:N+2
        xi = x(i-1);
        xi1 = x(i);
        
        % 获取局部形函数和导数
        [N_local, B_local] = local_basis_functions(xi, xi1);
        
        % 局部刚度矩阵和右端项初始化
        A_local = zeros(2, 2);
        F_local = zeros(2, 1);
        
        % 计算局部刚度矩阵
        for j = 1:2
            for k = 1:2
                A_local(j, k) = integral(@(x) B_local{j}(x) .* B_local{k}(x), xi, xi1, 'ArrayValued', true);
            end
            % 计算局部右端项
            F_local(j) = integral(@(x) N_local{j}(x) .* f(x), xi, xi1, 'ArrayValued', true);
        end
        
        % 将局部刚度矩阵和右端项组装到全局矩阵
        global_indices = [i-1, i];  % 每个单元上的两个节点
        A(global_indices, global_indices) = A(global_indices, global_indices) + A_local;
        F(global_indices) = F(global_indices) + F_local;
    end
    % 添加积分均值条件约束，修正解的唯一性问题
    % A(N+2, :) = 1;
    % A(N+2, 1) = 1/2;
    % A(N+2, N+2) = 1/2;

    % F(N+2) = integral(@(x) x.^2 .* (x - 1).^2, 0, 1);

    A(N+2,:)=0;
    A(N+2,1)=1;
    F(N+2)=0;
end

function L2_error = compute_L2_error(u_exact, u_h, N)
    % 生成网格
    [x] = generate_grid(N);
    
    % 定义误差
    L2_error = 0;
    
    % 逐个区间计算 L2 误差
    for i = 1:N+1
        xi = x(i);
        xi1 = x(i+1);
        
        % 数值解
        [N_local,~] = local_basis_functions(xi, xi1);
        u_num = @(x) u_h(i)*N_local{1}(x) + u_h(i+1)*N_local{2}(x);
        
        % L2 误差计算
        L2_error = L2_error + integral(@(x) (u_num(x) - u_exact(x)).^2, xi, xi1);
    end

    % 最终 L2 误差
    L2_error = sqrt(L2_error);
end

function H1_error = compute_H1_error(u_exact, u_h, u_exact_prime, u_h_prime, N)
    % 1. 计算 L2 误差部分
    L2_error = compute_L2_error(u_exact, u_h, N);

    % 2. 计算导数的 L2 误差部分，调用 compute_L2_error 来计算
    L2_error_derivative = compute_L2_error(u_exact_prime, u_h_prime, N);

    % 3. 最终的 H1 误差，结合 L2 误差部分和导数误差部分
    H1_error = sqrt(L2_error^2 + L2_error_derivative^2);
end