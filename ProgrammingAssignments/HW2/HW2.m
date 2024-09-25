% 主程序
f = @(x) 2 * cos(x) - (x - 1) .* sin(x);
u_exact = @(x) (x - 1) .* sin(x);
u_exact_derivative = @(x) (x - 1) .* cos(x) + sin(x);

N_values = [10,20,40,80];
L2_errors = zeros(size(N_values));
H1_errors = zeros(size(N_values));

for i = 1:length(N_values)
    N = N_values(i);
    [u_h,u_h_derivative] = solve_fem(N, f);

    L2_errors(i) = compute_L2_error(u_exact, u_h);
    H1_errors(i) = compute_H1_error(u_exact, u_h, u_exact_derivative, u_h_derivative);

    if i > 1
        L2_order = log(L2_errors(i-1)/L2_errors(i)) / log(N_values(i)/N_values(i-1));
        H1_order = log(H1_errors(i-1)/H1_errors(i)) / log(N_values(i)/N_values(i-1));
        fprintf('N = %d, L2_error = %.5e, L2_order = %.2f, H1_error = %.5e, H1_order = %.2f\n', ...
            N, L2_errors(i), L2_order, H1_errors(i), H1_order);
    else
        fprintf('N = %d, L2_error = %.5e, H1_error = %.5e\n', N, L2_errors(i), H1_errors(i));
    end
end

% 生成网格
function [x, h] = generate_mesh(N)
% N个内部节点，N+1个区间，N+2个节点， 索引从 1 起
x = linspace(0, 1, N+2); % 均匀划分区间 [0, 1]
h = 1 / (N+1);               % 网格步长
end

% 组装矩阵，局部刚度矩阵计算
function [A, F] = assemble_matrix(N, f)
    [x, h] = generate_mesh(N);
    A = zeros(N, N); % N+1个区间对应N+2个节点
    F = zeros(N, 1);   % 右端项

    % 遍历所有区间
    for i = 2:N % 遍历N+1个区间，但把1和N+1的情形单独拿出来讨论
        % 计算局部刚度矩阵
        A_local = [1 -1; -1 1] / h; % 局部刚度矩阵, 每个区间的大小都是 h
        F_local = zeros(2, 1); % 初始化局部右端项
        
        % 使用数值积分计算局部右端项 F_local
        F_local(1) = integral(@(x_var) f(x_var) .* basis_function_local(x_var, x(i), x(i+1), 0), x(i), x(i+1));
        F_local(2) = integral(@(x_var) f(x_var) .* basis_function_local(x_var, x(i), x(i+1), 1), x(i), x(i+1));
        
        % 将局部刚度矩阵添加到全局刚度矩阵
        A(i-1:i, i-1:i) = A(i-1:i, i-1:i) + A_local;
        
        % 将局部右端项添加到全局右端项
        F(i-1:i) = F(i-1:i) + F_local;
    end
    A(1,1)=A(1,1)+1/h;
    A(N,N)=A(N,N)+1/h;
    F(1,1)=F(1,1)+ integral(@(x_var) f(x_var) .* basis_function_local(x_var, x(1), x(2), 1), x(1), x(2));
    F(N,1)=F(N,1)+ integral(@(x_var) f(x_var) .* basis_function_local(x_var, x(N+1), x(N+2), 0), x(N+1), x(N+2));
end

% 局部基函数
function phi_local = basis_function_local(x_var, x_left, x_right, index)
    phi_local = zeros(size(x_var));

    % 判断 x_var 在哪个区间上并计算基函数值
    if index == 0
        % 斜向下局部基函数
        phi_local(x_var > x_left & x_var < x_right) = ...
            (x_right - x_var(x_var > x_left & x_var < x_right)) / (x_right - x_left);
    elseif index == 1
        % 斜向上局部基函数
        phi_local(x_var > x_left & x_var < x_right) = ...
            (x_var(x_var > x_left & x_var < x_right) - x_left) / (x_right - x_left);
    else
        error('基函数索引超出范围');
    end
end

% 求解有限元问题
function [u_h, u_h_derivative] = solve_fem(N, f)
[x, h] = generate_mesh(N);
[A, F] = assemble_matrix(N, f);
u_h_list = -A \ F; % 求解线性方程
% 定义 u_h 为一个匿名函数，通过基函数的线性组合得到
u_h = @(x_var) 0;  % 初始化为 0 函数

% 遍历所有基函数，将它们与对应的系数组合
for i = 1:N
    % 将每个基函数与系数 u_h_list(i-1) 相乘，叠加到 u_h 上
    u_h = @(x_var) u_h(x_var) + ...
        u_h_list(i) * (basis_function_local(x_var, x(i), x(i+1), 1)+basis_function_local(x_var, x(i+1), x(i+2), 0));
end

% 定义数值解的导数
u_h_derivative = @(x_var) 0;  % 初始化导数为0

% 遍历每个区间，计算每个区间上的导数
for i = 2:N
    % 每个区间上的导数是常数，等于系数的差分
    current_derivative = (u_h_list(i) - u_h_list(i-1)) / (x(i+1) - x(i));
    % 叠加导数值，使用区间常数值
    u_h_derivative = @(x_var) u_h_derivative(x_var) + ...
        (x_var >= x(i) & x_var <= x(i+1)) * current_derivative;
end

current_derivative = u_h_list(1) / (x(2) - x(1));
u_h_derivative = @(x_var) u_h_derivative(x_var) + ...
    (x_var >= x(1) & x_var <= x(2)) * current_derivative;

current_derivative = ( - u_h_list(N)) / (x(N+2) - x(N+1));
u_h_derivative = @(x_var) u_h_derivative(x_var) + ...
    (x_var >= x(N+1) & x_var <= x(N+2)) * current_derivative;
end

% 算L2误差
function L2_error = compute_L2_error(u_exact, u_h)
% 定义误差函数为 (u_exact(x) - u_h(x))^2
error_sq = @(x_var) (u_exact(x_var) - u_h(x_var)).^2;

% 在 [0, 1] 上对误差函数进行积分，计算平方误差
L2_error = integral(error_sq, 0, 1);

% 取平方根，得到最终的 L2 误差
L2_error = sqrt(L2_error);
end

% 算H1误差
function H1_error = compute_H1_error(u_exact, u_h, u_exact_derivative, u_h_derivative)
% 1. 计算 L2 误差部分
L2_error = compute_L2_error(u_exact, u_h);

% 2. 计算导数的 L2 误差部分，调用 compute_L2_error 来计算
L2_error_derivative = compute_L2_error(u_exact_derivative, u_h_derivative);

% 3. 最终的 H1 误差，结合 L2 误差部分和导数误差部分
H1_error = sqrt(L2_error^2 + L2_error_derivative^2);
end