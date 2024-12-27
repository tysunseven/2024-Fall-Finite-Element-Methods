function hwcode4()
% 主程序
N_values = [2];
u_exact = @(x, y) (x - 1) .* (y - 1) .* sin(x) .* sin(y);
f = @(x, y) -(-2 * sin(x) .* sin(y) + (x - 1) .* (y - 1) .* cos(x) .* cos(y));
plot3DSurf(u_exact, '3D Plot of u\_exact on [0,1] x [0,1] (No Color)', 'u\_exact(x, y)');

% % 测试函数 compute_local_shape_functions
% x_i = 0; y_i = 0;
% x_j = 1; y_j = 0;
% x_m = 0; y_m = 1;

% [N_local,B] = compute_local_shape_functions(x_i,y_i,x_j,y_j,x_m,y_m);

% plot3DSurf(N_local{1}, '3D Plot of N_i(x,y) on [0,1] x [0,1] (No Color)', 'N_i(x, y)',[x_i,y_i],[x_j,y_j],[x_m,y_m]);
% plot3DSurf(N_local{2}, '3D Plot of N_j(x,y) on [0,1] x [0,1] (No Color)', 'N_j(x, y)',[x_i,y_i],[x_j,y_j],[x_m,y_m]);
% plot3DSurf(N_local{3}, '3D Plot of N_m(x,y) on [0,1] x [0,1] (No Color)', 'N_m(x, y)',[x_i,y_i],[x_j,y_j],[x_m,y_m]);

% disp(B);
L2_errors = zeros(length(N_values), 1);
H1_errors = zeros(length(N_values), 1);
% 遍历N_values，生成网格
for i = 1:length(N_values)
    N = N_values(i);
    [x,y,tri,freenodes]=generateGrid(N);
    [A,F]=assemble_matrices(x,y,tri,f);
    u_h=zeros(length(x),1);
    u_h(freenodes)=A(freenodes,freenodes)\F(freenodes);
    % disp(full(u_h));
    %根据x,y,u_h绘制三维图
    figure;
    trisurf(tri,x,y,u_h,'EdgeColor', 'k', 'FaceColor', 'none');
    title(['3D Plot of u_h on [0,1] x [0,1] with N = ', num2str(N)]);
    xlabel('x');
    ylabel('y');
    zlabel('u_h(x, y)');
    
    L2_errors(i) = compute_L2_error(u_exact, u_h, x, y, tri);
    if i > 1
        L2_order = log(L2_errors(i-1) / L2_errors(i)) / log(N_values(i) / N_values(i-1));
        % H1_order = log(H1_errors(i-1) / H1_errors(i)) / log(N_values(i) / N_values(i-1));
        % fprintf('N = %d, L2_error = %.5e, L2_order = %.2f, H1_error = %.5e, H1_order = %.2f\n', ...
        %     N, L2_errors(i), L2_order, H1_errors(i), H1_order);
        fprintf('N = %d, L2_error = %.5e, L2_order = %.2f\n', N, L2_errors(i), L2_order);
    else
        % fprintf('N = %d, L2_error = %.5e, H1_error = %.5e\n', N, L2_errors(i), H1_errors(i));
        fprintf('N = %d, L2_error = %.5e\n', N, L2_errors(i));
    end
end

end

function [x,y,tri,freenodes]=generateGrid(N)
% 定义均匀网格点
[x, y] = meshgrid(linspace(0, 1, N+1), linspace(0, 1, N+1));
x = x(:);
y = y(:);
allnodes = [1:(N+1)*N+1];
% 遍历所有节点，如果该节点位于边界，就将其编号添加到 fixednodes 中
fixednodes = [];
for i = 1:(N+1)*N+1
    if x(i) == 0 || x(i) == 1 || y(i) == 0 || y(i) == 1
        fixednodes = [fixednodes, i];
    end
end
% 从 allnodes 中删除 fixednodes 中的节点编号
freenodes = setdiff(allnodes, fixednodes);
% 对点进行三角剖分
tri = delaunay(x, y);

% 绘制三角剖分效果图
figure;
triplot(tri, x, y, 'k');  % 绘制黑色的三角网格线
hold on;

% 在图中为每个结点标注编号
for i = 1:length(x)
    text(x(i), y(i), num2str(i), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right', 'FontSize', 8, 'Color', 'blue');
end

% 在图中为每个三角形标注编号
centroids = [(x(tri(:,1)) + x(tri(:,2)) + x(tri(:,3)))/3, ...
    (y(tri(:,1)) + y(tri(:,2)) + y(tri(:,3)))/3];  % 计算每个三角形的重心
for i = 1:size(tri, 1)
    text(centroids(i,1), centroids(i,2), num2str(i), 'Color', 'red', 'FontSize', 8, 'FontWeight', 'bold');
end

axis off;
hold off;
end

function [N_local,B] = compute_local_shape_functions(x_i,y_i,x_j,y_j,x_m,y_m)
syms x;
syms y;

% 计算三角形面积的两倍
area_2K = abs(x_i * (y_j - y_m) + x_j * (y_m - y_i) + x_m * (y_i - y_j));

% 计算形函数 N_i 的系数 a_i, b_i, c_i
a_i = y_j - y_m;
b_i = x_m - x_j;
c_i = x_j * y_m - x_m * y_j;
N_i = (a_i * x + b_i * y + c_i) / area_2K;

% 计算形函数 N_j 的系数 a_j, b_j, c_j
a_j = y_m - y_i;
b_j = x_i - x_m;
c_j = x_m * y_i - x_i * y_m;
N_j = (a_j * x + b_j * y + c_j) / area_2K;

% 计算形函数 N_m 的系数 a_m, b_m, c_m
a_m = y_i - y_j;
b_m = x_j - x_i;
c_m = x_i * y_j - x_j * y_i;
N_m = (a_m * x + b_m * y + c_m) / area_2K;


N_local = {matlabFunction(N_i, 'Vars', [x, y]), ...
    matlabFunction(N_j, 'Vars', [x, y]), ...
    matlabFunction(N_m, 'Vars', [x, y])};


% plot3DSurf(N_local{1}, '3D Plot of N_i(x,y) on [0,1] x [0,1] (No Color)', 'N_i(x, y)',[x_i,y_i],[x_j,y_j],[x_m,y_m]);
% plot3DSurf(N_local{2}, '3D Plot of N_j(x,y) on [0,1] x [0,1] (No Color)', 'N_j(x, y)',[x_i,y_i],[x_j,y_j],[x_m,y_m]);
% plot3DSurf(N_local{3}, '3D Plot of N_m(x,y) on [0,1] x [0,1] (No Color)', 'N_m(x, y)',[x_i,y_i],[x_j,y_j],[x_m,y_m]);

B = [a_i, a_j, a_m; b_i, b_j, b_m] / area_2K;
end

function [A, F] = assemble_matrices(x, y, tri, f)
% 获取节点数和单元数
numNodes = length(x);
numElements = size(tri, 1);

% 初始化全局刚度矩阵和载荷向量
A = sparse(numNodes, numNodes);  % 稀疏矩阵用于高效存储

F = zeros(numNodes, 1);          % 载荷向量

% 遍历每个单元，计算局部矩阵并进行拼装
for k = 1:numElements
    % 获取当前三角形单元的顶点索引
    nodeIndices = tri(k, :);
    x_i = x(nodeIndices(1));
    y_i = y(nodeIndices(1));
    x_j = x(nodeIndices(2));
    y_j = y(nodeIndices(2));
    x_m = x(nodeIndices(3));
    y_m = y(nodeIndices(3));
    
    % 计算局部形函数和刚度矩阵
    [N_local, B] = compute_local_shape_functions(x_i, y_i, x_j, y_j, x_m, y_m);
    
    % 计算局部刚度矩阵 A_local = B' * B * |K| (|K| 是单元面积)
    area_K = abs(x_i * (y_j - y_m) + x_j * (y_m - y_i) + x_m * (y_i - y_j))/2;
    A_local = (B' * B) * area_K;
    for i = 1:3
        for j = 1:3
            A(nodeIndices(i), nodeIndices(j)) = A(nodeIndices(i), nodeIndices(j)) + A_local(i, j);
        end
    end
    % 计算局部载荷向量 F_local
    % 使用三角形中心点近似法计算积分（更高精度可以使用数值积分）
    centroid_x = (x_i + x_j + x_m) / 3;
    centroid_y = (y_i + y_j + y_m) / 3;
    F_local = area_K * f(centroid_x, centroid_y) * [N_local{1}(centroid_x, centroid_y); N_local{2}(centroid_x, centroid_y); N_local{3}(centroid_x, centroid_y)];
    
    % 将局部载荷向量 F_local 拼装到全局载荷向量 F
    F(nodeIndices(:)) = F(nodeIndices(:)) + F_local;
end
end

function L2_error = compute_L2_error(u_exact, u_h, x, y, tri)
L2_error = 0;
numElements = size(tri, 1);

% 遍历每个三角形单元
for k = 1:numElements
    % 获取当前三角形单元的顶点索引
    nodeIndices = tri(k, :);
    x_i = x(nodeIndices(1));
    y_i = y(nodeIndices(1));
    x_j = x(nodeIndices(2));
    y_j = y(nodeIndices(2));
    x_m = x(nodeIndices(3));
    y_m = y(nodeIndices(3));
    
    % 计算三角形单元面积
    area_K = abs(x_i * (y_j - y_m) + x_j * (y_m - y_i) + x_m * (y_i - y_j)) / 2;
    
    % 二点高斯积分的积分点和权重
    xi1 = 2/3; eta1 = 1/6;  % 第一个积分点
    xi2 = 1/6; eta2 = 2/3;  % 第二个积分点
    w1 = 1/2; w2 = 1/2;     % 每个点的权重
    
    % 将参考积分点映射到当前三角形单元的实际坐标
    gx1 = x_i * (1 - xi1 - eta1) + x_j * xi1 + x_m * eta1;
    gy1 = y_i * (1 - xi1 - eta1) + y_j * xi1 + y_m * eta1;
    gx2 = x_i * (1 - xi2 - eta2) + x_j * xi2 + x_m * eta2;
    gy2 = y_i * (1 - xi2 - eta2) + y_j * xi2 + y_m * eta2;
    
    % 计算数值解 u_h 在积分点处的值（使用插值）
    u_h1 = u_h(nodeIndices(1)) * (1 - xi1 - eta1) + u_h(nodeIndices(2)) * xi1 + u_h(nodeIndices(3)) * eta1;
    u_h2 = u_h(nodeIndices(1)) * (1 - xi2 - eta2) + u_h(nodeIndices(2)) * xi2 + u_h(nodeIndices(3)) * eta2;
    
    % 计算精确解 u_exact 在积分点处的值
    u_exact1 = u_exact(gx1, gy1);
    u_exact2 = u_exact(gx2, gy2);
    
    % 计算当前单元上的 L2 误差，并累加
    L2_error = L2_error + area_K * (w1 * (u_h1 - u_exact1)^2 + w2 * (u_h2 - u_exact2)^2);
end

% 返回 L2 误差的平方根
L2_error = sqrt(L2_error);
end