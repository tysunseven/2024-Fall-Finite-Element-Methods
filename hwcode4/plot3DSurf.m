function plot3DSurf(func_handle, title_text, z_label_text, varargin)
% plot3DSurf - 绘制给定函数句柄在三角形区域上的 3D 表面图
%
% 输入参数:
%   func_handle: 函数句柄，表示需要绘制的函数，形式为 @(x, y) ...
%   title_text: 图像的标题
%   z_label_text: Z 轴的标签
%   varargin: 可变参数，包含 p1, p2, p3 三个点的坐标，构成一个三角形

% 创建 [0, 1] x [0, 1] 区域的网格
[x, y] = meshgrid(linspace(0, 1, 100), linspace(0, 1, 100));

% 计算函数值
z = func_handle(x, y);

% 判断是否有缺省参数
if nargin < 4
    % 在整个区域上绘图
    in_triangle = true(size(x));
else
    % 限制绘制区域为三角形部分
    p1 = varargin{1};
    p2 = varargin{2};
    p3 = varargin{3};
    in_triangle = inpolygon(x, y, [p1(1), p2(1), p3(1)], [p1(2), p2(2), p3(2)]);
end

z(~in_triangle) = NaN;

% 创建图像
figure;
surf(x, y, z, 'EdgeColor', 'k', 'FaceColor', 'none');
xlabel('x');
ylabel('y');
zlabel(z_label_text);
title(title_text);
end
