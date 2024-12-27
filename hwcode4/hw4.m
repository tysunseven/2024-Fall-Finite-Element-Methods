clear;clc;
tic;
% N=[8,16,32,64];%每个方向细分次数
N=[2,4,8,16];%每个方向细分次数
num=zeros(1,4);%单元数
nod=zeros(1,4);%总结点数
bnod=zeros(1,4);%边界结点数
inod=zeros(1,4);%内部结点数
Error1=zeros(1,4);%L2误差
Error2=zeros(1,4);%H1误差
order1=zeros(1,3);%L2误差阶
order2=zeros(1,3);%H1误差阶
for k = 1:4
    Lx = 1;Ly = 1; %定义单元边界
    nx = N(k);%定义分割的x方向单元数目（按矩形计算）
    ny = N(k);%定义分割的y方向单元数目（按矩形计算）
    hx = Lx/nx;%x方向上的单元长度（对应着三角形两条直角边之一）
    hy = Ly/ny;%y方向上的单元长度（对应着三角形两条直角边之一）
    num(k) = nx*ny*2;%小单元的数目,每个矩形分成两个三角形
    u_b = sparse(2*(nx+ny),1); %定义第一类边界条件，一圈过来都是0
    nodx = nx + 1;nody = ny + 1;
    nod(k) = nodx*nody;
    bnod(k)=2*(nx+ny);
    inod(k)=nod(k)-bnod(k);
    nel = 3;%单元自由度
    coordx = linspace(0,Lx,nodx)';
    coordy = linspace(0,Ly,nody)';
    [X, Y] = meshgrid(coordx,coordy);%张成网格
    X = X';Y = Y';coord = [X(:) Y(:)];
    connect = connect_mat(nodx,nody,nel);
    bdof = unique([1:nodx nodx*ny+1:nodx*ny+nodx nodx+1:nodx:nodx*(ny-1)+1 2*nodx:nodx:ny*nodx]); % 强制性边界点的编号
    bval = u_b; %假设边界值都为u_b
    B = sparse(nod(k),nod(k)); % 刚度矩阵[K]，初始化为0，使用稀疏矩阵存储
    F = sparse(nod(k),1);      % 载荷向量{f},初始化为0
    %%  计算系数矩阵K和右端项f
    for e = 1:num(k) %同一维的情况，依然按单元来扫描
        ke = elemstiff2d(e,nel,hx,hy,coord,connect);%计算单元刚度矩阵
        if k==1
            % 展示ke的同时展示e
            disp(e);
            disp(full(ke));
        end
        Fe = elemforce2d(e,coord,connect);%计算单元载荷向量
        sctr = connect(e,:);
        B(sctr,sctr) = B(sctr,sctr) + ke;
        F(sctr) = F(sctr) + Fe;
    end
    
    for i = 1:length(bdof)
        n = bdof(i);
        for j = 1:nod(k)
            if (isempty(find(bdof == j, 1))) % 第j个点若不是固定点
                F(j) = F(j) - B(j,n)*bval(i);
            end
        end
        B(n,:) = 0.0;
        B(:,n) = 0.0;
        B(n,n) = 1.0;
        F(n) = bval(i);
    end
    if k==1
        disp(full(B));
    end
    u_coeff = B\F;%求出系数，事实上也是函数在对应点上的值
    u_cal_re = reshape(u_coeff,nodx,nody);
    u_cal_re = full(u_cal_re);
    %% 求精确解
    L =Lx;
    nsamp = N(k)+1;
    xsamp = linspace(0,L,nsamp);
    [X,Y] = meshgrid(xsamp,xsamp);
    uexact = sol(X(:),Y(:));
    uexact_re = reshape(uexact,nsamp,nsamp);
    %mesh(xsamp,xsamp,uexact_re)
    %% 绘图，可视化
    
    subplot(2,2,k);
    h = mesh(coordx,coordy,u_cal_re);
    title(' FE Solutions');%标题
    
    %% 计算误差
    error1=zeros(1,num(k));error2=zeros(1,num(k));
    for e=1:num(k)
        error1(e)=ElemerrorL2(e,coord,u_cal_re,hx,hy,connect);
        error2(e)=ElemerrorH1(e,coord,u_cal_re,hx,hy,connect);
    end
    Error1(k)=sqrt(sum(error1));
    Error2(k)=sqrt(sum(error2));
end
for i=1:1:3
    order1(i)=log(Error1(i+1)/Error1(i))/log(N(i)/N(i+1));
    order2(i)=log(Error2(i+1)/Error2(i))/log(N(i)/N(i+1));
end
%% 输出目标表格
chart=[nod;inod;bnod;0,order1;0,order2]';
t=toc;
disp(t);

function u = sol(x,y)
u = (x-1).*(y-1).*sin(x).*sin(y);
return
end

function u=sol_diff1(x,y)
u=(y-1).*sin(y).*(sin(x)+(x-1).*cos(x));
end
function u=sol_diff2(x,y)
u=(x-1).*sin(x).*(sin(y)+(y-1).*cos(y));
end
function [ke] = elemstiff2d(e,nel,hx,hy,coord,connect)
T = hx*hy/2;
ke = sparse(nel,nel);
nodes = connect(e,:);%相关形函数（节点）编号
xe = coord(nodes,:); %相关节点的坐标
xi1 = xe(2,1) - xe(3,1);xi2 = xe(3,1) - xe(1,1);%xi3 = xe(1,1) - xe(2,1);
eta1 = xe(2,2) - xe(3,2);eta2 = xe(3,2) - xe(1,2);%eta3 = xe(1,2) - xe(2,2);
a11 = -((eta1*xi2 - eta2*xi1)*(eta1^2 + xi1^2))/(8*T^2);
a12 = ((eta1*eta2 + xi1*xi2)*(-eta1*xi2 + eta2*xi1))/(8*T^2);
a13 = -((eta1*xi2 - eta2*xi1)*(- eta1^2 + xi1*eta1 - eta2^2 + xi2*eta2))/(8*T^2);
a22 = -((xi1^2 + xi2^2)*(eta1*xi2 - eta2*xi1))/(8*T^2);
a23 = -((eta1*xi2 - eta2*xi1)*(- xi1^2 + eta1*xi1 - xi2^2 + eta2*xi2))/(8*T^2);
a33 = -((eta1*xi2 - eta2*xi1)*(eta1^2 - 2*eta1*xi1 + eta2^2 - 2*eta2*xi2 + xi1^2 + xi2^2))/(8*T^2);
ke = ke + [a11 a12 a13;a12 a22 a23;a13 a23 a33];
return
end

function [fe] = elemforce2d(e,coord,connect)
nodes = connect(e,:);%自由度编号
xe = coord(nodes,:); % 单元自由节点坐标
xi1 = xe(2,1) - xe(3,1);xi2 = xe(3,1) - xe(1,1);%xi3 = xe(1,1) - xe(2,1);
eta1 = xe(2,2) - xe(3,2);eta2 = xe(3,2) - xe(1,2);%eta3 = xe(1,2) - xe(2,2);
detJ = eta2*xi1 - eta1*xi2;
g1 = @(lam1,lam2) fun(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1)).*lam1*detJ;
g2 = @(lam1,lam2) fun(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1)).*lam2*detJ;
g3 = @(lam1,lam2) fun(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1)).*(1-lam1-lam2)*detJ;
lammax = @(lam1) 1 - lam1;
gx(1) = integral2(g1,0,1,0,lammax);
gx(2) = integral2(g2,0,1,0,lammax);
gx(3) = integral2(g3,0,1,0,lammax);
fe = [gx(1);gx(2);gx(3)];
end

function bx = fun(x,y)
bx = (x-1).*sin(x).*((y-1).*sin(y)-2*cos(y))+(y-1).*sin(y).*((x-1).*sin(x)-2*cos(x));
end

function connect_mat = connect_mat(nodx,nody,nel)
%输入横纵坐标的节点数目，和单元自由度
%输出连接矩阵，每个单元涉及的节点的编号
xn = 1:(nodx*nody);%拉成一条编号
A = reshape(xn,nodx,nody);%同形状编号
for i = 1:(nodx-1)*(nody-1)
    x = rem(i,nodx-1);%xg表示单元为左边界数起第几个
    if x == 0
        x = nodx-1;
    end
    y = ceil(i/(nodx-1));
    a = A(x:x+1,y:y+1);
    a_vec = a(:);
    connect_mat(2*i-1:2*i,1:nel) = [a_vec([1 4 3])';a_vec([4 1 2])'];
end
end

function error=ElemerrorL2(e,coord,u,hx,hy,connect)
%计算单元的L2误差
nodes = connect(e,:);%自由度编号
xe = coord(nodes,:); % 单元自由节点坐标
xi1 = xe(2,1) - xe(3,1);xi2 = xe(3,1) - xe(1,1);
eta1 = xe(2,2) - xe(3,2);eta2 = xe(3,2) - xe(1,2);
detJ = eta2*xi1 - eta1*xi2;
g1 = @(lam1,lam2) (sol(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1))-u(xe(1,1)/hx+1,xe(1,2)/hy+1).*lam1-u(xe(2,1)/hx+1,xe(2,2)/hy+1).*lam2-u(xe(3,1)/hx+1,xe(3,2)/hy+1).*(1-lam1-lam2)).^2.*detJ;
lammax = @(lam1) 1 - lam1;
error = integral2(g1,0,1,0,lammax);
end
function error=ElemerrorH1(e,coord,u,hx,hy,connect)
%计算单元的H1误差
nodes = connect(e,:);%自由度编号
xe = coord(nodes,:); % 单元自由节点坐标
xi1 = xe(2,1) - xe(3,1);xi2 = xe(3,1) - xe(1,1);
eta1 = xe(2,2) - xe(3,2);eta2 = xe(3,2) - xe(1,2);
detJ = eta2*xi1 - eta1*xi2;
g1 = @(lam1,lam2) (sol(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1))-u(xe(1,1)/hx+1,xe(1,2)/hy+1).*lam1-u(xe(2,1)/hx+1,xe(2,2)/hy+1).*lam2-u(xe(3,1)/hx+1,xe(3,2)/hy+1).*(1-lam1-lam2)).^2.*detJ;
g2 = @(lam1,lam2) ((sol_diff1(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1))*(-xi2)+sol_diff2(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1))*(-eta2)-u(xe(1,1)/hx+1,xe(1,2)/hy+1)+u(xe(3,1)/hx+1,xe(3,2)/hy+1)).^2+(sol_diff1(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1))*(xi1)+sol_diff2(xe(3,1) + lam1*(-xi2)+lam2*(xi1),xe(3,2) + lam1*(-eta2)+lam2*(eta1))*(eta1)-u(xe(2,1)/hx+1,xe(2,2)/hy+1)+u(xe(3,1)/hx+1,xe(3,2)/hy+1)).^2);
lammax = @(lam1) 1 - lam1;
error = integral2(g1,0,1,0,lammax)+integral2(g2,0,1,0,lammax);
end