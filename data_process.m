clear;
clc;
num = 3900;
load('imdata.mat');
num_imdata = imdata(:,1:num);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%计算平均脸并显示
average_face=mean(num_imdata,2); %按行求平均mean(a,2)  按列mean（a）
Average_face=reshape(average_face,128,128)'; %将[128*128,1]的脸灰度数据转成[128,128]
figure;
subplot(1,1,1);
imshow(Average_face,[]); %显示灰度图像 ，根据像素值范围对显示进行转换。
title(strcat('平均脸'));
clear i j a b addr

%图像预处理:减去平均均值
immin=zeros(128*128,num);
for i=1:num
    immin(:,i) = num_imdata(:,i) - average_face;
end
clear i 
%计算协方差矩阵
%W=immin*immin';%dxn*nxd =dxd，由N*N降为d*d 
W=immin'*immin; 
[V,D]=eig(W);%计算特征向量与特征值

%对特征向量进行排序
[D_sort,index] = sort(diag(D),'descend');
SumEigenValue=sum(D_sort); %特征值总大小
NowEigenValue=0;
%选取累计贡献大于%的前n个特征脸
for i=1:size(D_sort,1)
    NowEigenValue=NowEigenValue+D_sort(i);
    n=i;
    if(NowEigenValue>SumEigenValue*0.85)%累计贡献率达到85%以上即可
        break;
    end
end
V_sort = V(:,index); %对特征向量排序
VT=immin*V_sort; 
for i=1:num
    VT(:,i)=VT(:,i)/norm(VT(:,i));%归一化处理
end
newVT=VT(:,1:n);%取前n个特征值

feature=zeros(n,num);
for i=1:num
    %映射训练集图像
    Coefficient=newVT'*immin(:,i);  %k*d x d*1=k*1;
    feature(:,i)=Coefficient;
end
clear i Coefficient

%显示前12个特征脸
 figure;
for i=1:12
    v=newVT(:,i);
    out=reshape(v,128,128)'; %把(128*128)x1的列向量转成128*128的矩阵
    if i<=12
    subplot(2,6,i);
    imshow(out,[]);
    end
    title(strcat('Face',num2str(i)));
end
clear i v out

feature = feature';
save ('feature','feature'); %保存数据














