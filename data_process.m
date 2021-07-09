clear;
clc;
num = 3900;
load('imdata.mat');
num_imdata = imdata(:,1:num);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%����ƽ��������ʾ
average_face=mean(num_imdata,2); %������ƽ��mean(a,2)  ����mean��a��
Average_face=reshape(average_face,128,128)'; %��[128*128,1]�����Ҷ�����ת��[128,128]
figure;
subplot(1,1,1);
imshow(Average_face,[]); %��ʾ�Ҷ�ͼ�� ����������ֵ��Χ����ʾ����ת����
title(strcat('ƽ����'));
clear i j a b addr

%ͼ��Ԥ����:��ȥƽ����ֵ
immin=zeros(128*128,num);
for i=1:num
    immin(:,i) = num_imdata(:,i) - average_face;
end
clear i 
%����Э�������
%W=immin*immin';%dxn*nxd =dxd����N*N��Ϊd*d 
W=immin'*immin; 
[V,D]=eig(W);%������������������ֵ

%������������������
[D_sort,index] = sort(diag(D),'descend');
SumEigenValue=sum(D_sort); %����ֵ�ܴ�С
NowEigenValue=0;
%ѡȡ�ۼƹ��״���%��ǰn��������
for i=1:size(D_sort,1)
    NowEigenValue=NowEigenValue+D_sort(i);
    n=i;
    if(NowEigenValue>SumEigenValue*0.85)%�ۼƹ����ʴﵽ85%���ϼ���
        break;
    end
end
V_sort = V(:,index); %��������������
VT=immin*V_sort; 
for i=1:num
    VT(:,i)=VT(:,i)/norm(VT(:,i));%��һ������
end
newVT=VT(:,1:n);%ȡǰn������ֵ

feature=zeros(n,num);
for i=1:num
    %ӳ��ѵ����ͼ��
    Coefficient=newVT'*immin(:,i);  %k*d x d*1=k*1;
    feature(:,i)=Coefficient;
end
clear i Coefficient

%��ʾǰ12��������
 figure;
for i=1:12
    v=newVT(:,i);
    out=reshape(v,128,128)'; %��(128*128)x1��������ת��128*128�ľ���
    if i<=12
    subplot(2,6,i);
    imshow(out,[]);
    end
    title(strcat('Face',num2str(i)));
end
clear i v out

feature = feature';
save ('feature','feature'); %��������














