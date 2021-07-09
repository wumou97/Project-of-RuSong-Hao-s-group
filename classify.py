import scipy.io as sio #用于读取.mat文件
import numpy #用于进行数组操作
from sklearn.ensemble import RandomForestClassifier #导入随机森林的包
from sklearn import svm, tree #导入支持向量机和决策树的包
from sklearn.neighbors import KNeighborsClassifier #导入最近邻分类的包
from sklearn.naive_bayes import GaussianNB


data = sio.loadmat('feature.mat')['feature'] #读取特征数据，每张图44个特征
datalabel = sio.loadmat('faceLabel.mat')['faceLabel'] #读取标签数据
siel = numpy.append(data, datalabel, axis=1) #将特征和标记组合在一起，方便后续随机分成十份
numpy.random.shuffle(siel) #随机按行打乱数据，原始数据将类似的图片排放在以及，需要打乱，为后续交叉验证做准备
la = 44 #标签开始的第一列
data = siel[:, 0:44] #恢复特征数据
reg1 = 0 #用来保存十次最近邻分类的识别率
reg2 = 0 #用来保存十次支持向量机的识别率
reg3 = 0 #用来保存十次随机森林识别率
reg4 = 0 #用来保存十次高斯朴素贝叶斯识别率
reg5 = 0 #用来保存十次决策树的识别率
for j in ['sex', 'age', 'race', 'face']: #循环，进行4种标签的分类
    label = siel[:, la] #提取第la行，即对应上述标签的数据
    la = la+1 #加一，指向下一个标签，下一次循环使用
    for i in range(0, 3900, 390):
        a = range(i, i+390)
        x_test = data[a] #提取十分之一的测试数据
        y_test = label[a] #提取对应的十分之一的标签数据
        x_train = numpy.delete(data, a, 0) #删掉测试数据，即得训练数据集
        y_train = numpy.delete(label, a, 0) #删掉测试的标签数据，即得标签数据集
        classfidier1 = KNeighborsClassifier()#最近邻分类
        classfidier2 = svm.SVC()#支持向量机
        classfidier3 = RandomForestClassifier(n_estimators=50)#随机森林
        classfidier4 = GaussianNB()#高斯朴素贝叶斯
        classfidier5 = tree.DecisionTreeClassifier()#决策树
        y_pred1 = classfidier1.fit(x_train, y_train).predict(x_test) #分类，结果保存到y_pred，下面类似
        y_pred2 = classfidier2.fit(x_train, y_train).predict(x_test)
        y_pred3 = classfidier3.fit(x_train, y_train).predict(x_test)
        y_pred4 = classfidier4.fit(x_train, y_train).predict(x_test)
        y_pred5 = classfidier5.fit(x_train, y_train).predict(x_test)
        reg1 = reg1 + (1-((y_test != y_pred1).sum())*1.0 / (x_test.shape[0])) #计算y_pred的准确率，下同
        reg2 = reg2 + (1-((y_test != y_pred2).sum())*1.0 / (x_test.shape[0]))
        reg3 = reg3 + (1-((y_test != y_pred3).sum())*1.0 / (x_test.shape[0]))
        reg4 = reg4 + (1-((y_test != y_pred4).sum())*1.0 / (x_test.shape[0]))
        reg5 = reg5 + (1-((y_test != y_pred5).sum())*1.0 / (x_test.shape[0]))
    end1 = reg1/10 #把十次的识别率求平均得最终结果
    end2 = reg2/10
    end3 = reg3/10
    end4 = reg4/10
    end5 = reg5/10
    reg1 = 0 #清零，后续进行下一种标签的识别
    reg2 = 0
    reg3 = 0
    reg4 = 0
    reg5 = 0
    print("标签：%s" % j) #打印输出
    print("最近邻分类识别准确率为：%f" % end1)
    print("支持向量机识别准确率为：%f" % end2)
    print("随机森林识别准确率为：%f" % end3)
    print("高斯朴素贝叶斯准确率为：%f" % end4)
    print("决策树准确率为：%f" % end5)











