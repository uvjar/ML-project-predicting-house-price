import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib import cm
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from IPython import display
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error 
from sklearn.model_selection import KFold
kf = KFold(n_splits=10)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

df = pd.read_csv("data1.csv")
df=df.dropna(axis=0,how='any')# df.drop(['completionTime'], axis=1)
y=np.array(df['total'])
# 去掉downpayment 列
df=df.drop(['downPayment'], axis=1)
# df=df.drop(['AdministrativeDistrict'], axis=1)
# 去掉建成时间列
df=df.drop(['completionTime'], axis=1)
# # 去掉详细房间数
# df=df.drop(['RoomNumber'], axis=1)
# df=df.drop(['LivingRoomNumber'], axis=1)
# df=df.drop(['KitchenNumber'], axis=1)
# df=df.drop(['BathroomNumber'], axis=1)
# 挂牌时间string 转成数字
X=np.zeros(df.shape[0])
X1=np.array(df['existTime'])
for i in range(df.shape[0]):
    X[i]=int(X1[i][:-5])
df['existTime']=X
# floor_num=np.array(df['floor_num']).astype(int)
# onehot_encoder = OneHotEncoder(sparse=False)
# floor_num = floor_num.reshape(len(floor_num), 1)
# onehot_encoded = onehot_encoder.fit_transform(floor_num)
# df.insert(df.shape[1], 'floor_low', onehot_encoded[:,1])
# df.insert(df.shape[1], 'floor_mid', onehot_encoded[:,2])
# df.insert(df.shape[1], 'floor_high', onehot_encoded[:,3])
# df=df.drop(['floor_num'], axis=1)
df.head()

dfdata=np.array(df)
# 标准化总价
X=dfdata[:,1]
dfdata[:,1] = (X-X.mean()) / X.std()

# 标准化行政区
X=dfdata[:,2]
dfdata[:,2] = (X-X.mean()) / X.std()

# 标准化挂牌时间
X=dfdata[:,3]
dfdata[:,3] = (X-X.mean()) / X.std()

#去掉面积大于1000的
dfdata=dfdata[dfdata[:,-3]<1000]
print(dfdata.shape)
# 标准化面积
X=dfdata[:,-3]
dfdata[:,-3] = (X-X.mean()) / X.std()

print(dfdata)


#######选择hyperparameter########选择hyperparameter################选择hyperparameter############
####################选择hyperparameter#############选择hyperparameter###########################
######选择hyperparameter##########选择hyperparameter###############选择hyperparameter############


# LASSO: Cross validation q
X=dfdata[:,2:];y=dfdata[:,1]
def compare_q(X,y):
    C=100
    mean_error=[]; std_error=[];train_error=[];train_std=[];
    q_range = [1,2,3,4,5]
    for q in q_range: 
        # temp=X[:,-3:];
        # Xpoly = PolynomialFeatures(q).fit_transform(X[:,:-3]);
        # Xpoly=np.concatenate((Xpoly,temp),axis=1)
        Xpoly = PolynomialFeatures(q).fit_transform(X);
        model = linear_model.Lasso(alpha=1/(2*C)).fit(Xpoly,y)
        #model = LogisticRegression().fit(Xpoly,y)
        temp=[]; plotted = False;train_error_temp=[];
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            temp.append(mean_squared_error(y[test],ypred)) 
            train_error_temp.append(mean_squared_error(model.predict(Xpoly[train]),y[train]))
        train_error.append(np.array(train_error_temp).mean())
        train_std.append(np.array(train_error_temp).std()) 
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std()) 
    plt.errorbar(q_range,mean_error,yerr=std_error,linewidth=1) 
    plt.errorbar(q_range,train_error,yerr=train_std,linewidth=1)
    #plt.plot(q_range,train_error,linewidth=3)
    plt.xlabel('q')
    plt.ylabel('Mean square error')
    plt.legend(["Test Data","Training Data"])
    plt.show()
    
    
def compare_C(X,y,q):
    mean_error=[]; std_error=[];train_error=[];train_std=[];
    # temp=X[:,-3:];
    # Xpoly = PolynomialFeatures(q).fit_transform(X[:,:-3]);
    # Xpoly=np.concatenate((Xpoly,temp),axis=1)
    Xpoly = PolynomialFeatures(q).fit_transform(X);
    C_range = [1,10,20,30,40,50,60,70,80,90,100]
    for C in C_range: 
        model = linear_model.Lasso(alpha=1/(2*C)).fit(Xpoly,y)
        temp=[]; plotted = False;train_error_temp=[];
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            temp.append(mean_squared_error(y[test],ypred)) 
            train_error_temp.append(mean_squared_error(model.predict(Xpoly[train]),y[train]))
        train_error.append(np.array(train_error_temp).mean())
        train_std.append(np.array(train_error_temp).std())
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std()) 
    plt.errorbar(C_range,mean_error,yerr=std_error,linewidth=1) 
    plt.errorbar(C_range,train_error,yerr=train_std,linewidth=1)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.legend(["Test Data","Training Data"])
    plt.show()
# compare_q(X,y)
# compare_C(X,y,3)




###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
# KNN
# Cross validation k
from sklearn.neighbors import KNeighborsRegressor
X=dfdata[:,2:];y=dfdata[:,1]
def compare_k(X,y):
    mean_error=[]; std_error=[];train_error=[];train_std=[];
    k_range = [2,5,10,15,20]
    for k in k_range: 
        model = KNeighborsRegressor(n_neighbors=k,weights='uniform').fit(X, y)
        temp=[]; plotted = False;train_error_temp=[];
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            temp.append(mean_squared_error(y[test],ypred)) 
            train_error_temp.append(mean_squared_error(model.predict(Xpoly[train]),y[train]))
        train_error.append(np.array(train_error_temp).mean())
        train_std.append(np.array(train_error_temp).std())
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std()) 
    plt.errorbar(k_range,mean_error,yerr=std_error,linewidth=1) 
    plt.errorbar(k_range,train_error,yerr=train_std,linewidth=1)
    plt.xlabel('k')
    plt.ylabel('Mean square error')
    plt.legend(["Test Data","Training Data"])
    plt.show()
# compare_k(X,y)





###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################

# RIDGE: Cross validation q
# from sklearn.linear_model import Ridge 
X=dfdata[:,2:];y=dfdata[:,1]

def compare_C_ridge(X,y,q):
    mean_error=[]; std_error=[];train_error=[];train_std=[];
    # temp=X[:,-3:];
    # Xpoly = PolynomialFeatures(q).fit_transform(X[:,:-3]);
    # Xpoly=np.concatenate((Xpoly,temp),axis=1)
    Xpoly = PolynomialFeatures(q).fit_transform(X);
    C_range = [0.1,1,10,100]
    for C in C_range:
        model = linear_model.Ridge(alpha=1/(2*C)).fit(Xpoly,y)
        temp=[]; train_error_temp=[];
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            temp.append(mean_squared_error(y[test],ypred)) 
            train_error_temp.append(mean_squared_error(model.predict(Xpoly[train]),y[train]))
        train_error.append(np.array(train_error_temp).mean())
        train_std.append(np.array(train_error_temp).std())
        mean_error.append(np.array(temp).mean())
        print(np.array(temp).std())
        std_error.append(np.array(temp).std()) 
    plt.errorbar(C_range,mean_error,yerr=std_error,linewidth=1) 
    plt.errorbar(C_range,train_error,yerr=train_std,linewidth=1)
    plt.xlabel('C')
    plt.ylabel('Mean square error')
    plt.legend(["Test Data","Training Data"])
    plt.show()

def compare_q_ridge(X,y):
    C=1
    mean_error=[]; std_error=[];train_error=[];train_std=[];
    q_range = [1,2,3,4,5]
    for q in q_range: 
        # temp=X[:,-3:];
        # Xpoly = PolynomialFeatures(q).fit_transform(X[:,:-3]);
        # Xpoly=np.concatenate((Xpoly,temp),axis=1)
        Xpoly = PolynomialFeatures(q).fit_transform(X);
        model = linear_model.Ridge(alpha=1/(2*C)).fit(Xpoly,y)
        #model = LogisticRegression().fit(Xpoly,y)
        temp=[]; plotted = False;train_error_temp=[];
        for train, test in kf.split(Xpoly):
            model.fit(Xpoly[train], y[train])
            ypred = model.predict(Xpoly[test])
            temp.append(mean_squared_error(y[test],ypred)) 
            train_error_temp.append(mean_squared_error(model.predict(Xpoly[train]),y[train]))
        train_error.append(np.array(train_error_temp).mean())
        train_std.append(np.array(train_error_temp).std()) 
        mean_error.append(np.array(temp).mean())
        std_error.append(np.array(temp).std()) 
    plt.errorbar(q_range,mean_error,yerr=std_error,linewidth=1) 
    plt.errorbar(q_range,train_error,yerr=train_std,linewidth=1)
    #plt.plot(q_range,train_error,linewidth=3)
    plt.xlabel('q')
    plt.ylabel('Mean square error')
    plt.legend(["Test Data","Training Data"])
    plt.show()    
    
    
    
# compare_q_ridge(X,y)



###############################################################################################
#决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果
###############################################################################################
##决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果决策树结果
X=dfdata[:,2:];y=dfdata[:,1]
features = X;predict=y
features_train, features_test, predict_train, predict_test = train_test_split(features, predict, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor

rf_regressor=RandomForestRegressor(n_estimators=100,max_depth=10,min_samples_split=5,min_samples_leaf=5,max_features=None,oob_score=True,random_state=42)
rf_regressor.fit(features_train,predict_train)
predict_test_y=rf_regressor.predict(features_test)
import sklearn.metrics as metrics
print('MSE for RFR：{:,.3f}'.format(
    round(metrics.mean_squared_error(predict_test_y,predict_test),2)))
r2 = performance_metric(predict_test, predict_test_y)
print("R^2 score for RFR：{:,.2f}".format(r2))
#show the importance
importances = rf_regressor.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels =featuresDF
for name, importance in zip(feat_labels, rf_regressor.feature_importances_):
        print(name, "=", importance)


###############################################################################################
# LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO
###############################################################################################
# LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO LASSO
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
C=10; q=3; C2=0.1; q2=3
Xpoly_train = PolynomialFeatures(q).fit_transform(X_train);
Xpoly_test = PolynomialFeatures(q).fit_transform(X_test);
model = linear_model.Lasso(alpha=1/(2*C)).fit(Xpoly_train,y_train)
ypred = model.predict(Xpoly_test)
print("MSE of LASSO (C=10, q=3)")
print(mean_squared_error(y_test,ypred))

###############################################################################################
# RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE
###############################################################################################
# RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE RIDGE
model = linear_model.Ridge(alpha=1/(2*C)).fit(Xpoly_train,y_train)
ypred = model.predict(Xpoly_test)
print("MSE of RIDGE (C=0.1, q=3)")
print(mean_squared_error(y_test,ypred))

###############################################################################################
# KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN
###############################################################################################
# KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN KNN
knn_model = KNeighborsRegressor(n_neighbors=k,weights='uniform').fit(X_train,y_train)
ypred = knn_model.predict(X_test)
print("MSE of KNN (k=20)")
print(mean_squared_error(y_test,ypred))
