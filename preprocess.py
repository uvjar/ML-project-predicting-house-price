import numpy as np
import pandas as pd
df = pd.read_excel("shanghai.xlsx")
df=df.dropna(axis=0, how='any',inplace=False)

df.to_csv("shanghai_drop_nan.csv")
df = pd.read_csv("shanghai_drop_nan.csv")
y=df.iloc[:,3];


# 首付 downPayment
# 原始数据是“首付258万”这样的字符串，截取出数字部分并且转换成int
D=df.iloc[:,4];
downPayment=np.zeros(D.size)
for i in range(D.size):
    temp=str(D[i])[2:-1]
    try:
        downPayment[i]=int(temp)
    except:
        continue


# 建成时间 completionTime
# 0代表缺失值
# 转换为获取数据时间和建成时间的差值
F=np.array(df.iloc[:,6]);
completionTime=np.zeros(F.shape[0])
for i in range(F.shape[0]):
    try:
        completionTime[i]=2018-int(F[i][:4])
    except:
        continue

# 行政区 AdministrativeDistrict
X=np.array(df.iloc[:,9]);
district2id=dict([(district, i) for i, district in enumerate(set(list(X)),start=0)])
AdministrativeDistrict=np.zeros(X.shape[0])
for i in range(X.shape[0]):
    AdministrativeDistrict[i]=district2id[X[i]]


# 挂牌时间 res
transactionProperty=df.iloc[:,7].str.replace(' ', '')
transactionProperty = transactionProperty.str.split('/',expand= True)
res=transactionProperty.iloc[:,0]
res=res.str.replace("挂牌时间：","")
res=pd.to_datetime(res)
time=pd.to_datetime("2018-09-30")
res=np.array(time-res)


# Property right 商品房还是安置房
# ['动迁安置房' '售后公房' '商品房'] -> [0,1,2]
propertyRight = transactionProperty.iloc[:,1]
propertyRight=propertyRight.str.replace("交易权属：","")
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(propertyRight)

print(le.classes_)
propertyRight_num=le.transform(propertyRight)



'''
处理基本属性这一栏
'''
# 基本属性 basicAttribute
E=df.iloc[:,5];
is_villa=[];
layout=[];floor=[];area=[]; 
direction=[];building_type=[];
has_lift=[];lift_to_flats =[]



for r in range(len(E)):
    if(type(E[r])==float):
        is_villa.append('nan');
        layout.append('nan');
        floor.append('nan');
        area.append('nan');
        direction.append('nan');
        building_type.append('nan');
        has_lift.append('nan');
        lift_to_flats.append('nan');  
        continue
    temp= (str(E[r]).split('/'))
    for i in range(len(temp)):
        temp[i]=temp[i].strip()
    if len(temp)==10:
        is_villa.append(1);
        layout.append('nan');
        floor.append('nan');
        area.append('nan');
        direction.append('nan');
        building_type.append('nan');
        has_lift.append('nan');
        lift_to_flats.append('nan');        
    elif len(temp)==13 :
        is_villa.append(0);
        layout.append(temp[0]);
        floor.append(temp[1]);
        area.append(temp[2]);
        direction.append(temp[6]);
        building_type.append(temp[5]);
        has_lift.append(temp[-3]);
        lift_to_flats.append(temp[-4]);        
    elif len(temp)==15:
        is_villa.append(0);
        layout.append(temp[0]);
        floor.append(temp[1]);
        area.append(temp[2]);
        direction.append(temp[6]);
        building_type.append(temp[5]);
        has_lift.append(temp[-5]);
        lift_to_flats.append(temp[-6]);
    
# 是否是别墅 之后把别墅的行给删掉
is_villa=np.array(is_villa)    
print("number of villas "+str(is_villa[is_villa=='1'].size))

# 是否有电梯 电梯/户数 【无，有】->[0,1]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(has_lift)

print(le.classes_)
has_lift_num=le.transform(has_lift)
has_lift_num[has_lift_num==0]=2
has_lift_num[has_lift_num==1]=0
has_lift_num[has_lift_num==3]=1

#building_type 
# ['建筑类型：塔楼' ' '建筑类型：板塔结合' '建筑类型：板楼']
# -> [1, 3, 4]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(building_type)

print(le.classes_)
building_type_num=le.transform(building_type)


# 房屋户型 layout_num
layout_num=np.zeros((len(layout),4)) # 室->厅->厨->卫'
for i in range(len(layout)):
    if len(layout[i])!=13:
        continue
    for roomtype in range(4):
        layout_num[i,roomtype]=int(str(layout[i])[5+2*roomtype:6+2*roomtype])
print(layout[:4]);print(layout_num[:4]);



# 面积 AREA
area_num=np.zeros((len(area),1)) 
for i in range(len(area)):
    if (layout[i])=='nan':
        continue
    try:
        area_num[i,0]=float(str(area[i])[5:-1])
    except:
        continue
        

        
# 楼层
floor_num=np.zeros(len(floor))
total_floor=np.zeros(len(floor))
for i in range(len(floor)):
    temp=str(floor[i])[5:6]
    temp2=str(floor[i])[11:-2]
    if(temp=='' or temp2==''):
        continue
    else:
        total_floor[i]=int(temp2)
        if temp=='低':
            floor_num[i]=1
        elif temp=='中':
            floor_num[i]=2
        elif temp=='高':
            floor_num[i]=3
print(floor[:4]);print(floor_num[:4]);




d = {'total':np.array(y),
    'downPayment': downPayment, 'completionTime': completionTime,
     'AdministrativeDistrict': AdministrativeDistrict,
     'existTime':res,'propertyRight':propertyRight_num,
     'is_villa':is_villa,
     'RoomNumber':layout_num[:,0],
     'LivingRoomNumber':layout_num[:,1],
     'KitchenNumber':layout_num[:,2],
     'BathroomNumber':layout_num[:,3],
     'totalRoomNumber':layout_num.sum(axis=1),
     'floor_num':floor_num,
     'area':area_num.transpose()[0],     
     'has_lift':has_lift_num,
     'building_type':building_type_num    
    }
df = pd.DataFrame(data=d)

df.to_csv("data.csv")




# 再读取 处理缺失值
df = pd.read_csv("data.csv")
temp=np.array(df)
temp1=temp[temp[:,7]==0] # delete villa rows
temp1=temp1[:,[1,2,3,4,5,6,8,9,10,11,12,13,14,15,16]] # delete villa column
print(temp1.shape)
print(temp1[1,:])
temp2=temp1[temp1[:,13]!=2] # delete empty rows in has_lift column
temp3=temp2[temp2[:,14]!=2];temp3=temp3[temp3[:,14]!=0] # delete empty rows in building_type column
print(temp3.shape)
temp4=temp3[temp3[:,2]!=0];
print(temp4.shape)


d = {'total':temp4[:,0],
    'downPayment': temp4[:,1],
     'completionTime': temp4[:,2],
     'AdministrativeDistrict': temp4[:,3],
     'existTime':temp4[:,4],
     'propertyRight':temp4[:,5],
     'RoomNumber':temp4[:,6],
     'LivingRoomNumber':temp4[:,7],
     'KitchenNumber':temp4[:,8],
     'BathroomNumber':temp4[:,9],
     'totalRoomNumber':temp4[:,10],
     'floor_num':temp4[:,11],
     'area':temp4[:,12],  
     'has_lift':temp4[:,13],
     'building_type':temp4[:,14]  
    }
df = pd.DataFrame(data=d)

df.to_csv("data1.csv")




