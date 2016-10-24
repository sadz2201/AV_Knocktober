import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pylab as plt
import operator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

data_path = "input/Train/"
first_camp = pd.read_csv( data_path + "First_Health_Camp_Attended.csv" )
second_camp = pd.read_csv( data_path + "Second_Health_Camp_Attended.csv" )
third_camp = pd.read_csv( data_path + "Third_Health_Camp_Attended.csv" )

first_camp=first_camp.ix[:,:4]
col_names = [['Patient_ID','Health_Camp_ID','Outcome']]
second_camp = second_camp[['Patient_ID','Health_Camp_ID','Health Score']]
second_camp.columns = col_names

first_camp = first_camp[['Patient_ID','Health_Camp_ID','Health_Score']]
first_camp.columns = col_names

third_camp = third_camp[['Patient_ID','Health_Camp_ID','Number_of_stall_visited']]
third_camp = third_camp[third_camp['Number_of_stall_visited']>0]
third_camp.columns = col_names

all_camps = pd.concat([first_camp, second_camp, third_camp])
all_camps['Outcome'] = 1

train = pd.read_csv(data_path + "Train.csv")
train = train.merge(all_camps, on=['Patient_ID','Health_Camp_ID'], how='left')
train['Outcome'] = train['Outcome'].fillna(0).astype('int')

test = pd.read_csv('input/Test.csv')
test['Outcome']=0
tr_rows=train.shape[0]

full=pd.concat((train,test)).reset_index(drop=True)
full.Registration_Date=pd.to_datetime(full.Registration_Date)
full['id']=range(full.shape[0]) #useful to re-order the dataframe in the end after doing all merges & manipulations

patient_profile=pd.read_csv(data_path + "Patient_Profile.csv")
patient_profile.Income[patient_profile.Income=="None"] = -1
patient_profile.Education_Score[patient_profile.Education_Score=="None"]=-1
patient_profile.Age[patient_profile.Age=="None"]= -1
patient_profile.First_Interaction=pd.to_datetime(patient_profile.First_Interaction)
patient_profile.City_Type=patient_profile.City_Type.fillna("Unknown")
patient_profile.Employer_Category=patient_profile.Employer_Category.fillna("Unknown")
patient_profile.Income=patient_profile.Income.astype(int)
patient_profile.Education_Score=patient_profile.Education_Score.astype(float)
patient_profile.Age=patient_profile.Age.astype(int)
for c in ['City_Type','Employer_Category']:
    patient_profile[c] = pd.factorize(patient_profile[c], sort=True)[0]
    
patient_profile['FI_month']=patient_profile.First_Interaction.dt.month
patient_profile['FI_year']=patient_profile.First_Interaction.dt.year
patient_profile['FI_day']=patient_profile.First_Interaction.dt.day
patient_profile['FI_wkday']=patient_profile.First_Interaction.dt.weekday

hc_detail=pd.read_csv(data_path + "Health_Camp_Detail.csv")
hc_detail.Camp_Start_Date=pd.to_datetime(hc_detail.Camp_Start_Date)
hc_detail.Camp_End_Date=pd.to_datetime(hc_detail.Camp_End_Date)
hc_detail['duration']=(hc_detail.Camp_End_Date-hc_detail.Camp_Start_Date).dt.days
hc_detail.Category1[hc_detail.Category1=="First"]=1
hc_detail.Category1[hc_detail.Category1=="Second"]=2
hc_detail.Category1[hc_detail.Category1=="Third"]=3
hc_detail.Category1=hc_detail.Category1.astype(int)
hc_detail.Category2=pd.factorize(hc_detail.Category2, sort=True)[0]

full=full.merge(patient_profile)
full=full.merge(hc_detail)

full.Registration_Date[full.Registration_Date.isnull()==True]=full.Camp_Start_Date[full.Registration_Date.isnull()==True]
full['Reg_month']=full.Registration_Date.dt.month
full['Reg_year']=full.Registration_Date.dt.year
full['Reg_day']=full.Registration_Date.dt.day
full['Reg_wkday']=full.Registration_Date.dt.weekday
#time delta (days) between registration & camp starting
full['delta_reg_start']=(full.Camp_Start_Date-full.Registration_Date).dt.days
#time delta between registration & first interaction
full['delta_reg_FI']=(full.Registration_Date-full.First_Interaction).dt.days

#sort by patient id & camp start date
full=full.sort_values(by=['Patient_ID','Camp_Start_Date']).reset_index(drop=True)
dup=full.duplicated(subset='Patient_ID')
#difference between any row & it's previous row
diff_df=full.diff(1)

#ending date of the camp previously attended by the patient
prev_camp_end=full.iloc[0:(full.shape[0]-1)].Camp_End_Date
prev_camp_end=pd.concat((pd.Series(full.iloc[0].Camp_Start_Date),prev_camp_end)).reset_index(drop=True)
full['prev_camp_end']=prev_camp_end

#delta between current camp start date & prev camp end date
full['delta_prev_camp']=(full.Camp_Start_Date-full.prev_camp_end).dt.days
#delta between current camp registration & prev camp registration date
full['delta_prev_reg']=diff_df.Registration_Date.dt.days
#overwrite to 0 for non-duplicate patients (i.e no previous camp attended)
full.delta_prev_camp[dup==False] = 0
full.delta_prev_reg[dup==False] = 0

full=full.drop(['Var3','Var4'],axis=1) #these 2 variables are useless.
#re-sort by id so that train rows are first & test rows come later
full=full.sort_values(by="id").reset_index(drop=True)

train=full[:tr_rows]
test=full[tr_rows:]

#split train into tr & val. All camps starting before 31st Aug 2005 go into tr, others into val
#this gives approx 50K rows in tr, 25K in val
tr=train[train.Camp_Start_Date<='2005-08-31']
val=train[train.Camp_Start_Date>'2005-08-31']

Y_train=train.Outcome
Y_tr=tr.Outcome
Y_val=val.Outcome

test_PID=test.Patient_ID
test_HID=test.Health_Camp_ID

#drop all date, ID & Outcome columns. 
to_drop=['Registration_Date','Camp_Start_Date','Camp_End_Date','First_Interaction','id','Patient_ID','Health_Camp_ID','Outcome','prev_camp_end']
train=train.drop(to_drop,axis=1)
tr=tr.drop(to_drop,axis=1)
val=val.drop(to_drop,axis=1)
test=test.drop(to_drop,axis=1)

dtrain = xgb.DMatrix(train, label=Y_train)
dtr=xgb.DMatrix(tr, label=Y_tr)
dval=xgb.DMatrix(val, label=Y_val)
dtest = xgb.DMatrix(test)
num_rounds = 1000
watchlist = [(dtr, 'tr'),(dval, 'val')]

xgb_params = {
    'seed': 619, 
    'colsample_bytree': 0.75,
    'silent': 1,
    'subsample': 0.8,
    'learning_rate': 0.1,
    'objective': 'reg:logistic',
    'max_depth': 3, 
    'min_child_weight': 30, 
    'alpha': 0.02,
    'eval_metric': 'auc'
}

model1 = xgb.train(xgb_params, dtr, num_rounds, watchlist, early_stopping_rounds=20)

xgb1= xgb.train(xgb_params,dtrain,num_boost_round=model1.best_iteration)
pred1=xgb1.predict(dtest)

#convert predictions to percentiles, for rank-averaging while ensembling
temp = pred1.argsort()
ranks1 = np.empty(len(pred1), int)
ranks1[temp] = np.arange(len(pred1))
ranks1=np.array(ranks1, dtype=float)

min1=min(ranks1)
max1=max(ranks1)
pred1_perc=(ranks1-min1)/(max1-min1)

#build RandomForest Classifier for same features
RF=RandomForestClassifier(n_estimators=200,bootstrap=True,max_depth=6,min_samples_split= 2,min_samples_leaf= 2,
                         max_features="auto",verbose=1,n_jobs=-1,random_state=619)
RF.fit(tr,Y_tr)
val_pred_rf=RF.predict_proba(val)
roc_auc_score(Y_val,val_pred_rf[:,1]) #validation score

RF.fit(train,Y_train)
pred_rf=RF.predict_proba(test)[:,1]

#convert predictions to percentiles for rank-averaging
temp = pred_rf.argsort()
ranks2 = np.empty(len(pred_rf), int)
ranks2[temp] = np.arange(len(pred_rf))
ranks2=np.array(ranks2, dtype=float)

min2=min(ranks2)
max2=max(ranks2)
pred2_perc=(ranks2-min2)/(max2-min2)

#final prediction=weighted average 
pred_ens= pred1_perc* 0.9 + pred2_perc* 0.1
#write submission
submission = pd.DataFrame({"Patient_ID": test_PID,"Health_Camp_ID":test_HID, "Outcome": pred_ens})
submission.to_csv('ens_xgb_rf.csv',index=False)


#Plotting the XGB feature importance, saving to file for future analysis
def create_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1

    outfile.close()

create_feature_map(train.columns.values)

importance = model1.get_fscore(fmap='xgb.fmap')
importance = sorted(importance.items(), key=operator.itemgetter(1))

df = pd.DataFrame(importance, columns=['feature', 'fscore'])
df['fscore'] = df['fscore'] / df['fscore'].sum()
 
plt.figure()
df.plot()
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(10, 10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.gcf().savefig('feature_importance_xgb.png')