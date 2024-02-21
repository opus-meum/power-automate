##Automated recommendations in response to survey answers.
"""
Created on Mon Dec 04 2023

@authori ehlke_hepworth
"""
import os

#%% STRATEGY

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd
import glob
import os

#%%
excel_data = 'Capability Assessment Survey.xlsx'
df = pd.read_excel(excel_data)
df.to_csv('Capability Assessment Survey.csv')

os.remove('Capability Assessment Survey.xlsx')



dir_ = 'results/'
#dir2_= '/Users/ehlke/Desktop/'
data = pd.read_csv('Capability Assessment Survey.csv').iloc[-1]

#data = pd.read_csv(dir2_+'Capability Assessment Survey.csv')
print(data)
print(data2)

company = data.iloc[6]

#company = data.iloc[:,6]
fn = company



word_to_number = {'nan':0, 'Nascent': 1, 'Emerging': 2, 'Expanding': 3, 'Optimising':4, 'Mature': 5}  # Add more words and their corresponding numbers


####STRATEGY
strategy_capp = pd.read_csv(dir_+'Data/CapA_00_capabilitypurpose.csv')
strategy_caps = pd.read_csv(dir_+'Data/CapA_00_capabilitystakeholders.csv')
strategy_imps = pd.read_csv(dir_+'Data/CapA_00_imapctstrategy.csv')

sentences_list = []
for i in range(len(data)):
    row_data = data.iloc[8:].apply(lambda x: str(x).split(';')).sum()
    sentences_list.append(row_data)


one=[]
two=[]
three=[]
four=[]
five=[]


for i in range(len(sentences_list)):
    oo = strategy_capp[strategy_capp.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one.append(oo)
    tw = strategy_capp[strategy_capp.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two.append(tw)
    th = strategy_capp[strategy_capp.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three.append(th)
    fo = strategy_capp[strategy_capp.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four.append(fo)
    fi = strategy_capp[strategy_capp.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five.append(fi)
    print(one)
    print(two)
    print(three)
    print(four)
    print(five)

# Repeating the script for strategy_caps
one_caps=[]
two_caps=[]
three_caps=[]
four_caps=[]
five_caps=[]

for i in range(len(sentences_list)):
    oo_caps = strategy_caps[strategy_caps.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one_caps.append(oo_caps)
    tw_caps = strategy_caps[strategy_caps.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two_caps.append(tw_caps)
    th_caps = strategy_caps[strategy_caps.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three_caps.append(th_caps)
    fo_caps = strategy_caps[strategy_caps.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four_caps.append(fo_caps)
    fi_caps = strategy_caps[strategy_caps.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five_caps.append(fi_caps)
    print(three_caps)
    print(four_caps)
    print(five_caps)

# Repeating the script for strategy_imps
one_imps=[]
two_imps=[]
three_imps=[]
four_imps=[]
five_imps=[]

for i in range(len(sentences_list)):
    oo_imps = strategy_imps[strategy_imps.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one_imps.append(oo_imps)
    tw_imps = strategy_imps[strategy_imps.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two_imps.append(tw_imps)
    th_imps = strategy_imps[strategy_imps.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three_imps.append(th_imps)
    fo_imps = strategy_imps[strategy_imps.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four_imps.append(fo_imps)
    fi_imps = strategy_imps[strategy_imps.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five_imps.append(fi_imps)


#combined_data = pd.concat([one, one_caps, one_imps], axis=1)
data_dict = {'dimension':['Capability Purpose', 'Capability Stakeholders', 'Impact Strategy'],'Nascent': [one,one_caps,one_imps], 'Emerging': [two,two_caps,two_imps], 'Expanding': [three,three_caps,three_imps],\
              'Optimising': [four,four_caps,four_imps], 'Mature': [five,five_caps,five_imps]}

df = pd.DataFrame(data_dict)

tst=[]
#fn=[]
save_=[]

#for k in range(len(strategy_capp)):

for i in range(len(df)):
    print(i)
    tmp = df.iloc[:,1:].apply(lambda x: x.str[i])
    tst.append(tmp)
    #filename = data.iloc[i, 6] + '_.csv'
    #fn.append(filename)
    total_row = tst[i].sum()
    tmp = pd.concat([tst[i], pd.DataFrame([total_row])], ignore_index=True)
    tmp.insert(0,'Dimension',['Capability Purpose', 'Capability Stakeholders', 'Impact Strategy','Total'])
    columns_greater_than_zero = tmp.iloc[:,1:].apply(lambda row: row.index[row.astype(float) > 0].tolist(), axis=1)
    columns_greater_than_zero_df = columns_greater_than_zero.apply(pd.Series)
    columns_greater_than_zero_df = columns_greater_than_zero_df.replace(word_to_number)
    average_values = columns_greater_than_zero_df.mean(axis=1)
    average_values = average_values.fillna(0)
    print(average_values)   
    tmp['level'] = average_values
    last_value_level = tmp['level'].iloc[-1]
    tmp['level'].iloc[-1] = tmp['level'].mean()
    tmp['max'] = tmp['level'].round()
    tmp['max'] = tmp['max'].replace(0, 1)
    tmp['max_column'] = tmp['max'].map({v: k for k, v in word_to_number.items()})
 #   max_columns = tmp.iloc[:, 1:-1].apply(lambda x: x.idxmax() if x.max() > 0 else 'nan', axis=1)

    print(tmp)

    
    tmp.to_csv(str(dir_)+'Strategy_'+str(fn)+'_.csv', index=False)
    print(tmp)
    add_ = tmp[['level'][-1]].to_frame().rename(columns={"level":'Strategy'}).tail(1)
    add_=pd.DataFrame(add_)
    print(add_['Strategy'].values[0])
   # tmp.tail(1)[['level']].rename(columns={"level": 'Strategy'}).to_csv(dir_+'Summary_'+fn+ '_.csv', index=False)    
    add_.to_csv(str(dir_)+'Summary_'+str(fn)+'_.csv', index=False)   
    print(add_)


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
####TALENT
    
file1 = pd.read_csv(dir_+'Data/CapA_01_equipping.csv')

file2 = pd.read_csv(dir_+'Data/CapA_01_impactperformance.csv')
file3 = pd.read_csv(dir_+'Data/CapA_01_teamcomposition.csv')

one=[]
two=[]
three=[]
four=[]
five=[]


for i in range(len(sentences_list)):
    oo = file1[file1.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one.append(oo)
    tw = file1[file1.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two.append(tw)
    th = file1[file1.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three.append(th)
    fo = file1[file1.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four.append(fo)
    fi = file1[file1.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five.append(fi)

# Repeating the script for strategy_caps
one2=[]
two2=[]
three2=[]
four2=[]
five2=[]

for i in range(len(sentences_list)):
    oo_caps = file2[file2.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one2.append(oo_caps)
    tw_caps = file2[file2.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two2.append(tw_caps)
    th_caps = file2[file2.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three2.append(th_caps)
    fo_caps = file2[file2.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four2.append(fo_caps)
    fi_caps = file2[file2.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five2.append(fi_caps)

one3=[]
two3=[]
three3=[]
four3=[]
five3=[]

for i in range(len(sentences_list)):
    oo_imps = file3[file3.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one3.append(oo_imps)
    tw_imps = file3[file3.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two3.append(tw_imps)
    th_imps = file3[file3.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three3.append(th_imps)
    fo_imps = file3[file3.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four3.append(fo_imps)
    fi_imps = file3[file3.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five3.append(fi_imps)

#combined_data = pd.concat([one, one_caps, one_imps], axis=1)
data_dict = {'dimension':['Equipping', 'Impact Performance', 'Team Composition'],'Nascent': [one,one2,one3], 'Emerging': [two,two2,two3], 'Expanding': [three,three2,three3],\
          'Optimising': [four,four2,four3], 'Mature': [five,five2,five3]}

df = pd.DataFrame(data_dict)

tst=[]
#fn=[]
save_=[]
summary = []

#for k in range(len(strategy_capp)):

for i in range(len(df)):
    tmp = df.iloc[:,1:].apply(lambda x: x.str[i])
    tst.append(tmp)
# filename = data.iloc[i, 6] + '_.csv'
#fn.append(filename)
    total_row = tst[i].sum()
    tmp = pd.concat([tst[i], pd.DataFrame([total_row])], ignore_index=True)
    tmp.insert(0,'Dimension',['Equipping', 'Impact Performance', 'Team Composition','Total'])

    columns_greater_than_zero = tmp.iloc[:,1:].apply(lambda row: row.index[row.astype(float) > 0].tolist(), axis=1)
    columns_greater_than_zero_df = columns_greater_than_zero.apply(pd.Series)
    columns_greater_than_zero_df = columns_greater_than_zero_df.replace(word_to_number)
    average_values = columns_greater_than_zero_df.mean(axis=1)
    average_values = average_values.fillna(0)

    tmp['level'] = average_values
    last_value_level = tmp['level'].iloc[-1]
    tmp['level'].iloc[-1] = tmp['level'].mean()

    tmp['max'] = tmp['level'].round()
    tmp['max'] = tmp['max'].replace(0, 1)
    tmp['max_column'] = tmp['max'].map({v: k for k, v in word_to_number.items()})
    
    tmp['idx'] = tmp.index +1


    tmp.to_csv(dir_+'Talent_'+fn+'_.csv', index=False)
    print(tmp)
    add_ = tmp[['level'][-1]].to_frame().rename(columns={"level":'Talent'}).tail(1)
    add_=pd.DataFrame(add_)
    print(add_['Talent'].values[0])

    summary_ = pd.read_csv(dir_+'Summary_'+fn+'_.csv')
    print(summary_)
    print(add_['Talent'].values[0])

    summary_['Talent'] = add_['Talent'].values[0]
    print(summary_)
    summary_.to_csv(dir_+'Summary_' +fn+ '_.csv', index=False)


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
####PROCESSES
file1 = pd.read_csv(dir_+'Data/CapA_02_processes.csv')
file2 = pd.read_csv(dir_+'Data/CapA_02_responsibilityframework.csv')


one=[]
two=[]
three=[]
four=[]
five=[]


for i in range(len(sentences_list)):
    oo = file1[file1.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one.append(oo)
    tw = file1[file1.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two.append(tw)
    th = file1[file1.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three.append(th)
    fo = file1[file1.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four.append(fo)
    fi = file1[file1.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five.append(fi)

one2=[]
two2=[]
three2=[]
four2=[]
five2=[]

for i in range(len(sentences_list)):
    oo_caps = file2[file2.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one2.append(oo_caps)
    tw_caps = file2[file2.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two2.append(tw_caps)
    th_caps = file2[file2.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three2.append(th_caps)
    fo_caps = file2[file2.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four2.append(fo_caps)
    fi_caps = file2[file2.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five2.append(fi_caps)


#combined_data = pd.concat([one, one_caps, one_imps], axis=1)
data_dict = {'dimension':['Processes', 'Responsibility \nFramework'],'Nascent': [one,one2], 'Emerging': [two,two2], 'Expanding': [three,three2],\
      'Optimising': [four,four2], 'Mature': [five,five2]}

df = pd.DataFrame(data_dict)

tst=[]
#fn=[]
save_=[]

#for k in range(len(strategy_capp)):

for i in range(len(df)):
    tmp = df.iloc[:,1:].apply(lambda x: x.str[i])
    tst.append(tmp)
# filename = data.iloc[i, 6] + '_.csv'
   # fn.append(filename)
   # print(fn)
    total_row = tst[i].sum()
    print(total_row)
    tmp = pd.concat([tst[i], pd.DataFrame([total_row])], ignore_index=True)
    tmp.insert(0,'Dimension',['Processes', 'Responsibility \nFramework','Total'])

    columns_greater_than_zero = tmp.iloc[:,1:].apply(lambda row: row.index[row.astype(float) > 0].tolist(), axis=1)
    columns_greater_than_zero_df = columns_greater_than_zero.apply(pd.Series)
    columns_greater_than_zero_df = columns_greater_than_zero_df.replace(word_to_number)
    average_values = columns_greater_than_zero_df.mean(axis=1)
    average_values = average_values.fillna(0)
    print(average_values)   

    tmp['level'] = average_values
    last_value_level = tmp['level'].iloc[-1]
    tmp['level'].iloc[-1] = tmp['level'].mean()
            
    tmp['max'] = tmp['level'].round()
    tmp['max'] = tmp['max'].replace(0, 1)
    tmp['max_column'] = tmp['max'].map({v: k for k, v in word_to_number.items()})

    tmp['idx'] = tmp.index +1

    tmp.to_csv(dir_+'Processes_' +fn+ '_.csv', index=False)
    print(tmp)
    add_ = tmp[['level'][-1]].to_frame().rename(columns={"level":'Processes'}).tail(1)
    add_=pd.DataFrame(add_)
    print(add_['Processes'].values[0])

    summary_ = pd.read_csv(dir_+'Summary_'+fn+'_.csv')
    print(summary_)
    summary_['Processes'] = add_['Processes'].values[0]
    print(summary_)
    summary_.to_csv(dir_+'Summary_' +fn+ '_.csv', index=False)



#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
####DATA
file1 = pd.read_csv(dir_+'Data/CapA_03_dataaccess.csv')
file2 = pd.read_csv(dir_+'Data/CapA_03_datacollection.csv')
file3 = pd.read_csv(dir_+'Data/CapA_03_dataquality.csv')


one=[]
two=[]
three=[]
four=[]
five=[]


for i in range(len(sentences_list)):
    oo = file1[file1.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one.append(oo)
    tw = file1[file1.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two.append(tw)
    th = file1[file1.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three.append(th)
    fo = file1[file1.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four.append(fo)
    fi = file1[file1.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five.append(fi)

one2=[]
two2=[]
three2=[]
four2=[]
five2=[]

for i in range(len(sentences_list)):
    oo_caps = file2[file2.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one2.append(oo_caps)
    tw_caps = file2[file2.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two2.append(tw_caps)
    th_caps = file2[file2.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three2.append(th_caps)
    fo_caps = file2[file2.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four2.append(fo_caps)
    fi_caps = file2[file2.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five2.append(fi_caps)

one3=[]
two3=[]
three3=[]
four3=[]
five3=[]

for i in range(len(sentences_list)):
    oo_imps = file3[file3.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one3.append(oo_imps)
    tw_imps = file3[file3.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two3.append(tw_imps)
    th_imps = file3[file3.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three3.append(th_imps)
    fo_imps = file3[file3.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four3.append(fo_imps)
    fi_imps = file3[file3.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five3.append(fi_imps)


#combined_data = pd.concat([one, one_caps, one_imps], axis=1)
data_dict = {'dimension':['Data Access', 'Data Collection', 'Data Quality'],'Nascent': [one,one2,one3], 'Emerging': [two,two2,two3], 'Expanding': [three,three2,three3],\
  'Optimising': [four,four2,four3], 'Mature': [five,five2,five3]}

df = pd.DataFrame(data_dict)

tst=[]
#fn=[]
save_=[]

#for k in range(len(strategy_capp)):

for i in range(len(df)):
    tmp = df.iloc[:,1:].apply(lambda x: x.str[i])
    tst.append(tmp)
# filename = data.iloc[i, 6] + '_.csv'
# fn.append(filename)
    total_row = tst[i].sum()
    print(total_row)
    tmp = pd.concat([tst[i], pd.DataFrame([total_row])], ignore_index=True)
    tmp.insert(0,'Dimension',['Data Access', 'Data Collection', 'Data Quality','Total'])
    print(tmp)

    columns_greater_than_zero = tmp.iloc[:,1:].apply(lambda row: row.index[row.astype(float) > 0].tolist(), axis=1)
    columns_greater_than_zero_df = columns_greater_than_zero.apply(pd.Series)
    columns_greater_than_zero_df = columns_greater_than_zero_df.replace(word_to_number)
    average_values = columns_greater_than_zero_df.mean(axis=1)
    average_values = average_values.fillna(0)
    print(average_values)   
    tmp['level'] = average_values

    last_value_level = tmp['level'].iloc[-1]
    tmp['level'].iloc[-1] = tmp['level'].mean()

    tmp['max'] = tmp['level'].round()
    tmp['max'] = tmp['max'].replace(0, 1)
    tmp['max_column'] = tmp['max'].map({v: k for k, v in word_to_number.items()})

    tmp['idx'] = tmp.index +1

    tmp.to_csv(dir_+'Data_' +fn+ '_.csv', index=False)

    add_ = tmp[['level'][-1]].to_frame().rename(columns={"level":'Data'}).tail(1)
    add_=pd.DataFrame(add_)
    print(add_['Data'].values[0])

    summary_ = pd.read_csv(dir_+'Summary_'+fn+ '_.csv')
    print(summary_)
    summary_['Data'] = add_['Data'].values[0]
    print(summary_)
    summary_.to_csv(dir_+'Summary_' +fn+ '_.csv', index=False)

#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
####REPORTING  
file1 = pd.read_csv(dir_+'Data/CapA_05_framework.csv')
file2 = pd.read_csv(dir_+'Data/CapA_05_standards.csv')

one=[]
two=[]
three=[]
four=[]
five=[]


for i in range(len(sentences_list)):
    oo = file1[file1.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one.append(oo)
    tw = file1[file1.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two.append(tw)
    th = file1[file1.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three.append(th)
    fo = file1[file1.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four.append(fo)
    fi = file1[file1.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five.append(fi)

one2=[]
two2=[]
three2=[]
four2=[]
five2=[]

for i in range(len(sentences_list)):
    oo_caps = file2[file2.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one2.append(oo_caps)
    tw_caps = file2[file2.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two2.append(tw_caps)
    th_caps = file2[file2.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three2.append(th_caps)
    fo_caps = file2[file2.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four2.append(fo_caps)
    fi_caps = file2[file2.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five2.append(fi_caps)


#combined_data = pd.concat([one, one_caps, one_imps], axis=1)
data_dict = {'dimension':['Reporting Framework', 'Reporting Standards'],'Nascent': [one,one2], 'Emerging': [two,two2], 'Expanding': [three,three2],\
'Optimising': [four,four2], 'Mature': [five,five2]}

df = pd.DataFrame(data_dict)

tst=[]
#fn=[]
save_=[]

#for k in range(len(strategy_capp)):

for i in range(len(df)):
    tmp = df.iloc[:,1:].apply(lambda x: x.str[i])
    tst.append(tmp)
#filename = data.iloc[i, 6] + '_.csv'
    #fn.append(filename)
    total_row = tst[i].sum()
    print(total_row)
    tmp = pd.concat([tst[i], pd.DataFrame([total_row])], ignore_index=True)
    tmp.insert(0,'Dimension',['Reporting Framework', 'Reporting Standards','Total'])
    print(tmp)

    columns_greater_than_zero = tmp.iloc[:,1:].apply(lambda row: row.index[row.astype(float) > 0].tolist(), axis=1)
    columns_greater_than_zero_df = columns_greater_than_zero.apply(pd.Series)
    columns_greater_than_zero_df = columns_greater_than_zero_df.replace(word_to_number)
    average_values = columns_greater_than_zero_df.mean(axis=1)
    average_values = average_values.fillna(0)
    print(average_values)   

    tmp['level'] = average_values
    last_value_level = tmp['level'].iloc[-1]
    tmp['level'].iloc[-1] = tmp['level'].mean()

    tmp['max'] = tmp['level'].round()
    tmp['max'] = tmp['max'].replace(0, 1)
    tmp['max_column'] = tmp['max'].map({v: k for k, v in word_to_number.items()})

    tmp['idx'] = tmp.index +1

    tmp.to_csv(dir_+'Reporting_' +fn+ '_.csv', index=False)

    print(tmp)
    add_ = tmp[['level'][-1]].to_frame().rename(columns={"level":'Reporting'}).tail(1)
    add_=pd.DataFrame(add_)
    print(add_['Reporting'].values[0])

    summary_ = pd.read_csv(dir_+'Summary_'+fn+ '_.csv')
    print(summary_)
    summary_['Reporting'] = add_['Reporting'].values[0]
    print(summary_)
    summary_.to_csv(dir_+'Summary_' +fn+ '_.csv', index=False)


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
####TECHNOLOGY  

file1 = pd.read_csv(dir_+'Data/CapA_06_technology.csv')


one=[]
two=[]
three=[]
four=[]
five=[]


for i in range(len(sentences_list)):
    oo = file1[file1.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one.append(oo)
    tw = file1[file1.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two.append(tw)
    th = file1[file1.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three.append(th)
    fo = file1[file1.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four.append(fo)
    fi = file1[file1.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five.append(fi)



#combined_data = pd.concat([one, one_caps, one_imps], axis=1)
data_dict = {'dimension':['Techology'],'Nascent': [one], 'Emerging': [two], 'Expanding': [three],\
        'Optimising': [four], 'Mature': [five]}

df = pd.DataFrame(data_dict)
print(len(data))

tst=[]
#fn=[]
save_=[]

#for k in range(len(strategy_capp)):

for i in range(len(data)):
    tmp = df.iloc[:,1:].apply(lambda x: x.str[i])
    tst.append(tmp)
#filename = data.iloc[i, 6] + '_.csv'
#fn.append(filename)
    total_row = tst[i].sum()
    print(total_row)
    tmp = pd.concat([tst[i], pd.DataFrame([total_row])], ignore_index=True)
    tmp.insert(0,'Dimension',['Techology','Total'])

    columns_greater_than_zero = tmp.iloc[:,1:].apply(lambda row: row.index[row.astype(float) > 0].tolist(), axis=1)
    columns_greater_than_zero_df = columns_greater_than_zero.apply(pd.Series)
    columns_greater_than_zero_df = columns_greater_than_zero_df.replace(word_to_number)
    average_values = columns_greater_than_zero_df.mean(axis=1)
    average_values = average_values.fillna(0)
    print(average_values)   

    tmp['level'] = average_values
    last_value_level = tmp['level'].iloc[-1]
    tmp['level'].iloc[-1] = tmp['level'].mean()

    tmp['max'] = tmp['level'].round()
    tmp['max'] = tmp['max'].replace(0, 1)
    tmp['max_column'] = tmp['max'].map({v: k for k, v in word_to_number.items()})

    tmp['idx'] = tmp.index +1

    tmp.to_csv(dir_+'Technology_' +fn+ '_.csv', index=False)

    print(tmp)

    add_ = tmp[['level'][-1]].to_frame().rename(columns={"level":'Technology'}).tail(1)
    add_=pd.DataFrame(add_)
    print(add_['Technology'].values[0])

    summary_ = pd.read_csv(dir_+'Summary_'+fn+ '_.csv')
    print(summary_)
    summary_['Technology'] = add_['Technology'].values[0]
    print(summary_)
    summary_.to_csv(dir_+'Summary_' +fn+ '_.csv', index=False)


#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------
####MEASUREMENT

file1 = pd.read_csv(dir_+'Data/CapA_04_IMF.csv')
file2 = pd.read_csv(dir_+'Data/CapA_04_TT.csv')
file3 = pd.read_csv(dir_+'Data/CapA_04_evaluation.csv')
file4 = pd.read_csv(dir_+'Data/CapA_04_rki.csv')


one=[]
two=[]
three=[]
four=[]
five=[]


for i in range(len(sentences_list)):
    oo = file1[file1.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one.append(oo)
    tw = file1[file1.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two.append(tw)
    th = file1[file1.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three.append(th)
    fo = file1[file1.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four.append(fo)
    fi = file1[file1.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five.append(fi)

one2=[]
two2=[]
three2=[]
four2=[]
five2=[]

for i in range(len(sentences_list)):
    oo_caps = file2[file2.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one2.append(oo_caps)
    tw_caps = file2[file2.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two2.append(tw_caps)
    th_caps = file2[file2.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three2.append(th_caps)
    fo_caps = file2[file2.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four2.append(fo_caps)
    fi_caps = file2[file2.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five2.append(fi_caps)

one3=[]
two3=[]
three3=[]
four3=[]
five3=[]

for i in range(len(sentences_list)):
    oo_imps = file3[file3.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one3.append(oo_imps)
    tw_imps = file3[file3.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two3.append(tw_imps)
    th_imps = file3[file3.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three3.append(th_imps)
    fo_imps = file3[file3.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four3.append(fo_imps)
    fi_imps = file3[file3.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five3.append(fi_imps)

one4=[]
two4=[]
three4=[]
four4=[]
five4=[]

for i in range(len(sentences_list)):
    oo_imps = file4[file4.isin(sentences_list[i])].groupby(['nascent']).size().sum()
    one4.append(oo_imps)
    tw_imps = file4[file4.isin(sentences_list[i])].groupby(['emerging']).size().sum()
    two4.append(tw_imps)
    th_imps = file4[file4.isin(sentences_list[i])].groupby(['expanding']).size().sum()
    three4.append(th_imps)
    fo_imps = file4[file4.isin(sentences_list[i])].groupby(['optimising']).size().sum()
    four4.append(fo_imps)
    fi_imps = file4[file4.isin(sentences_list[i])].groupby(['mature']).size().sum()
    five4.append(fi_imps)


#combined_data = pd.concat([one, one_caps, one_imps], axis=1)
data_dict = {'dimension':['IMF', 'Tools & Templates', 'Evaulation', 'Risks, Knowledge,\n& Insights'],'Nascent': [one,one2,one3,one4], 'Emerging': [two,two2,two3,two4], 'Expanding': [three,three2,three3,three4],\
            'Optimising': [four,four2,four3,four4], 'Mature': [five,five,five3,five4]}

df = pd.DataFrame(data_dict)
print(df)

tst=[]
#fn=[]
save_=[]

#for k in range(len(strategy_capp)):

for i in range(len(df)):
    tmp = df.iloc[:,1:].apply(lambda x: x.str[i])
    tst.append(tmp)
# filename = data.iloc[i, 6] + '_.csv'
    #print(filename)
# fn.append(filename)
#print(fn)
    total_row = tst[i].sum()

    print(total_row)
    tmp = pd.concat([tst[i], pd.DataFrame([total_row])], ignore_index=True)
    tmp.insert(0,'Dimension',['IMF', 'Tools & Templates', 'Evaulation', 'Risks, Knowledge,\n& Insights','Total'])
    columns_greater_than_zero = tmp.iloc[:,1:].apply(lambda row: row.index[row.astype(float) > 0].tolist(), axis=1)
    columns_greater_than_zero_df = columns_greater_than_zero.apply(pd.Series)
    columns_greater_than_zero_df = columns_greater_than_zero_df.replace(word_to_number)
    average_values = columns_greater_than_zero_df.mean(axis=1)
    average_values = average_values.fillna(0)
    print(average_values)   

    tmp['level'] = average_values
    last_value_level = tmp['level'].iloc[-1]
    tmp['level'].iloc[-1] = tmp['level'].mean()

    tmp['max'] = tmp['level'].round()
    tmp['max'] = tmp['max'].replace(0, 1)
    tmp['max_column'] = tmp['max'].map({v: k for k, v in word_to_number.items()})

    tmp['idx'] = tmp.index +1
    
    tmp.to_csv(dir_+'Measurement_' +fn+ '_.csv', index=False)

    add_ = tmp[['level'][-1]].to_frame().rename(columns={"level":'Measurement'}).tail(1)
    add_=pd.DataFrame(add_)
    print(add_['Measurement'].values[0])

    summary_ = pd.read_csv(dir_+'Summary_'+fn+ '_.csv')
    print(summary_)
    summary_['Measurement'] = add_['Measurement'].values[0]
    print(summary_)
    summary_.to_csv(dir_+'Summary_' +fn+ '_.csv', index=False)
                            
