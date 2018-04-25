# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:58:39 2017

@author: Vishnu
"""
#
#import pandas as pd
#from collections import Counter
#
#excel = pd.ExcelFile("C:/Users/hp/Documents/Python Scripts/Rental_Prediction/Sample_Rental_Data_20171121.xlsx")
#sheets = excel.sheet_names
#final_list = []
#for sheet in sheets:
#    df = excel.parse(sheet)
#    unique_unit = list(df['Unit'].unique())
#    list_unit = [row['Unit'] for index, row in df.iterrows()]
#    list_name = [row['Trade Name'] for index, row in df.iterrows()]
#    df_unit = pd.DataFrame(list_unit)
#    df_name = pd.DataFrame(list_name)
#    frames = [df_unit, df_name]
#    result = pd.concat(frames, axis=1)
#    result.columns = [ "Unit", "Trade Name"]
#    list_new = [result[(result['Unit']==i)]['Trade Name'] for i in unique_unit]
#    new_df = pd.concat(list_new, axis=1, ignore_index=True)
#    new_df.columns = unique_unit
#    dict_df = new_df.to_dict()
#    dict_df = {k1:{k:v for k,v in v1.items() if pd.notnull(v)} for k1, v1 in dict_df.items()}
#    list_df = []
#    for key, value in dict_df.items():
#        df_dict = {}
#        df_dict[key] = list(value.values())
#        list_df.append(df_dict)
#    new_list = []
#    for i in list_df:
#        for k, v in i.items():
#            dict_new = {}
#            dict_new[k] = dict(Counter(v))
#            new_list.append(dict_new)
#    list_new = []
#    for j in new_list:
#        for k1, v1 in j.items(): 
#            dict_new = {}
#            dict_new[k1] = list(v1.values())
#            list_new.append(dict_new)
#    list_final = []
#    for k in list_new:
#        for k2, v2 in k.items():
#            dict_final = {}
#            if len(v2) > 2 and v2.count(1) == len(v2):
#                dict_final[k2] = 'Not Re-Rented'
#                print (sheet, k2)
#                list_final.append(dict_final)
#            else:
#                dict_final[k2] = 'Re-Rented'
#                list_final.append(dict_final)
#    final_dict = {}
#    final_dict[sheet] = list_final
#    final_list.append(final_dict)



import pandas as pd
from statistics import mean

excel = pd.ExcelFile("C:/Users/hp/Documents/Python Scripts/Rental_Prediction/Sample_Rental_Data_20171121.xlsx")
sheets = excel.sheet_names
final_list = []
for sheet in sheets:
    df = excel.parse(sheet)
    df = df.fillna(df.mean())
    unique_trade = list(df['Trade Name'].unique())
    unique_unit = list(df['Unit'].unique())
    list_unit = [row['Unit'] for index, row in df.iterrows()]
    list_name = [row['Trade Name'] for index, row in df.iterrows()]
    df_unit = pd.DataFrame(list_unit)
    df_name = pd.DataFrame(list_name)
    frames = [df_name, df_unit]
    result = pd.concat(frames, axis=1)
    result.columns = [ "Trade Name", "Unit"]
    list_new = [result[(result['Trade Name']==i)]['Unit'] for i in unique_trade]
    new_df = pd.concat(list_new, axis=1, ignore_index=True)
    new_df.columns = unique_trade
    dict_df = new_df.to_dict()
    dict_df = {k1:{k:v for k,v in v1.items() if pd.notnull(v)} for k1, v1 in dict_df.items()}
    list_df = []
    for key, value in dict_df.items():
        df_dict = {}
        df_dict[key] = list(value.values())
        list_df.append(df_dict)
    new_list = []
    for i in list_df:
        for k, v in i.items():
            for unit in unique_unit:
                if unit in v:
                    dict_new = {}
                    dict_new['Unit'] = unit
                    dict_new['Trade Name'] = k
                    dict_new['Count'] = v.count(unit)
                    dict_new['Average MFA'] = mean(df[(df.Unit == unit)]['MFA'].tolist())
                    dict_new['Average Face Rental'] = mean(df[(df.Unit == unit)]['Face Rental'].tolist())
                    dict_new['Average Net Rental'] = mean(df[(df.Unit == unit)]['Net Rental'].tolist())
                    dict_new['Average Eff. Rental'] = mean(df[(df.Unit == unit)]['Eff. Rental'].tolist())
                    dict_new['Average Face PSF'] = mean(df[(df.Unit == unit)]['Face PSF'].tolist())
                    dict_new['Average Net PSF'] = mean(df[(df.Unit == unit)]['Net PSF'].tolist())
                    dict_new['Average Eff PSF'] = mean(df[(df.Unit == unit)]['Eff PSF'].tolist())
                    dict_new['Average Govt. Rent/ Mth'] = mean(df[(df.Unit == unit)]['Govt. Rent/ Mth'].tolist())     
                    dict_new['Average Govt. Rates/ Mth'] = mean(df[(df.Unit == unit)]['Govt. Rates/ Mth'].tolist())        
                    dict_new['Average Govt Rent PSF/ Mth'] = mean(df[(df.Unit == unit)]['Govt Rent PSF/ Mth '].tolist())
                    dict_new['Average Govt Rates PSF/ Mth'] = mean(df[(df.Unit == unit)]['Govt Rates PSF/ Mth'].tolist())        
                    if v.count(unit) > 1:
                        dict_new['Label'] = 1
                    else:
                        dict_new['Label'] = 0
                    new_list.append(dict_new)
    df_new = pd.DataFrame(new_list)
    df_new.to_csv(sheet + '.csv', sep='\t', encoding='utf-8')
