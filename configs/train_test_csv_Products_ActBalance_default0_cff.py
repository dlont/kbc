config={'annotation': 'Convert multiple .csv tables with unique identifier field into single file .csv file. Essentially concatenate different tables according to the unique index Client. Missing values of the Products_ActBalance table are filled with 0.',
 'command': 'python source/train_test_preparation/train_test_datasets.py -c configs/train_test_csv_Products_Act_Balance_cff.py',
 'Soc_Dem_file':'data/27_01_2020/Soc_Dem.csv',
 'Inflow_Outflow_file':'data/27_01_2020/Inflow_Outflow.csv',
 'Products_ActBalance_file':'data/27_01_2020/Products_ActBalance.csv',
 'Sales_Revenues_file':'data/27_01_2020/Sales_Revenues.csv',
 'output_file':{'file':'data/28_01_2020_1584entries/data_Products_ActBalance_default0.csv',
                'provenance':'data/28_01_2020_1584entries/data_Products_ActBalance_default0.prov'},
 'imputation':{
     'Count_CA':0,'Count_SA':0,'Count_MF':0,'Count_OVD':0,'Count_CC':0,'Count_CL':0,
     'ActBal_CA':0,'ActBal_SA':0,'ActBal_MF':0,'ActBal_OVD':0,'ActBal_CC':0,'ActBal_CL':0,
     'Sale_MF':-1,'Sale_CC':-1,'Sale_CL':-1,'Revenue_MF':-1,'Revenue_CC':-1,'Revenue_CL':-1
 },
 'dropna':['Sex','VolumeCred','VolumeCred_CA','TransactionsCred','TransactionsCred_CA','VolumeDeb',
           'VolumeDeb_CA','VolumeDebCash_Card','VolumeDebCashless_Card','VolumeDeb_PaymentOrder',
           'TransactionsDeb','TransactionsDeb_CA','TransactionsDebCash_Card','TransactionsDebCashless_Card','TransactionsDeb_PaymentOrder'],
}