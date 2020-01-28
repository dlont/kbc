config={'annotation': 'Convert multiple .csv tables with unique identifier field into single file .csv file. Essentially concatenate different tables according to the unique index.',
 'command': 'python source/train_test_preparation/train_test_datasets_pre.py -c configs/train_test_csv_pre_cff.py',
 'Soc_Dem_file':'data/27_01_2020/Soc_Dem.csv',
 'Inflow_Outflow_file':'data/27_01_2020/Inflow_Outflow.csv',
 'Products_ActBalance_file':'data/27_01_2020/Products_ActBalance.csv',
 'Sales_Revenues_file':'data/27_01_2020/Sales_Revenues.csv',
 'output_train':{'file':'data/27_01_2020/data_train.csv',
                 'provenance':'data/27_01_2020/data_train.prov'},
 'output_test':{'file':'data/27_01_2020/data_test.csv',
                'provenance':'data/27_01_2020/data_test.prov'},
 'training_fraction':0.6
}