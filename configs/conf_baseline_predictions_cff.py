from numpy import linspace
config={
  'annotation': 'Modelling data distributions.',
  'compatibility_version':'1.7pre',
  'command': 'bin/kbc_direct_marketing_mock.py -c configs/conf_baseline_predictions_cff.py',
  'latex_main': 'latex/report.tex',
  'model':{
   'type':'predictor',
   'multiclass_model':'results/Multiclass_RF_1.6pre_wo_outliers_all_v3/Multiclassification5_RF.pkl',
   'regression_CC':'results/Multiclass_RF_1.6pre_wo_outliers_all_v3/Revenue_CC_regression.pkl',
   'regression_CL':'results/Multiclass_RF_1.6pre_wo_outliers_all_v3/Revenue_CL_regression.pkl',
   'regression_MF':'results/Multiclass_RF_1.6pre_wo_outliers_all_v3/Revenue_MF_regression.pkl',
   'data_provider':'model_data_provider',
   'input_features_reg_CC':['Sex', 'Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder', 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL'],
   'input_features_reg_MF':['Sex', 'Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder', 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL'],
   'input_features_reg_CL':['Sex', 'Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder', 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL'],
   'input_features_multiclass':['Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'VolumeDeb', 'ActBal_CA', 'ActBal_SA'],
  },
  'mode': 'report',
  'views':['offers'],
  'offers':{
     'annotation': 'Ranked list of clients for marketing campaign',
     'type':'clients_rank',
     'output_filename':'clients_rank.csv',
  },
  'model_data_provider':{
        'type':'PandasDataProviderRespondingClientsNoOutliers',
        'remove_all':False,
        'training_set':False,
        'input_file':'data/28_01_2020_1584entries/data_Products_ActBalance_default0.csv'
 }
}