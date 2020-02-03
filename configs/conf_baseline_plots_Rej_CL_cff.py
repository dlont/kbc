from numpy import linspace
from dataprovider import SaleRejectedClassSelector, SaleCLClassSelector
config={
  'annotation': 'Modelling data distributions.',
  'compatibility_version':'1.6pre',
  'command': 'bin/kbc_direct_marketing_mock.py -c configs/conf_baseline_plots_cff.py',
  'latex_main': 'latex/report.tex',
  'model':{
   'type':'advanced_classification',
   'data_provider':'model_data_provider',
   'input_features':['Sex', 'Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder', 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL'],
   'target':['Sale_Multiclass'],
   'n_estimators':200,
   'max_depth':5,
   'learning_rate':0.01,
   'objective':'multi:softprob',
   'num_class':3
  },
  'mode': 'report',
  'views':['features_Rej_CL_1d_Soc_Dem_p1','features_Rej_CL_1d_Inflow_Outflow_p1','features_Rej_CL_1d_Inflow_Outflow_p2', 
  'features_Rej_CL_1d_Products_ActBalance_p1', 'features_Rej_CL_1d_Products_ActBalance_p2',
  'features_Rej_CL_1d_Targets_p1'],
  'features_Rej_CL_1d_Soc_Dem_p1':{
     'annotation': 'Age, Sex, Tenure train/test distributions p1.',
     'type':'1d_train_test',
     'output_filename':'features_Rej_CL_1d_Soc_Dem_p1',
     'layout':{'nrows':1, 'ncols':3},
     'size': [8.5,2.5],
     'features':['Sex','Age','Tenure']
  },
  'features_Rej_CL_1d_Inflow_Outflow_p1':{
     'annotation': 'Inflow/Outflow distribution p1.',
     'type':'1d_train_test',
     'output_filename':'features_Rej_CL_1d_Inflow_Outflow_p1',
     'layout':{'nrows':2, 'ncols':4},
     'size': [8.5,5.0],
     'features':['VolumeCred','VolumeCred_CA','TransactionsCred','TransactionsCred_CA','VolumeDeb','VolumeDeb_CA','VolumeDebCash_Card']
  },
  'features_Rej_CL_1d_Inflow_Outflow_p2':{
     'annotation': 'Inflow/Outflow distribution p2.',
     'type':'1d_train_test',
     'output_filename':'features_Rej_CL_1d_Inflow_Outflow_p2',
     'layout':{'nrows':2, 'ncols':4},
     'size': [8.5,5.0],
     'features':['VolumeDebCashless_Card','VolumeDeb_PaymentOrder','TransactionsDeb','TransactionsDeb_CA','TransactionsDebCash_Card','TransactionsDebCashless_Card','TransactionsDeb_PaymentOrder']
  },
  'features_Rej_CL_1d_Products_ActBalance_p1':{
     'annotation': 'Products account balance after imputation distributions p1.',
     'type':'1d_train_test',
     'output_filename':'features_Rej_CL_1d_Products_ActBalance_p1',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':['Count_CA','Count_SA','Count_MF','ActBal_CA','ActBal_SA','ActBal_MF']
  },
  'features_Rej_CL_1d_Products_ActBalance_p2':{
     'annotation': 'Products account balance after imputation distributions p2.',
     'type':'1d_train_test',
     'output_filename':'features_Rej_CL_1d_Products_ActBalance_p2',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':['Count_OVD','Count_CC','Count_CL','ActBal_OVD','ActBal_CC','ActBal_CL']
  },
  'features_Rej_CL_1d_Targets_p1':{
     'annotation': 'Sales and revenues p1.',
     'type':'1d_train_test',
     'output_filename':'features_Rej_CL_1d_Targets_p1',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL']
  },
  ##########
  'Sex':{
        'data_provider':'model_data_provider',
        'title':'Gender (M=0,F=1)',
        'style':{
            'type' : 'categorical',
            'under_over_flow' : [False,False],
            # 'bins':['male','female']
            'bins':[0,1]
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Age':{
        'data_provider':'model_data_provider',
        'title':'Age, years',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0,100,11),
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Tenure':{
        'data_provider':'model_data_provider',
        'title':'Tenure, month',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0,300,11),
            'logy':True,
            'legend':{'prop':{'size': 6}}
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  ###########
  'VolumeCred':{
        'data_provider':'model_data_provider',
        'title':'monthly credit \n turnover [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50000,21),
            'logy':True,
            'legend':{'prop':{'size': 6}}
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'VolumeCred_CA':{
        'data_provider':'model_data_provider',
        'title':'monthly credit turnover \n on current accounts \n [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50000.,21),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'TransactionsCred':{
        'data_provider':'model_data_provider',
        'title':'number of all \n credit transactions',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0,50.,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'TransactionsCred_CA':{
        'data_provider':'model_data_provider',
        'title':'number of credit \n transactions on \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0,50,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'VolumeDeb':{
        'data_provider':'model_data_provider',
        'title':'monthly debit \n turnover [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,10000.,21),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'VolumeDeb_CA':{
        'data_provider':'model_data_provider',
        'title':'monthly debit turnover \n on current accounts [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,10000.,21),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'VolumeDebCash_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly volume of \n debit cash transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,5000.,21),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  #############
  'VolumeDebCashless_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly volume of debit \n cashless transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,2000.,11),
            'logy':True,
            'legend':{'prop':{'size': 6}}
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'VolumeDeb_PaymentOrder':{
        'data_provider':'model_data_provider',
        'title':'monthly volume of \n debit transactions \n via payment order [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,10000.,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'TransactionsDeb':{
        'data_provider':'model_data_provider',
        'title':'number of all \n debit transactions',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50.,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'TransactionsDeb_CA':{
        'data_provider':'model_data_provider',
        'title':'number of debit \n transactions on \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50.,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'TransactionsDebCash_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly number of \n debit cash transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,20.,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'TransactionsDebCashless_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly number of \n debit cashless transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50.,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'TransactionsDeb_PaymentOrder':{
        'data_provider':'model_data_provider',
        'title':'monthly number of \n debit transactions \n via payment order [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,20.,11),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  #############
  'Sale_MF':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n sale of mutual fund',
        'style':{
            'type' : 'categorical',
            'under_over_flow' : [False,False],
            'bins':[-1,0,1],
            'legend':{'prop':{'size': 6}}
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Sale_CC':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n sale of credit card',
        'style':{
            'type' : 'categorical',
            'under_over_flow' : [False,False],
            'bins':[-1,0,1],
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Sale_CL':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n sale of consumer loan',
        'style':{
            'type' : 'categorical',
            'under_over_flow' : [False,False],
            'bins':[-1,0,1],
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Revenue_MF':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n revenue from mutual fund',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50.,26),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Revenue_CC':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n revenue from credit card',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50.,26),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Revenue_CL':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n revenue from consumer loan',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,50.,26),
            'logy':True,
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  #############
  'Count_CA':{
        'data_provider':'model_data_provider',
        'title':'number of live \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,5,6)-0.5,
            'legend':{'prop':{'size': 6}},
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Count_SA':{
        'data_provider':'model_data_provider',
        'title':'number of live \n saving accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,8.,9)-0.5,
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Count_MF':{
        'data_provider':'model_data_provider',
        'title':'number of live \n mutual funds',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,10.,11)-0.5,
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Count_OVD':{
        'data_provider':'model_data_provider',
        'title':'number of live \n overdrafts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,3.,4)-0.5,
            'legend':{'prop':{'size': 6}},
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Count_CC':{
        'data_provider':'model_data_provider',
        'title':'number of live \n credit cards',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,3.,4)-0.5,
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'Count_CL':{
        'data_provider':'model_data_provider',
        'title':'number of live \n consumer loans',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0,5,6)-0.5,
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'ActBal_CA':{
        'data_provider':'model_data_provider',
        'title':'actual current accounts \n balance [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0,20000.,21),
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'ActBal_SA':{
        'data_provider':'model_data_provider',
        'title':'actual saving accounts \n balance [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,20000.,21),
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'ActBal_MF':{
        'data_provider':'model_data_provider',
        'title':'actual mutual funds \n balance [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,20000.,21),
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'ActBal_OVD':{
        'data_provider':'model_data_provider',
        'title':'actual overdrafts balance \n (liability) [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,2000.,21),
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'ActBal_CC':{
        'data_provider':'model_data_provider',
        'title':'actual credit cards balance \n (liability) [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,2000.,21),
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  'ActBal_CL':{
        'data_provider':'model_data_provider',
        'title':'actual consumer loans \n balance (liability) [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(0.,10000.,21),
            'logy':True
        },
        'class1':SaleRejectedClassSelector(),
        'class2':SaleCLClassSelector(1),
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Rej all (train)',
        'class1_label_test':'Rej all (test)',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Sale_CL=1 (train)',
        'class2_label_test':'Sale_CL=1 (test)',
  },
  #############
  
  'model_data_provider':{
        'type':'PandasDataProviderRespondingClientsNoOutliers',
        'remove_all':True,
        'training_set':True,
        'input_file':'data/28_01_2020_1584entries/data_Products_ActBalance_default0.csv'
  }
 }