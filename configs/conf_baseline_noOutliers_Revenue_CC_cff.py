from numpy import linspace
config={
  'annotation': 'Modelling data distributions.',
  'compatibility_version':'1.5pre',
  'command': 'bin/kbc_direct_marketing_mock.py -c configs/conf_cff.py',
  'latex_main': 'latex/report.tex',
  'model':{
   'type':'advanced_regression',
   'data_provider':'model_data_provider',
   'input_features':['Sex', 'Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder', 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL'],
   'target':['Revenue_CC'],
   'n_estimators':200,
   'max_depth':5,
   'learning_rate':0.01,
  },
  'mode': 'report',
  'views':['features_1d_Soc_Dem_p1','features_1d_Inflow_Outflow_p1','features_1d_Inflow_Outflow_p2',
           'features_1d_Products_ActBalance_p1','features_1d_Products_ActBalance_p2',
           'features_1d_Targets_p1','features_2d_Targets_correlations_p1','features_2d_Targets_Inputs_correlations_p1',
           'features_2d_Targets_Inputs_correlations_p2','features_2d_Targets_Inputs_correlations_p3',
           'features_2d_Targets_Inputs_correlations_p4','learning_curve'],
  'learning_curve':{
     'annotation': 'Learning curve XGBoost',
     'type':'regression_model_learning_curve',
     'output_filename':'Revenue_CC_learning_curve',
     'layout':{'nrows':1, 'ncols':2},
     'size': [8.5,2.5],
     'metrics':['rmse','mae']
  },
  'features_1d_Soc_Dem_p1':{
     'annotation': 'Age, Sex, Tenure train/test distributions p1.',
     'type':'1d_train_test',
     'output_filename':'features_1d_Soc_Dem_p1',
     'layout':{'nrows':1, 'ncols':3},
     'size': [8.5,2.5],
     'features':['Sex','Age','Tenure']
  },
  'features_1d_Inflow_Outflow_p1':{
     'annotation': 'Inflow/Outflow distribution p1.',
     'type':'1d_train_test',
     'output_filename':'features_1d_Inflow_Outflow_p1',
     'layout':{'nrows':2, 'ncols':4},
     'size': [8.5,5.0],
     'features':['VolumeCred','VolumeCred_CA','TransactionsCred','TransactionsCred_CA','VolumeDeb','VolumeDeb_CA','VolumeDebCash_Card']
  },
  'features_1d_Inflow_Outflow_p2':{
     'annotation': 'Inflow/Outflow distribution p2.',
     'type':'1d_train_test',
     'output_filename':'features_1d_Inflow_Outflow_p2',
     'layout':{'nrows':2, 'ncols':4},
     'size': [8.5,5.0],
     'features':['VolumeDebCashless_Card','VolumeDeb_PaymentOrder','TransactionsDeb','TransactionsDeb_CA','TransactionsDebCash_Card','TransactionsDebCashless_Card','TransactionsDeb_PaymentOrder']
  },
  'features_1d_Products_ActBalance_p1':{
     'annotation': 'Products account balance after imputation distributions p1.',
     'type':'1d_train_test',
     'output_filename':'features_1d_Products_ActBalance_p1',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':['Count_CA','Count_SA','Count_MF','ActBal_CA','ActBal_SA','ActBal_MF']
  },
  'features_1d_Products_ActBalance_p2':{
     'annotation': 'Products account balance after imputation distributions p2.',
     'type':'1d_train_test',
     'output_filename':'features_1d_Products_ActBalance_p2',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':['Count_OVD','Count_CC','Count_CL','ActBal_OVD','ActBal_CC','ActBal_CL']
  },
  'features_1d_Targets_p1':{
     'annotation': 'Sales and revenues p1.',
     'type':'1d_train_test',
     'output_filename':'features_1d_Targets_p1',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL']
  },
  'features_2d_Targets_Inputs_correlations_p1':{
     'annotation': 'Revenue CC VS inputs correlations p1.',
     'type':'2d_train_correlations',
     'output_filename':'features_2d_Targets_Inputs_correlations_p1',
     'layout':{'nrows':1, 'ncols':2},
     'size': [8.5,5.0],
     'features':[['Revenue_CC','Age'], ['Revenue_CC','Tenure']]
  },
  'features_2d_Targets_Inputs_correlations_p2':{
     'annotation': 'Revenue CC VS inputs correlations p2.',
     'type':'2d_train_correlations',
     'output_filename':'features_2d_Targets_Inputs_correlations_p2',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':[['Revenue_CC','VolumeCred'],['Revenue_CC','VolumeCred_CA'],
                 ['Revenue_CC','TransactionsCred'], ['Revenue_CC','TransactionsCred_CA'],
                 ['Revenue_CC','VolumeDeb'], ['Revenue_CC','VolumeDeb_CA']]
  },
  'features_2d_Targets_Inputs_correlations_p3':{
     'annotation': 'Revenue CC VS inputs correlations p3.',
     'type':'2d_train_correlations',
     'output_filename':'features_2d_Targets_Inputs_correlations_p3',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':[['Revenue_CC','VolumeDebCash_Card'], ['Revenue_CC','VolumeDebCashless_Card'], 
                 ['Revenue_CC','VolumeDeb_PaymentOrder'], ['Revenue_CC','TransactionsDebCash_Card'],
                 ['Revenue_CC','TransactionsDebCashless_Card'], ['Revenue_CC','TransactionsDeb_PaymentOrder']]
  },
  'features_2d_Targets_Inputs_correlations_p4':{
     'annotation': 'Revenue CC VS inputs correlations p4.',
     'type':'2d_train_correlations',
     'output_filename':'features_2d_Targets_Inputs_correlations_p4',
     'layout':{'nrows':1, 'ncols':2},
     'size': [8.5,5.0],
     'features':[['Revenue_CC','TransactionsDeb'], ['Revenue_CC','TransactionsDeb_CA']]
  },
  'features_2d_Targets_correlations_p1':{
     'annotation': 'Revenue CC VS targets correlations p1.',
     'type':'2d_train_correlations',
     'output_filename':'features_2d_Targets_correlations_p1',
     'layout':{'nrows':1, 'ncols':3},
     'size': [8.5,5.0],
     'features':[['Revenue_CC','Revenue_MF'],['Revenue_CC','Revenue_CL'],['Revenue_CL','Revenue_MF']]
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
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Age':{
        'data_provider':'model_data_provider',
        'title':'Age, years',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,99.5,20,endpoint=False),
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Tenure':{
        'data_provider':'model_data_provider',
        'title':'Tenure, month',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,299.5,20,endpoint=False),
            'logy':True,
            'legend':{'prop':{'size': 6}}
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  ###########
  'VolumeCred':{
        'data_provider':'model_data_provider',
        'title':'monthly credit \n turnover [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49999.5,20,endpoint=False),
            'logy':True,
            'legend':{'prop':{'size': 6}}
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'VolumeCred_CA':{
        'data_provider':'model_data_provider',
        'title':'monthly credit turnover \n on current accounts \n [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49999.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'TransactionsCred':{
        'data_provider':'model_data_provider',
        'title':'number of all \n credit transactions',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'TransactionsCred_CA':{
        'data_provider':'model_data_provider',
        'title':'number of credit \n transactions on \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'VolumeDeb':{
        'data_provider':'model_data_provider',
        'title':'monthly debit \n turnover [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,9999.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'VolumeDeb_CA':{
        'data_provider':'model_data_provider',
        'title':'monthly debit turnover \n on current accounts [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,9999.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'VolumeDebCash_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly volume of \n debit cash transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,4999.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  #############
  'VolumeDebCashless_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly volume of debit \n cashless transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,1999.5,20,endpoint=False),
            'logy':True,
            'legend':{'prop':{'size': 6}}
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'VolumeDeb_PaymentOrder':{
        'data_provider':'model_data_provider',
        'title':'monthly volume of \n debit transactions \n via payment order [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,9999.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'TransactionsDeb':{
        'data_provider':'model_data_provider',
        'title':'number of all \n debit transactions',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'TransactionsDeb_CA':{
        'data_provider':'model_data_provider',
        'title':'number of debit \n transactions on \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'TransactionsDebCash_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly number of \n debit cash transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,19.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'TransactionsDebCashless_Card':{
        'data_provider':'model_data_provider',
        'title':'monthly number of \n debit cashless transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,20,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'TransactionsDeb_PaymentOrder':{
        'data_provider':'model_data_provider',
        'title':'monthly number of \n debit transactions \n via payment order [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,24.5,25,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
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
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Sale_CC':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n sale of credit card',
        'style':{
            'type' : 'categorical',
            'under_over_flow' : [False,False],
            'bins':[-1,0,1],
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Sale_CL':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n sale of consumer loan',
        'style':{
            'type' : 'categorical',
            'under_over_flow' : [False,False],
            'bins':[-1,0,1],
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Revenue_MF':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n revenue from mutual fund',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,25,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Revenue_CC':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n revenue from credit card',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,25,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Revenue_CL':{
        'data_provider':'model_data_provider',
        'title':'target variable for \n revenue from consumer loan',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,49.5,25,endpoint=False),
            'logy':True,
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  #############
  'Count_CA':{
        'data_provider':'model_data_provider',
        'title':'number of live \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,5.5,6,endpoint=False),
            'legend':{'prop':{'size': 6}},
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Count_SA':{
        'data_provider':'model_data_provider',
        'title':'number of live \n saving accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,7.5,8,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Count_MF':{
        'data_provider':'model_data_provider',
        'title':'number of live \n mutual funds',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,10.5,11,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Count_OVD':{
        'data_provider':'model_data_provider',
        'title':'number of live \n overdrafts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,2.5,3,endpoint=False),
            'legend':{'prop':{'size': 6}},
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Count_CC':{
        'data_provider':'model_data_provider',
        'title':'number of live \n credit cards',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,2.5,3,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'Count_CL':{
        'data_provider':'model_data_provider',
        'title':'number of live \n consumer loans',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,5.5,6,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'ActBal_CA':{
        'data_provider':'model_data_provider',
        'title':'actual current accounts \n balance [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,99999.5,20,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'ActBal_SA':{
        'data_provider':'model_data_provider',
        'title':'actual saving accounts \n balance [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,99999.5,20,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'ActBal_MF':{
        'data_provider':'model_data_provider',
        'title':'actual mutual funds \n balance [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,99999.5,20,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'ActBal_OVD':{
        'data_provider':'model_data_provider',
        'title':'actual overdrafts balance \n (liability) [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,1999.5,20,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'ActBal_CC':{
        'data_provider':'model_data_provider',
        'title':'actual credit cards balance \n (liability) [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,1999.5,20,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  'ActBal_CL':{
        'data_provider':'model_data_provider',
        'title':'actual consumer loans \n balance (liability) [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,9999.5,20,endpoint=False),
            'logy':True
        },
        'class1':None,
        'class2':None,
        'class1_marker_test':'.',
        'class1_color_train':'r',
        'class1_color_test':'r',
        'class1_line_train':'',
        'class1_line_test':'',
        'class1_label_train':'Train',
        'class1_label_test':'Test',
        'class2_marker_test':'.',
        'class2_color_train':'b',
        'class2_color_test':'b',
        'class2_line_train':'',
        'class2_line_test':'',
        'class2_label_train':'Drowned (train)',
        'class2_label_test':'Drowned (test)',
  },
  #############

  'model_data_provider':{
        'type':'PandasDataProviderRespondingClientsNoOutliersRevenueCC',
        'remove_all':False,
        'input_file':'data/28_01_2020_1584entries/data_Products_ActBalance_default0.csv'
  }
 }