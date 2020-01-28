from dataprovider import PandasSurvivedClassSelector, PandasDrownedClassSelector
from numpy import linspace
config={'annotation': 'Modelling data distributions.',
 'command': 'bin/kbc_direct_marketing_mock.py -c configs/conf_cff.py',
 'latex_main': 'latex/report.tex',
 'model':'advanced',
 'mode': 'report',
#  'views':['features_1d_Soc_Dem_p1','features_1d_Inflow_Outflow_p1','features_2d_train_test_correlations_p1'],
 'views':['features_1d_Soc_Dem_p1','features_1d_Inflow_Outflow_p1','features_1d_Inflow_Outflow_p2', 'features_1d_Targets_p1',
 'features_2d_Targets_correlations_p1'],
  'features_1d_Soc_Dem_p1':{
     'annotation': 'Age, Sex, Tenure train/test distributions.',
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
  'features_1d_Targets_p1':{
     'annotation': 'Sales and revenues p1.',
     'type':'1d_train_test',
     'output_filename':'features_1d_Targets_p1',
     'layout':{'nrows':2, 'ncols':3},
     'size': [8.5,5.0],
     'features':['Sale_MF','Sale_CC','Sale_CL','Revenue_MF','Revenue_CC','Revenue_CL']
  },
  'features_2d_Targets_correlations_p1':{
     'annotation': 'Revenue targets correlations.',
     'type':'2d_train_correlations',
     'output_filename':'features_2d_Targets_correlations_p1',
     'layout':{'nrows':1, 'ncols':3},
     'size': [8.5,5.0],
     'features':[['Revenue_MF','Revenue_CC'],['Revenue_MF','Revenue_CL'],['Revenue_CC','Revenue_CL']]
  },
  ##########
  'Sex':{
        'data_provider':'train_data_provider',
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
        'data_provider':'train_data_provider',
        'title':'Age, years',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,99.5,20),
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
        'data_provider':'train_data_provider',
        'title':'Tenure, month',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,300.5,31),
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
        'data_provider':'train_data_provider',
        'title':'monthly credit \n turnover [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50000.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly credit turnover \n on current accounts \n [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50000.5,20),
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
        'data_provider':'train_data_provider',
        'title':'number of all \n credit transactions',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,20),
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
        'data_provider':'train_data_provider',
        'title':'number of credit \n transactions on \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly debit \n turnover [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,10000.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly debit turnover \n on current accounts [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,10000.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly volume of \n debit cash transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,5000.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly volume of debit \n cashless transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,2000.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly volume of \n debit transactions \n via payment order [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,10000.5,20),
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
        'data_provider':'train_data_provider',
        'title':'number of all \n debit transactions',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,20),
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
        'data_provider':'train_data_provider',
        'title':'number of debit \n transactions on \n current accounts',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly number of \n debit cash transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,20.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly number of \n debit cashless transactions \n via card [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,20),
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
        'data_provider':'train_data_provider',
        'title':'monthly number of \n debit transactions \n via payment order [EUR]',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,25.5,25),
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
        'data_provider':'train_data_provider',
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
        'data_provider':'train_data_provider',
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
        'data_provider':'train_data_provider',
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
        'data_provider':'train_data_provider',
        'title':'target variable for \n revenue from mutual fund',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,25),
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
        'data_provider':'train_data_provider',
        'title':'target variable for \n revenue from credit card',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,25),
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
        'data_provider':'train_data_provider',
        'title':'target variable for \n revenue from consumer loan',
        'style':{
            'type' : 'numerical',
            'under_over_flow' : [False,True],
            'bins':linspace(-0.5,50.5,25),
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
  'train_data_provider':{
        'input_file':'data/28_01_2020/data_Products_ActBalance_default0.csv'
  }
 }