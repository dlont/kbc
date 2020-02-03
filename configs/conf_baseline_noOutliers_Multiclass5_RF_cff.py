from numpy import linspace
config={
  'annotation': 'Modelling data distributions.',
  'compatibility_version':'1.6pre',
  'command': 'bin/kbc_direct_marketing_mock.py -c configs/conf_baseline_noOutliers_Multiclass_cff.py',
  'latex_main': 'latex/report.tex',
  'model':{
   'type':'advanced_classification_rf',
   'output_filename':'build/Multiclassification5_RF.pkl',
   'data_provider':'model_data_provider',
   'do_training':True,
#    'input_features':['Sex', 'Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'TransactionsCred', 'TransactionsCred_CA', 'VolumeDeb', 'VolumeDeb_CA', 'VolumeDebCash_Card', 'VolumeDebCashless_Card', 'VolumeDeb_PaymentOrder', 'TransactionsDeb', 'TransactionsDeb_CA', 'TransactionsDebCash_Card', 'TransactionsDebCashless_Card', 'TransactionsDeb_PaymentOrder', 'Count_CA', 'Count_SA', 'Count_MF', 'Count_OVD', 'Count_CC', 'Count_CL', 'ActBal_CA', 'ActBal_SA', 'ActBal_MF', 'ActBal_OVD', 'ActBal_CC', 'ActBal_CL'],
   'input_features':['Age', 'Tenure', 'VolumeCred', 'VolumeCred_CA', 'VolumeDeb', 'ActBal_CA', 'ActBal_SA'],
   'target':['Sale_Multiclass'],
   'n_estimators':[100,500,1000,5000],
   'max_depth':[5,7],
   'criterion':'gini',
   'class_weight':'balanced',
   'min_samples_leaf':[20]
  },
  'mode': 'report',
#   'views':['prob_correlations','confusion_matrix','output_classifier','ROC'],
  'views':['prob_correlations','confusion_matrix','output_classifier','importance','ROC'],
  'importance':{
     'annotation': 'RF_feature_improtance',
     'type':'rf_feature_importance',
     'output_filename':'rf_feature_importance',
     'size': [8.5,5.0],
  },
  'ROC':{
     'annotation': 'ROC Random Forest',
     'type':'multiclassification_roc',
     'output_filename':'Sale_multiclass_roc',
     'layout':{'nrows':1, 'ncols':2},
     'size': [8.5,5.0],
     'class_names':['0','1','2','3','4']
  },
  'learning_curve':{
     'annotation': 'Learning curve Random Forest',
     'type':'multiclassification_learning_curve',
     'output_filename':'Sale_multiclass_learning_curve',
     'layout':{'nrows':1, 'ncols':2},
     'size': [8.5,2.5],
     'metrics':["merror", "mlogloss"],
  },
  'output_classifier':{
     'annotation': 'Classifier train/test samples',
     'type':'multiclassification_classifier_distribution',
     'output_filename':'multiclass_classifier_distribution',
     'layout':{'nrows':1, 'ncols':2},
     'size': [8.5,5.0],
     'distributions':['classifier'],
     'style':{
        'type' : 'numerical',
        'under_over_flow' : [False,False],
        'bins':linspace(0,4,5)-0.5,
     },
     'class1_line_test':'',
     'class1_line_train':'',
     'class1_marker_train':'.',
     'class1_marker_test':'.',
     'class1_color_train':'r',
     'class1_color_test':'r',
     'class1_label_train':'Train',
     'class1_label_test':'Test',
  },
  'confusion_matrix':{
     'annotation': 'Classifier confusion matrix',
     'type':'multiclassification_confusion_matrix',
     'output_filename':'multiclass_confusion_matrix',
     'layout':{'nrows':1, 'ncols':2},
     'size': [8.5,5.0],
     'class_names':['0','1','2','3','4']
  },
  'prob_correlations':{
     'annotation': 'Classifier train/test samples',
     'type':'multiclassification_prob_correlations',
     'output_filename':'multiclass_prob_correlations',
     'layout':{'nrows':2, 'ncols':2},
     'size': [8.5,8.5],
     'distributions':['correlation_Rej','correlation_MF','correlation_CC','correlation_CL'],

     'correlation_Rej':{
     'features_id':[(0,1),(0,2),(0,3)],
     'xlabel':'Probability Reject',
     'ylabel':'Probability Sale_MF=1,Sale_CC=1,Sale_CL=1',
     'class0_marker_train':'.',
     'class0_marker_test':'x',
     'class0_color_train':'g',
     'class0_color_test':'g',
     'class0_label_train':'Rej VS MF (train)',
     'class0_label_test':'Rej VS MF (test)',
     'class1_line_test':'',
     'class1_line_train':'',
     'class1_marker_train':'.',
     'class1_marker_test':'x',
     'class1_color_train':'r',
     'class1_color_test':'r',
     'class1_label_train':'Rej VS CC (train)',
     'class1_label_test':'Rej VS CC (test)',
     'class2_marker_train':'.',
     'class2_marker_test':'x',
     'class2_color_train':'b',
     'class2_color_test':'b',
     'class2_label_train':'Rej VS CL (train)',
     'class2_label_test':'Rej VS CL (test)',
     },
     'correlation_MF':{
     'features_id':[(1,0),(1,2),(1,3)],
     'xlabel':'Probability Sale_MF=1',
     'ylabel':'Probability Reject,Sale_CC=1,Sale_CL=1',
     'class0_marker_train':'.',
     'class0_marker_test':'x',
     'class0_color_train':'g',
     'class0_color_test':'g',
     'class0_label_train':'MF VS Rej (train)',
     'class0_label_test':'MF VS Rej (test)',
     'class1_line_test':'',
     'class1_line_train':'',
     'class1_marker_train':'.',
     'class1_marker_test':'x',
     'class1_color_train':'r',
     'class1_color_test':'r',
     'class1_label_train':'MF VS CC (train)',
     'class1_label_test':'MF VS CC (test)',
     'class2_marker_train':'.',
     'class2_marker_test':'x',
     'class2_color_train':'b',
     'class2_color_test':'b',
     'class2_label_train':'MF VS Rej (train)',
     'class2_label_test':'MF VS Rej (test)',
     },
     'correlation_CC':{
     'features_id':[(2,0),(2,1),(2,3)],
     'xlabel':'Probability Sale_CC=1',
     'ylabel':'Probability Reject,Sale_MF=1,Sale_CL=1',
     'class0_marker_train':'.',
     'class0_marker_test':'x',
     'class0_color_train':'g',
     'class0_color_test':'g',
     'class0_label_train':'CC VS Rej (train)',
     'class0_label_test':'CC VS Rej (test)',
     'class1_line_test':'',
     'class1_line_train':'',
     'class1_marker_train':'.',
     'class1_marker_test':'x',
     'class1_color_train':'r',
     'class1_color_test':'r',
     'class1_label_train':'CC VS MF (train)',
     'class1_label_test':'CC VS MF (test)',
     'class2_marker_train':'.',
     'class2_marker_test':'x',
     'class2_color_train':'b',
     'class2_color_test':'b',
     'class2_label_train':'CC VS CL (train)',
     'class2_label_test':'CC VS CL (test)',
     },
     'correlation_CL':{
     'features_id':[(3,0),(3,1),(3,2)],
     'xlabel':'Probability Sale_CL=1',
     'ylabel':'Probability Reject,Sale_MF=1,Sale_CC=1',
     'class0_marker_train':'.',
     'class0_marker_test':'x',
     'class0_color_train':'g',
     'class0_color_test':'g',
     'class0_label_train':'CL VS Rej (train)',
     'class0_label_test':'CL VS Rej (test)',
     'class1_line_test':'',
     'class1_line_train':'',
     'class1_marker_train':'.',
     'class1_marker_test':'x',
     'class1_color_train':'r',
     'class1_color_test':'r',
     'class1_label_train':'CL VS MF (train)',
     'class1_label_test':'CL VS MF (test)',
     'class2_marker_train':'.',
     'class2_marker_test':'x',
     'class2_color_train':'b',
     'class2_color_test':'b',
     'class2_label_train':'CL VS CC (train)',
     'class2_label_test':'CL VS CC (test)',
     },
  }, 

  'model_data_provider':{
        'type':'PandasDataProviderRespondingClientsNoOutliers',
        'remove_all':True,
        'training_set':True,
        'input_file':'data/28_01_2020_1584entries/data_Products_ActBalance_default0.csv'
  }
 }