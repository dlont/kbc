import os

def PhonyTargets(env = None, **kw):
    if not env: env = DefaultEnvironment()
    for target,action in kw.items():
        env.AlwaysBuild(env.Alias(target, [], action))
        
env = Environment(ENV = os.environ)

# output_dir = 'results/Revenue_MF_1.5pre_wo_outliers_all'
# output_dir = 'results/Revenue_CC_1.5pre_wo_outliers_all'
# output_dir = 'results/Revenue_CL_1.5pre_wo_outliers_all'
output_dir = 'results/Multiclass_XGBoost_1.6pre_wo_outliers_all'
# output_dir = 'results/Multiclass_zoom_XGBoost_1.6pre_wo_outliers_all'
# output_dir = 'results/Multiclass_XGBoost_Binarized_1.6pre_wo_outliers_all'
# output_dir = 'results/Multiclass_RF_1.6pre_wo_outliers_all'
# output_dir = 'results/Multiclass_RF_1.6pre_wo_outliers_all_v1'
# output_dir = 'results/Multiclass_RF_Binarized_1.6pre_wo_outliers_all'
# output_dir = 'results/Multiclass_SVC_1.6pre_wo_outliers_all'
# output_dir = 'results/Multiclass_MLP_1.6pre_wo_outliers_all'


ref_config = 'configs/conf_baseline_noOutliers_Multiclass_RF_cff.py'

# dev_config = 'configs/conf_baseline_Revenue_MF_cff.py'
# dev_config = 'configs/conf_baseline_Revenue_CC_cff.py'
# dev_config = 'configs/conf_baseline_Revenue_CL_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Revenue_MF_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Revenue_CC_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Revenue_CL_cff.py'
# dev_config = 'configs/conf_baseline_noOutliersAll_Revenue_MF_cff.py'
# dev_config = 'configs/conf_baseline_noOutliersAll_Revenue_CC_cff.py'
# dev_config = 'configs/conf_baseline_noOutliersAll_Revenue_CL_cff.py'
dev_config = 'configs/conf_baseline_noOutliers_Multiclass_XGBoost_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Multiclass_XGBoost_Binarized_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Multiclass_RF_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Multiclass_RF_Binarized_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Multiclass_SVC_cff.py'
# dev_config = 'configs/conf_baseline_noOutliers_Multiclass_MLP_cff.py'
# dev_config = 'configs/conf_baseline_plots_Rej_MF_cff.py'
# dev_config = 'configs/conf_baseline_zoom_plots_Rej_MF_cff.py'

configs_compare = ['configs/conf_cff.py','configs/conf_baseline_vanilla_cff.py']
configs_revenue_targets = ['configs/conf_baseline_plots_cff.py',
                           'configs/conf_ridgeCV_Revenue_MF_cff.py',
                           'configs/conf_baseline_Revenue_CC_cff.py','configs/conf_baseline_Revenue_CL_cff.py','configs/conf_baseline_Revenue_MF_cff.py',
                           'configs/conf_baseline_Sale_CC_cff.py','configs/conf_baseline_Sale_CL_cff.py','configs/conf_baseline_Sale_MF_cff.py']
configs_plots = ['configs/conf_baseline_plots_Rej_MF_cff.py',
                 'configs/conf_baseline_plots_Rej_CC_cff.py',
                 'configs/conf_baseline_plots_Rej_CL_cff.py',
                 'configs/conf_baseline_plots_CC_CL_cff.py',
                 'configs/conf_baseline_plots_CC_MF_cff.py',
                 'configs/conf_baseline_plots_MF_CL_cff.py',
                 ]
configs_zoom_plots = ['configs/conf_baseline_zoom_plots_Rej_MF_cff.py',
                      'configs/conf_baseline_zoom_plots_Rej_CC_cff.py',
                      'configs/conf_baseline_zoom_plots_Rej_CL_cff.py',
                      'configs/conf_baseline_zoom_plots_CC_CL_cff.py',
                      'configs/conf_baseline_zoom_plots_CC_MF_cff.py',
                      'configs/conf_baseline_zoom_plots_MF_CL_cff.py',
                      ]
configs = [dev_config]
# configs = configs_plots
# configs = configs_zoom_plots
# configs = configs_plots + configs_zoom_plots + [dev_config]

# kbc_compare = env.Command([output_dir+'features_1d_Inflow_Outflow_p1.png',
#                   output_dir+'features_1d_Inflow_Outflow_p2.png',
#                   output_dir+'features_1d_Products_ActBalance_p1.png',
#                   output_dir+'features_1d_Products_ActBalance_p2.png',
#                   output_dir+'features_1d_Soc_Dem_p1.png',
#                   output_dir+'features_1d_Targets_p1.png',
#                   output_dir+'features_2d_Targets_correlations_p1.png'],
#              configs_compare,
#              "python3 bin/kbc_direct_marketing_mock.py -b -c $SOURCES --dir={0}".format(output_dir))

kbc = env.Command([output_dir+'features_1d_Inflow_Outflow_p1.png',
                  output_dir+'features_1d_Inflow_Outflow_p2.png',
                  output_dir+'features_1d_Products_ActBalance_p1.png',
                  output_dir+'features_1d_Products_ActBalance_p2.png',
                  output_dir+'features_1d_Soc_Dem_p1.png',
                  output_dir+'features_1d_Targets_p1.png',
                  output_dir+'features_2d_Targets_correlations_p1.png'],
             configs,
             "python3 bin/kbc_direct_marketing_mock.py -c $SOURCES --dir={0}".format(output_dir))

PhonyTargets(env, make_output_folder = '-[ ! -d {0} ] && mkdir {0}'.format(output_dir))
PhonyTargets(env, trash_plots = '-rm {0}/features_* {0}/*learning_curve* {0}/*lasso_lars_ic_criterion* {0}/multiclass_classifier_distribution*'.format(output_dir))
PhonyTargets(env, train_test_tables = 'python3 source/train_test_preparation/train_test_datasets.py -c configs/train_test_csv_cff.py')
PhonyTargets(env, train_test_tables_Products_ActBalance_default = 'python source/train_test_preparation/train_test_datasets.py -c configs/train_test_csv_Products_ActBalance_default0_cff.py')

env.Alias('kbc',['make_output_folder',kbc])
env.Alias('runall', ['trash_plots',kbc])
# env.Alias('runcompareall', ['trash_plots',kbc_compare])

PhonyTargets(env, latex_report_clean = '-rm {0}/main.aux {0}/main.fdb_latexmk {0}/main.fls {0}/main.log {0}/main.pdf {0}/main.synctex.gz'.format('report'))