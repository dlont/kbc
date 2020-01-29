# Prerequisites

**Software**
+ Python 2.7
+ Scons 3.1.2
In order to install Scons
1. download the latest version from https://scons.org/pages/download.html to the root folder of the project
2. untar and cd to the Scons source folder
3. issue from the command line `python setup.py install --prefix=/home/iihe/Desktop/JOB_search/non_academy/KBC_data_science/KBC_test_task/scons-3.1.2-install`

**Python packages**
+ scikit-learn
+ numpy
+ pandas
+ logging
+ argparse
+ textwrap

# Usage
Old style input data tables join, without imputation
```bash
./scons-3.1.2-install/bin/scons train_test_tables
```

New style with imputation
```bash
./scons-3.1.2-install/bin/scons train_test_tables_Products_ActBalance_default
```

Plotting and modelling
```bash
./scons-3.1.2-install/bin/scons runall
```

Clean plots in results folder
```bash
./scons-3.1.2-install/bin/scons trash_plots
```
