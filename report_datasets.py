import pandas as pd
import os
from ydata_profiling import ProfileReport


datasets = [pd.read_csv(os.getcwd() + '/datasets/' + file) for file in os.listdir(os.getcwd() + '/datasets') if file.endswith('.csv')]

if not os.path.exists(os.getcwd() + '/reports'):
    os.mkdir(os.getcwd() + '/reports')

for i in range(len(datasets)):
    profile = ProfileReport(datasets[i], title="Profiling Report")
    profile.to_file(os.getcwd() + '/reports/profile_report_' + str(i) + '.html')