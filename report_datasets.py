import kagglehub
import shutil
import os
import pandas as pd
from ydata_profiling import ProfileReport



if not os.path.exists(os.getcwd() + '/datasets'):
    os.mkdir('datasets')

dataset_urls = ['anthonytherrien/depression-dataset', 'divaniazzahra/mental-health-dataset', 'mdismielhossenabir/psychosocial-dimensions-of-student-life']

for url in dataset_urls:
    path = kagglehub.dataset_download(url)
    path = path + '/' + os.listdir(path)[0]
    shutil.move(path, os.getcwd() + '/datasets/' + os.path.basename(path))

datasets = [pd.read_csv(os.getcwd() + '/datasets/' + file) for file in os.listdir(os.getcwd() + '/datasets')]

if not os.path.exists(os.getcwd() + '/reports'):
    os.mkdir(os.getcwd() + '/reports')

for i in range(len(datasets)):
    profile = ProfileReport(datasets[i], title="Profiling Report")
    profile.to_file(os.getcwd() + '/reports/profile_report_' + str(i) + '.html')