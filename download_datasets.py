import kagglehub
import shutil
import os

if not os.path.exists(os.getcwd() + '/datasets'):
    os.mkdir('datasets')

dataset_urls = ['hamjashaikh/mental-health-detection-dataset']
# unuse 'anthonytherrien/depression-dataset', 'divaniazzahra/mental-health-dataset', 'mdismielhossenabir/psychosocial-dimensions-of-student-life'

for url in dataset_urls:
    path = kagglehub.dataset_download(url)
    path = path + '/' + os.listdir(path)[0]
    shutil.move(path, os.getcwd() + '/datasets/' + os.path.basename(path))   