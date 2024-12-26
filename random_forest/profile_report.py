from ydata_profiling import ProfileReport
import pandas as pd

# Load data
data = pd.read_csv("datasets/train.csv")

# Generate the report
profile = ProfileReport(data, title="Depression Report", explorative=True)
profile.to_file("reports/depression_report.html")