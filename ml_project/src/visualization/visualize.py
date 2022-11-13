from pandas_profiling import ProfileReport
import pandas as pd


df = pd.read_csv("data/raw/heart_cleveland_upload.csv")
profile = ProfileReport(df, title='EDA_Report', explorative=True)
profile.to_file("reports/EDA.html")
