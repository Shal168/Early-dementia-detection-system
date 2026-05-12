import pandas as pd
import os

RESULT_FILE = "results/dementia_results.xlsx"

def save_results(data):

    os.makedirs("results", exist_ok=True)
    if os.path.exists(RESULT_FILE):
        df_existing = pd.read_excel(RESULT_FILE)
    else:
        df_existing = pd.DataFrame()

    df_new = pd.DataFrame([data])

    df = pd.concat([df_existing, df_new], ignore_index=True)

    df.to_excel(RESULT_FILE, index=False)

    return RESULT_FILE