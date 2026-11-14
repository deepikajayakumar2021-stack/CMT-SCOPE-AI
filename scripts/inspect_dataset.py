import os
import pandas as pd

ROOT = r"D:\CMT_PROJECT"

for root, dirs, files in os.walk(ROOT):

    for file in files:

        if file.lower().endswith((".csv", ".xlsx", ".parquet")):

            path = os.path.join(root, file)

            print("\n" + "=" * 80)
            print("FILE:", path)
            print("=" * 80)

            try:

                if file.lower().endswith(".csv"):

                    df = pd.read_csv(
                        path,
                        nrows=10
                    )

                elif file.lower().endswith(".xlsx"):

                    df = pd.read_excel(
                        path,
                        nrows=10
                    )

                elif file.lower().endswith(".parquet"):

                    df = pd.read_parquet(
                        path
                    ).head(10)

                print("Columns:")
                print(df.columns.tolist())

                print("\nFirst rows:")
                print(df.head(3).to_string(index=False))

                print("\nShape sample:")
                print(df.shape)

            except Exception as e:

                print("ERROR:", e)