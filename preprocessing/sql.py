import psycopg2
import pandas as pd

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--tab', type=str)
args = parser.parse_args()

HOST = ""
PORT = ""
DBNAME = ""
USER = ""
PASSWORD = ""


query = f"SELECT * FROM jmk_251020.{args.tab};"

# psycopg2 연결
conn = psycopg2.connect(
    host=HOST,
    port=PORT,
    dbname=DBNAME,
    user=USER,
    password=PASSWORD
)

# pandas로 바로 DataFrame 변환
df = pd.read_sql(query, conn)

# CSV로 저장
df.to_csv(f"{args.tab}.csv", index=False)

conn.close()

print(f"✅ CSV 파일 저장 완료: {args.tab}.csv")
