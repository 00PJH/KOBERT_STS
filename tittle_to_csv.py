import pymysql
import pandas as pd

# MySQL 연결
db = pymysql.connect(
    host="localhost",
    user="root",
    password="1514",
    database="news_db",
    charset="utf8mb4")

# title 개수
sql = '''SELECT title 
FROM news

'''

# Connection 객체로부터 cursor() 메서드를 호출하여 Cursor 객체를 가져옴
cursor = db.cursor(pymysql.cursors.DictCursor)

# 실행하기
cursor.execute(sql)

# 모든 데이터를 가져온다.
datas = cursor.fetchall() 

# 사용했던 데이터베이스 관련 자원들을 닫아준다.
cursor.close()
db.close()

# 결과를 pandas 데이터프레임으로 변환
df = pd.DataFrame(datas)

# label 0으로 지정
df['label'] = 1

df.to_csv('./news_titles_label1.csv', index = False)
