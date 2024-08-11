import pandas as pd

# 파일 경로
file_path = 'bigkinds_news.csv'

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 제목 열만 추출
df_titles_only = df[['제목']]

# 새로운 CSV 파일로 저장
output_path = 'bigkinds_news_only_title.csv'
df_titles_only.to_csv(output_path, index=False, encoding='utf-8-sig')

output_path
