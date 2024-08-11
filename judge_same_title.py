import pandas as pd
import numpy as np

# csv 읽어오기
df_military_title = pd.read_csv('news_titles.csv')
df_bigkinds_title = pd.read_csv('bigkinds_news_titles_6066.csv')

#값을 이차원 배열 형식으로 반환한다.
#반환된 타입은 numpy.ndarray
matrix_military_title = df_military_title.values
matrix_naver_title = df_bigkinds_title.values


# 1차원 배열로 변환 후 리스트로 변환
flattened_list_military_title = matrix_military_title.ravel().tolist()
flattened_list_bigkinds_title = matrix_naver_title.ravel().tolist()


count = 0
# print(flattened_list_military_title)
for i in range(len(flattened_list_military_title)) :
    result = (flattened_list_military_title[i] == flattened_list_bigkinds_title[i])
    if result == True : count += 1
    print(result)
print(count)


