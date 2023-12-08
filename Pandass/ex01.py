import requests
from bs4 import BeautifulSoup
import csv

file = open('시가총액.csv','w', encoding='utf-8', newline='')
writer = csv.writer(file)
title = 'N	종목명	현재가	전일비	등락률	액면가	시가총액	상장주식수	외국인비율	거래량	PERROE	토론실'.split('\t')
writer.writerow(title)

def fn_data(soup):
    data = soup.find('table', attrs={'class' : 'type_2'}).find('tbody')
    data_rows = data.find_all('tr')
    for row in data_rows:
        columns = row.find_all('td')
        if len(columns) <= 1 :
            continue
        data = [col.get_text().strip() for col in columns]
        print(data)
        writer.writerow(data)
        
for page in range(1, 5):
    url = 'https://finance.naver.com/sise/sise_market_sum.nhn?sosok=0&page={}'.format(page)
    rse = requests.get(url)
    soup = BeautifulSoup(rse.text, 'lxml')
    fn_data(soup)
    
