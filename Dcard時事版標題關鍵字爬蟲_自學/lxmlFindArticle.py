import time
import sys
from selenium import webdriver 
from lxml import etree

print("請問你要在Dcard時事版搜尋什麼內容?")
searchstr1 = input("搜尋條件1：")
searchstr2 = input("搜尋條件2：")
PageStart = 1000
PageEnd = int(input("設定搜尋長度(1000以上)："))
PageJump = 1000

#載入chromedriver，將可自動控制chrome模擬人們瀏覽行為
StartLoading = int(round(time.time() * 1000))
browser = webdriver.Chrome('./chromedriver')
#讓瀏覽器前往目標網站，並爬出其網站內容
browser.get('https://www.dcard.tw/f/trending?latest=true')
browser.set_window_position(-3000, 0)
print("正在努力搜尋中...")

#模擬人們瀏覽網站的行為，模擬下滑網站，每次下滑2000，
#因載入網站需要一點時間，給予0.5秒的等待時間
#當下滑達到100000時，停止瀏覽。並將網站Html內容讀取出來
SCROLL_PAUSE_TIME = 0.5
for i in range(PageStart,PageEnd + PageJump,PageJump):
    #每次螢幕下滑1000
    browser.execute_script("window.scrollTo(0,"+str(i)+");")
    sys.stdout.write('\r')
    sys.stdout.write("[%s] %d%%" % ('='*int(i/PageEnd*100),int(i/PageEnd*100)))
    sys.stdout.flush()
    #等待載入
    time.sleep(SCROLL_PAUSE_TIME)

content = browser.page_source
browser.quit()
FinishLoading = int(round(time.time() * 1000))
LoadingTime = (FinishLoading - StartLoading)/1000
print("")
if (LoadingTime / 60) >= 1:
    print("網頁內容載入完畢!共花費了"+str(int(LoadingTime / 60))+"分"+str(LoadingTime % 60)+"秒")
else:
    print("網頁內容載入完畢!共花費了"+str((FinishLoading - StartLoading)/1000)+"秒")


#用lxml讀取Html檔，並找出放置作者名稱的地方
#其放置在class名稱叫PostAuthor_root_3vAJfe的span裡
html = etree.HTML(content)
articleTitle = html.xpath("//h3[contains(@class, 'Title__Text-v196i6-0 gmfDU')]")
articleURL = html.xpath("//a[contains(@class, 'PostEntry_root_V6g0rd')]")
articleDate = html.xpath("//*[@id='root']/div/div[1]/div/div/main/div/div/div/a/div/div[1]/header/span/span")


countArticle = 0
authorpoint = 0
title = []

for i in range(len(articleTitle)):
    countArticle += 1
    getTitle = articleTitle[i].xpath('text()')
    getURL = articleURL[i].xpath('@href')
    getDate = articleDate[i].xpath('text()')
    if getTitle[0].find(searchstr1) != -1 or getTitle[0].find(searchstr2) != -1:
        title.append([])
        title[authorpoint].append(getDate[0])
        title[authorpoint].append(getTitle[0])
        title[authorpoint].append("https://www.dcard.tw"+getURL[0])
        authorpoint += 1

print("------------------------------")
print("在"+title[0][0]+"到"+title[int(len(title)-1)][0]+"之間，總共有"+str(countArticle)+"篇文章")
print("其中找出"+str(authorpoint)+"篇文章符合'"+searchstr1+"'或'"+searchstr2+"'的條件")
print("------------------------------")
for a in title:
    print(a)
print("------------------------------")


