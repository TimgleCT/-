import time
from selenium import webdriver 
from lxml import etree
import sys

print("本程式可統計Dcard近期大家喜歡的關注的版!")
PageStart = 1000
PageEnd = int(input("請設定搜尋長度(1000以上)："))
PageJump = 1000

#載入chromedriver，將可自動控制chrome模擬人們瀏覽行為
StartLoading = int(round(time.time() * 1000))
browser = webdriver.Chrome('./chromedriver')
#讓瀏覽器前往目標網站，並爬出其網站內容
browser.get('https://www.dcard.tw/f?latest=false')
browser.set_window_position(-3000, 0)
print("正在努力搜尋熱門版的文章中...")

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

StartTime = int(round(time.time() * 1000))

#用lxml讀取Html檔，並找出放置作者名稱的地方
#其放置在class名稱叫PostAuthor_root_3vAJfe的span裡
html = etree.HTML(content)
articleBoard = html.xpath("//*[@id='root']/div/div[1]/div/div/main/div/div/div/a/header/div/div/div[2]/span")
articleDate = html.xpath("//*[@id='root']/div/div[1]/div/div/main/div/div/div/a/header/span/span")
#將作者名稱取出，並切割學校與系的字串，只保留校名
#若字串裡有"大學"或是"學院"的話，放入作者陣列中
countArticle = 0
authorpoint = 0
author = []
for i in range(len(articleBoard)):
    countArticle += 1
    school = articleBoard[i].xpath('text()') 
    getDate = articleDate[i].xpath('text()')
    author.append([])
    author[authorpoint].append(getDate[0])
    author[authorpoint].append(school[0])
    authorpoint += 1

#宣告一個數字陣列存各個大學出現了幾次(也就是有幾篇文章)
authorpoint = 0

for authorItem in author:
    count = 0
    for authorCounting in author:
        if authorItem[1] == authorCounting[1]:
            count = count + 1
    author[authorpoint].append(count)
    authorpoint += 1

print("------------------------------")
print("在"+author[0][0]+"到"+author[int(len(author)-1)][0]+"之間，總共有"+str(countArticle)+"篇文章")
print("------------------------------")
# 氣泡排序法
for i  in range(0,len(author)-1):
    for j in range(0,len(author)-1-i):
        if author[j][2] < author[j+1][2]: #是否需交換
                tmp = author[j]
                author[j] = author[j+1]
                author[j+1] = tmp
listpoint = 0
for authorItem in author:
#比對前面有沒有顯示過，以避免重複顯示
    printstatus = True 
    for check in range(listpoint):
        if authorItem[1] == author[check][1]:
            printstatus = False
            
    listpoint = listpoint + 1
#如果過去沒有顯示過，就顯示
    if printstatus:
        print(authorItem[1]+"版有: "+str(authorItem[2])+"篇文章")
print("------------------------------")

FinishTime = int(round(time.time() * 1000))
print("統計與排序執行時間："+str((FinishTime - StartTime)/1000)+"秒")