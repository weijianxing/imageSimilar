#-*- coding: utf-8 -*-
# 
# __author__ : jianxing.wei
import socket

from utils.generateImages import PageSnapshot
from utils.initRemoteDriver import RemoteDriver


def connectDriver(remoteURL, caps):
    pass

def testSnapshot():
    remoteurl = "http://selenium-hub.wuage-inc.com/wd/hub"
    # ip = socket.gethostbyname("selenium-hub.wuage-inc.com")
    # print(ip)
    rurl = "http://10.2.20.48:4444/wd/hub"
    browerType = "firefox"
    browerVersion = ""
    testURL = "https://www.wuage.com/xianhuo?psa=W1.a211.c1.23"
    snapname = "test_firefox.png"
    ord = RemoteDriver("")
    remotedriver = ord.getDriver(browerType,browerVersion,rurl)
    pss = PageSnapshot(remotedriver,"")
    # pss.shapshot_full_image(testURL, snapname)
    pss.shapshot_fixsize(testURL, snapname, 1920,1080)


    remotedriver.quit()

if __name__ == '__main__':
    testSnapshot()
