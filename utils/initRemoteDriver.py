#-*- coding: utf-8 -*-
#
# __author__ : jianxing.wei


from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

class RemoteDriver():

    def loadConfData(self,driverConf):
        """

        :param driverConf:
        :return:
        """
        pass



    def __init__(self, driverConf):
        self.loadConfData(driverConf)
        self.remoteURL = ""
        self.browerType = ""
        # self.caps = DesiredCapabilities
        self.caps = {}



    def init_desired_capabilities(self, browerType, browerVersion):

        self.caps['version'] = browerVersion
        self.caps["browserName"] = browerType
        self.caps["platform"] = "ANY"
        self.caps["javascriptEnabled"] = True

        # self.caps["incognito"] = True
        # self.caps["disable-gpu"] = True
        # self.caps["no-sandbox"] = True
        # self.caps["start-maximized"] = True
        # self.caps["window-size"] = "1920,1080"
        self.caps["headless"] = True




    def init_brower_options(self, browerType):
        options = []
        options.append("--headless")
        options.append("--incognito")
        options.append("enable-automation")
        options.append("--disable-gpu")
        options.append("--no-sandbox")
        options.append("--window-size=1920,1080")
        options.append("--start-maximized")


    def getDriver(self, browerType, browerVersion, driverURL):
        self.init_desired_capabilities(browerType, browerVersion)
        from selenium.webdriver.remote.remote_connection import RemoteConnection

        executor = RemoteConnection(driverURL, resolve_ip=False)
        if browerType == "chrome":

            executeDriver = webdriver.Remote(command_executor=executor, desired_capabilities=self.caps)
        elif browerType == "firefox":
            executeDriver = webdriver.Remote(command_executor=executor, desired_capabilities=self.caps)
        else:
            print("brower type not support.")
            executeDriver = None

        return executeDriver

    def diff(self, browerlist):
        pass