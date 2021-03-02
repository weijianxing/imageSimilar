#-*- coding: utf-8 -*-
# 
# __author__ : jianxing.wei
import unittest

from utils.loadConf import load_file


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        self.fileName = "remoteDriverConf.yaml"
        global conf
        # print(max(12,6))
        conf = load_file(self.fileName)
    def testLoadValue(self):
        assert conf["remoteDriverURL"] == "http://10.2.20.47:4444/wd/hub"
        assert conf["PC"][0]["browerType"] ==  "chrome"
        assert conf["PC"][0]["version"] == "v84.20"
        assert conf["PC"][1]["browerType"]== "firefox"
        assert conf["PC"][1]["version"] == ""

if __name__ == '__main__':
    unittest.main()