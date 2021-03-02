import time

__author__ = 'jianxing.wei '

class PageSnapshot():
    def __init__(self,driver,dir):
        """

        :param driver: 初始化一个浏览器driver
        :param dir: 截图保存路径
        """
        self.driver = driver
        self.dir = dir

    def shapshot(self,url,filename):
        """

        :param url: 生成快照页面地址
        :param filename: 生成的快照名称
        :return:
        """
        #todo the full page.
        self.driver.get(url)
        self.driver.get_screenshot_as_file(filename)


    def shapshot_full_image(self, url, filename):

        self.driver.get(url)
        time.sleep(1)
        # 接下来是全屏的关键，用js获取页面的宽高，如果有其他需要用js的部分也可以用这个方法
        width = self.driver.execute_script("return document.documentElement.scrollWidth")
        height = self.driver.execute_script("return document.documentElement.scrollHeight")
        # print(width, height)
        # 将浏览器的宽高设置成刚刚获取的宽高
        self.driver.set_window_size(width, height)
        time.sleep(1)
        # 截图并关掉浏览器
        self.driver.get_screenshot_as_file(filename)
    def shapshot_full_image(self, url, filename, maxWidth, maxHeight):

        self.driver.get(url)
        time.sleep(1)
        # 接下来是全屏的关键，用js获取页面的宽高，如果有其他需要用js的部分也可以用这个方法
        width = self.driver.execute_script("return document.documentElement.scrollWidth")
        height = self.driver.execute_script("return document.documentElement.scrollHeight")
        # print(width, height)
        # 将浏览器的宽高设置成刚刚获取的宽高
        if width< maxWidth:
            width = maxWidth
        if height< maxHeight:
            height = maxHeight
        self.driver.set_window_size(width, height)

        time.sleep(1)
        # 截图并关掉浏览器
        self.driver.get_screenshot_as_file(filename)

    def shapshot_fixsize(self, url, filename, width, height):

        self.driver.get(url)
        time.sleep(1)
        # 接下来是全屏的关键，用js获取页面的宽高，如果有其他需要用js的部分也可以用这个方法
        # width = self.driver.execute_script("return document.documentElement.scrollWidth")
        # height = self.driver.execute_script("return document.documentElement.scrollHeight")
        # print(width, height)
        # 将浏览器的宽高设置成刚刚获取的宽高
        self.driver.set_window_size(width, height)
        time.sleep(1)
        # 截图并关掉浏览器
        self.driver.get_screenshot_as_file(filename)