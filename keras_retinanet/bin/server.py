import uuid
from predict_core import detect_image

uuid.uuid4()

import tornado.web
import tornado.ioloop
import tornado.httpserver
import os

SERVER_TMP = './server_tmp'

class ScratchDetectorHandler(tornado.web.RequestHandler):
    def set_default_header(self):
    # 后面的*可以换成ip地址，意为允许访问的地址
        self.set_header('Access-Control-Allow-Origin', '*') 
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')
    #tornado.httputil.HTTPFile对象三个属性
    #1.filename文件名
    #2.body文件内部实际内容
    #3.type文件的类型
    def get(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*') 
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')
        image_name = self.get_argument('image')
        ext = image_name.split('.')[-1]
        if ext == 'jpg':
            ext = 'jpeg'
        self.set_header('content-type', 'image/'+ext)
        image_path = os.path.join(SERVER_TMP, image_name)
        with open(image_path, 'rb') as f:
            self.write(f.read())
        return


    async def post(self, *args, **kwargs):
        self.set_header('Access-Control-Allow-Origin', '*') 
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')
        #查看上传文件的完整格式，files以字典形式返回
        #print(self.request.files)
        #{'file1':
        #[{'filename': '新建文本文档.txt', 'body': b'61 60 -83\r\n-445 64 -259', 'content_type': 'text/plain'}],
        #'file2':
        filesDict=self.request.files
        # for inputname in filesDict:
        #     #第一层循环取出最外层信息，即input标签传回的name值
        #     #用过filename键值对对应，取出对应的上传文件的真实属性
        #     http_file=filesDict[inputname]
        #     for fileObj in http_file:
        #         #第二层循环取出完整的对象
        #         #取得当前路径下的upfiles文件夹+上fileObj.filename属性(即真实文件名)
        #         filePath=os.path.join(os.path.dirname(__file__),fileObj.filename)
        #         with open(filePath,'wb') as f:
        #              f.write(fileObj.body)
        file_uuid = uuid.uuid4()
        ext = filesDict['image'][0]['filename'].split('.')[-1]

        acceptable_ext = ['jpg', 'jpeg', 'png']
        if ext not in acceptable_ext:
            self.write('格式不正确')
            return

        file_name = str(file_uuid) + '.' +ext
        file_path = os.path.join(SERVER_TMP, file_name)
        with open(file_path,'wb') as f:
            f.write(filesDict['image'][0]['body'])
        marked_path = await detect_image(file_path)
        self.write(marked_path)

    def option(self):
        self.set_header('Access-Control-Allow-Origin', '*') 
        self.set_header('Access-Control-Allow-Headers', 'x-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, PUT, DELETE')
        

if __name__ == '__main__':
    app=tornado.web.Application(
        [(r'/detect',ScratchDetectorHandler)])
    httpserver=tornado.httpserver.HTTPServer(app)
    httpserver.bind(4000)
    httpserver.start()
    tornado.ioloop.IOLoop.instance().start()