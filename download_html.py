import urllib.request
import requests
from bs4 import BeautifulSoup
import os
import json
import time

class DownloadContent:
    """
        class gồm các chức năng dành mục tải nội dung từ các trang web
    """
    def __init__(self):
        """
            Phương thức khởi tạo không có tham số
        """
        self.output_path = './data/samples/raw' 

    def download_content_web (self,url_path, output_path = None):
        try :
            res = requests.get(url_path)
            soup = BeautifulSoup(res.text, "lxml")
            print (url_path,end ='\r')
            time.sleep(0.0001)

            title = soup.find('h1').text.strip() #tieu de
            summary = soup.find('h2').text.splitlines()[1].strip() #tom tat
            #xoa title hinh ảnh , video xuất hiện trong nội dung
            sb = soup.find(id='divNewsContent')
            [s.extract() for s in sb('figure')]
            contents = '\n'.join(sub.text for sub in sb.find_all('p') if "text-align:right" not in str(sub))  # noi dung
            text = title + '\n \n' + summary + '\n \n' + contents

            if output_path is None:
                print(text)
                return text
            else:
                with open(output_path, 'w', encoding="utf-8") as f:
                    f.write(text)

        except:
            print('lỗi tải trang')

    def _download_newspaper_new(self):
        topics = ['phap-luat','giao-duc-khuyen-hoc','the-thao','van-hoa','suc-khoe','kinh-doanh']
        for topic in topics:
            self.__update_data_topic(topic)

    def __update_data_topic(self,name_topic,page = 2): 
        ouput_path = self.output_path
        data_path = ouput_path + '/data/' + name_topic + '/'
        os.makedirs(data_path,exist_ok= True)

        path = 'https://dantri.com.vn'
        if os.path.exists(ouput_path + '/' + 'id_newspaper.json') :
            with open(ouput_path + '/' + 'id_newspaper.json', encoding='utf-8') as f:
                id_title = json.load(f)
        else:
            id_title = {} # {id:tile}
			
        for i in range(1,page):
            res = requests.get('https://dantri.com.vn' + '/'+name_topic+'/'+'trang-'+str(i)+'.htm')
            soup = BeautifulSoup(res.text, "lxml")
			
            #lấy bài báo mới nhất
            p = soup.find(class_='mt3 clearfix')
            print('\n'+"page : " +str(i)+" | " + name_topic)
            if str(p.h2.a['data-id'] ) not in  id_title:
                id_title.update({p.h2.a['data-id']:str(p.a['href']).split('-')[0]})
                self.download_content_web(path+p.a['href'],data_path+p.h2.a['data-id']+'.txt')
				
            #lấy những bài báo còn lại
            for s in soup.find_all(class_='mt3 clearfix eplcheck'):
                if str (s.div['data-id'] ) not in id_title:
                    id_title.update({s.div['data-id']:str(s.a['href']).split('/')[0]})
                    self.download_content_web(path+s.a['href'],data_path+s.div['data-id']+'.txt')
            #save
            with open(ouput_path+'/'+'id_newspaper.json', 'w', encoding='utf8') as json_file:
                json.dump(id_title, json_file, ensure_ascii=False)

    def download_newspaper(self,output_path = None):
        if output_path is not None:
            self.output_path = output_path
        os.makedirs(self.output_path+'/data',exist_ok= True)
        self._download_newspaper_new()
		
def Test():
    dw = DownloadContent()
    dw.download_newspaper()
	
if __name__ == '__main__':
    Test()
