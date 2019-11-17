import pandas as pb
import re
from pyvi import ViTokenizer
import nltk
import os
import time


class TextProcessing():
    """
        class dùng để xử lý text
    """

    def __init__(self):
        """
        """
        pass

    def load_file(self, input_path):
        """
        load file với tham số đầu vào lưu dưới dạng text
        :param input_path: đầu vào là file_path
        :return: không trả về
        """
        with open(input_path, 'r', encoding="utf-8") as f:
            text_file = f.read()
        return text_file

    def _text_to_sentences(self, text):
        """
        tách dữ liệu text trong file thành từng câus
        :param text: đầu vào là một text
        :return: list sentence
        """
        text = text.replace('_', ' ')
        sens = re.split('\.|\,|\-|\n', text)
        return list(sens)

    def _word_tokenize(self, sens):
        """
        đánh từ vựng cho text tiếng việt
        :param sens: đầu vào sens chưa đc đánh từ vừng
        :return: text đã được đánh
        """
        for i, sen in enumerate(sens):
            sen = ViTokenizer.tokenize(sen)
            sens[i] = sen.lower()
            
        return sens

    def _clean_stopwords(self, text, file_stopwords='./data/stop_words/vnstopword.txt'):
        #load stop_words
        with open(file_stopwords, 'r', encoding="utf-8") as f:
            t = f.read()
        stop_words = {word.strip() for word in t.split('\n')}

        for stop_word in stop_words:
            if len(stop_word.split(' ')) > 1:
                text = text.replace(stop_word, '')

        words_text = text.split(' ')
        while "" in words_text:
            words_text.remove("")

        for stop_word in stop_words:
            while stop_word in words_text:
                words_text.remove(stop_word)

        text = ' '.join(words_text)
        return text

    def _clean_Text(self, text):
        text = re.sub("(\W)+", " ", text)
        text = re.sub("(\W)+", " ", text)
        text = re.sub('\s+', ' ', text)
        text = text.strip()
        return text

    def dir_preProcessing(self, dir_path='./data/samples/raw/data', output_path='./data/samples/training/data'):
        # tạo thư mục lưu file
        os.makedirs(output_path, exist_ok=True)
        # load list file
        listdir = os.listdir(dir_path)
        for dirt in listdir:
            dirt_out_topic =  output_path + '/' + dirt+'/'
            os.makedirs(dirt_out_topic,exist_ok= True)
            # load file
            files = os.listdir(dir_path+'/'+dirt)
            for i, file in enumerate(files):
                print('processing : ', i, '/', len(files) , ' | ' , dirt,end ='\r')
                time.sleep(0.0001)
                text = self.load_file(dir_path + '/'+dirt+'/' + file)
                text = self.pre_processing(text)
                # ghi file đã xử lý
                with open(dirt_out_topic + '/' + file, 'w', encoding="utf-8") as f:
                    f.write(text)


    def pre_processing(self, text='', tokenize=True, covert_acronyms=True, reurn=True, output_path=None, stop_words=True):
        sens = self._text_to_sentences(text)
        #tokenize
        if tokenize is True:
            sens = self._word_tokenize(sens)
        for i, sen in enumerate(sens):
            # xoa stop_words
            if stop_words is True:
                sen = self._clean_stopwords(sen.strip())
            # clean_Text
            sens[i] = self._clean_Text(sen.strip())

        #xoa nhung dong khong co tu
        sens = list(filter(lambda x: x != '', sens))
        text = '\n'.join(sens)
        # sau khi kết thúc xử lý thì return hoặc ghi ra file
        if output_path is not None:
            with open(output_path, 'w', encoding="utf-8") as f:
                f.write(text)
        if reurn is True:
            return text

def test():
    a = TextProcessing()
    a.dir_preProcessing()

if __name__ == '__main__':
    test()
