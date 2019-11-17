import os
import pandas as pd
import numpy as np
from text_processing import TextProcessing as tp
import re

class ExtractText:
    """
        class thực hiện các chức năng tính toán, thống kê 
        trich xuat du dieu ,các từ xuất hiện trong văn bản
    """

    def __init__(self):
        pass

    @staticmethod
    def __load_file(file_path,lower = True):
        """
        :param file_path:  file path
        :return:  text
        """
        with open(file_path, 'r', encoding="utf-8") as f:
            if lower == True:
                text = f.read().lower()
            else:
                text = f.read()
        return text

    def __count_word_in_file(self, file_path):
        """
            Đếm các từ trong văn bản file
        """
        text = self.__load_file(file_path)
        count = self.__count_words_in_text(text)
        return count

    def __load_texts_dir(self, dir_path):
        """
           Load file 
        """
        dirs_name = os.listdir(dir_path)
        texts = []
        files = []
        for dir_name in dirs_name:
            files_name = os.listdir(dir_path+'/'+dir_name)
            for file in files_name:
                files.append(file)
                texts.append(self.__load_file(dir_path+'/'+dir_name+'/'+file))

        # print(dirs_name)
        return texts, files

    @staticmethod
    def __count_in_bags(wordDict, bow):
        """
            Đếm các từ trong văn bản text
        """
        for word in bow:
            try:
                wordDict[word] += 1
            except:
                pass
        return wordDict

    def count_words(self, dir_path='./data/samples/training/data', output_path='./data/pre', tf_idf=True):
        """
            thống kê các từ xuất hiện trong từng file text
            của tất cả các file trong thư mục
        """
        # tạo thư mụcs
        os.makedirs(output_path, exist_ok=True)
        # load text của tất cả các file trong thư mục
        texts, names = self.__load_texts_dir(dir_path)
        # tao dictionary
        word_dict, wordDicts, bows = self.__word_dict_and_bow(texts)
        for i, text in enumerate(texts):
            wordDicts[i] = self.__count_in_bags(wordDicts[i], bows[i])
        # tinh tf
        tf_bows = []
        for i, worddict in enumerate(wordDicts):
            tf_bows.append(self.__compute_TF(worddict, bows[i]))
        #tinh idf
        idfs = self.__compute_IDF(wordDicts)
        #tinh tf-idf
        tf_idf_bows = []
        for i, tf_bow in enumerate(tf_bows):
                tf_idf_bows.append(self.__compute_TFIDF(tf_bow, idfs))

        if tf_idf is True:
            self.__save_file(tf_idf_bows=tf_idf_bows,word_dict=word_dict,names=names, output_path=output_path)
        else:
            self.__save_file(word_dict=word_dict, output_path=output_path)

    @staticmethod
    def __word_dict_and_bow(texts):
        import re
        word_dict = set()
        wordDicts = []
        bows = []
        for text in texts:
            bow = re.split(' ' '|\n', text)
            word_dict = word_dict.union(set(bow))
            bows.append(bow)
        for i in range(len(texts)):
            wordDicts.append(dict.fromkeys(word_dict, 0))

        return word_dict, wordDicts, bows

    @staticmethod
    def __compute_TF(word_dict, bow):
        tf_dict = {}
        bow_count = len(bow)
        for word, count in word_dict.items():
            tf_dict[word] = count/float(bow_count)
        return tf_dict

    @staticmethod
    def __compute_IDF(doc_list):
        import math
        idf_dict = {}
        N = len(doc_list)
        idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
        for doc in doc_list:
            for word, count in doc.items():
                if count > 0:
                    idf_dict[word] += 1

        for word, count in idf_dict.items():
            idf_dict[word] = math.log(N/float(count))

        return idf_dict

    @staticmethod
    def __compute_TFIDF(tf_bow, idfs):
        tfidf = {}
        for word, val in tf_bow.items():
            tfidf[word] = val*idfs[word]
        return tfidf

    @staticmethod
    def __save_file(tf_idf_bows=None, word_dict=None, names=None, output_path=''):
        if tf_idf_bows is not None and names is not None:
            # tao dataframe
            df = pd.DataFrame(tf_idf_bows, index=names)
            df.to_csv(output_path + '/tf_idf.csv', header=True,encoding='utf-8', index=True)
        # ghi words
        with open(output_path + '/Dictionars', 'w', encoding="utf-8") as f:
            for word in word_dict:
                f.write(word+'\n')

    @staticmethod
    def __make_Dictionary(files,most_dictionary = None ,save = True,):
        from collections import Counter
        all_words = []
        for f in files:
            with open(f, encoding="utf-8") as m:
                for line in m:
                    words = line.split()
                    all_words += words
        dictionary = (Counter(all_words))
        if most_dictionary is not None:
            words = [d[0] for i,d in enumerate(dictionary.most_common(most_dictionary))]
        else:
            words = dictionary
        if save == True:
            with open('./data/pre' + '/Dictionars', 'w', encoding="utf-8") as f:
                for word in words:
                    f.write(word+'\n')
        return words

    def __word_bow(self,files,words):
        wordDicts = []
        bows = []
        texts = [ self.__load_file(fil) for fil in files]
        for text in texts:
            text = text.strip()
            bow = text.split()
            bows.append(bow)

        for i in range(0,len(texts)):
            wordDicts.append(dict.fromkeys(words, 0))
        return wordDicts,bows

    def extract_features(self,dir_path='./data/samples/training/data',most_dictionary = None):
        """
            extract features file in dir 
        """
        dirs_name = os.listdir(dir_path)
        # print(dirs_name)
        files = []
        # features_matrix
        train_labels = []
        train_file_names = []
		
        for dir_name in dirs_name:
            files_name = os.listdir(dir_path+'/'+dir_name+'/')
            for file_name in files_name:
                files.append((dir_path+'/'+dir_name+'/'+file_name))
                train_labels.append(dir_name)
                train_file_names.append(file_name)
        # print(train_file_names)
        if most_dictionary is None:
            dictionary = self.__make_Dictionary(files)
        else:
            dictionary = self.__make_Dictionary(files,most_dictionary)
        wordDicts, bows = self.__word_bow(files,dictionary)
        for i , wd in enumerate(wordDicts):
            wordDicts[i] = self.__count_in_bags(wd,bows[i])
        
        dataMatrix = np.array([[ wd[word] for word in dictionary] for wd in wordDicts])
        return  dataMatrix , train_labels, train_file_names 
    
    def text2matrix(self,text,voca_path = './data/pre/Dictionars'):
        text = tp().pre_processing(text)
        with open(voca_path, encoding='utf-8') as fh:
            text_file = fh.read()
        voca = text_file.strip().split('\n')
        text = text.strip()
        bow = text.split()
        matrix= (dict.fromkeys(voca, 0))
        matrix = self.__count_in_bags(matrix, bow)
        dataMatrix = np.array([ matrix[word] for word in voca])
        return dataMatrix

def load_file(file_path,lower = True):
    """
    :param file_path:  file path
    :return:  text
    """
    with open(file_path, 'r',encoding="utf-8") as f:
        if lower == True:
            text = f.read().lower()
        else:
            text = f.read()
    return text

def test():
    a = ExtractText()
    a.count_words(tf_idf=True)
    a.extract_features()

if __name__ == '__main__':
    test()
    
