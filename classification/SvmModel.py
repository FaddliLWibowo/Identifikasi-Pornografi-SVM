#import Library
import re
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm

class SvmModel():
    #constructor
    def __init__(self, datasetPath):
        self.classifier = None
        self.tf_vect = None

        dataset = pd.read_csv(datasetPath)

        self.label = dataset.label      #set kolom label pd dataset ke dalam variabel
        
        self.dataset = dataset.drop("label", axis = 1)      #drop kolom label pada dataset

    #fungsi untuk train_test_split pada scikit-learn
    def variabel(self):
        #split dataset, 6/10 menjadi data latih, 4/10 menjadi data uji
        cerita_train, cerita_test, label_train, label_test = train_test_split(self.dataset["text"], self.label, test_size = 0.4, random_state = 42)
        return cerita_train, cerita_test, label_train, label_test

    #fungsi untuk vectorizing split_data with TFIDF
    def split_data(self):
        self.tf_vect = TfidfVectorizer()
        self.tf_vect.fit(self.dataset['text'])
        cerita_train, cerita_test, label_train, label_test = self.variabel()

        #ubah dataset menjadi vector
        cerita_train_tf = self.tf_vect.fit_transform(cerita_train)
        cerita_test_tf = self.tf_vect.transform(cerita_test)
        return cerita_train_tf, cerita_test_tf, label_train, label_test

    #fungsi untuk vectorizing data uji
    def vectorizer_data_test(self, datatestPath):
        return self.tf_vect.transform(datatestPath)
    
    #fungsi untuk klasifikasi svm
    def svm(self, cerita_train_tf, label_train):
        self.classifier = svm.SVC(C=1, kernel = 'linear')
        self.classifier.fit(cerita_train_tf, label_train)
    
    #fungsi untuk prediksi data uji
    def predict(self, svm_test):
        return self.classifier.predict(svm_test)

    #fungsi untuk skor akurasi
    def akurasi(self, y_asli, y_output):
        return metrics.accuracy_score(y_asli, y_output)

    #fungsi untuk nampilin klasifikasi report (precission, recall, f1-score, support)
    def classi_report(self, y_test, y_output):
        return classification_report(y_test, y_output)

    #fungsi untuk preprocessing data testing
    def read_split(self, datatestPath):
        with open(datatestPath, "r",encoding="utf8") as file:
            test_file = file.read()
            file.close()

        #casefolding menjadi huruf kecil
        text_low = test_file.lower()

        #cleaning text
        re_clean = re.sub(r'[^a-z0-9. -]','',text_low, flags = re.IGNORECASE|re.MULTILINE)


        sentence = re_clean.split(" ")
        new_text = ""
        for i in sentence:
            new_text += i+" "
        a = new_text
        b = a.split(".")
        del b[-1]

        return b

    #fungsi untuk proses prediksi data uji
    def ProcessingText(self,datatestPath):
        sentence_vect = self.vectorizer_data_test(datatestPath)
        prediksi_test = self.predict(sentence_vect)

        #set nilai jumlah kalimat positif dan negatif  = 0
        positif = 0
        negatif = 0

        for i in range(len(prediksi_test)):
            if prediksi_test[i]=='positif':
                positif +=1
            else:
                negatif +=1

        #persentase total kalimat positif dan negatif
        perc_pos = positif/len(prediksi_test)*100
        perc_neg = negatif/len(prediksi_test)*100

        # print("Positif : %s"%positif+" |Negatif : %s"%negatif)
        # print("Positif : %0.2f"%perc_pos +" %")
        # print("Negatif : %0.2f"%perc_neg +" %")
        # print("Total data : " + str(positif+negatif))

        dataHasil = {}
        # result = pd.DataFrame({'Kalimat' : datatestPath, 'Label' : prediksi_test})
        for i in range(len(prediksi_test)):
            dataHasil = {
                'kalimat':datatestPath,
                'label':prediksi_test
            }
        # print(dataHasil["label"])
        return [dataHasil,positif,negatif,perc_pos,perc_neg,str(positif+negatif)]

#Main
if __name__=='__main__':
    print("berhasil")