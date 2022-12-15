import numpy as py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import datetime

import nltk
import re

print("Downloading required libraries from NLTK.")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("Downloaded all the required libraries.")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer


try:
    import src.utilities as utilities
except:
    print("Something went wrong..")

#from imap_tools import MailBox, AND
import imaplib, email

import os, sys, json

#Original Source: https://www.youtube.com/watch?v=dVKNy4eA3tI

class Main(object):
    def __init__(self):
        self.root = os.path.abspath(os.curdir)
        self.config = self.root + "/config/conf.json"

    def read_emails(self, user, passw):
        print("Trying to connect to imap.gmail.com")
        mb= imaplib.IMAP4_SSL("imap.gmail.com")
        print("Connected to the server.")
        rv, message = mb.login(user, passw)
        print("Logging into the account.")
        # 'OK', [b'LOGIN completed']
        rv, num_emails = M.select('Inbox')
        # 'OK', [b'22']

        # Get unread messages
        rv, messages = M.search(None, 'UNSEEN')
        # 'OK', [b'21 22']

        # Download a message
        typ, data = M.fetch(b'21', '(RFC822)')

        # Parse the email
        msg = email.message_from_bytes(data[0][1])
        print(msg['From'], ":", msg['Subject'])

        # Print the Plain Text (is this always the plain text?)
        print(msg.get_payload()[0].get_payload())

    def worker(self):
        data = {}
        spam = False
        scores = {}
        with open(self.config) as af:
            data = json.load(af)
            #print(data['datasets'])
        for email in os.listdir("input/mails"):

            utils = utilities.Api()

            f = open("input/mails/" + email, 'r', encoding="utf-8")
            self.getData("input/mails/" + email)

            content = utils.extract(f, f.name)['text']
            for datasete in data['datasets']:
                dataset = data['datasets'][datasete]
                source = dataset['source']
                path = dataset['path']
                title = dataset['title']
                description = dataset['description']
                cat = dataset['gatName']
                text = dataset['textName']
                
                raw_mail_data = pd.read_csv(self.root + path, encoding="ISO-8859-1")
                mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

                mail_data.loc[mail_data[cat] == 'spam', cat,] = 0
                mail_data.loc[mail_data[cat] == 'ham', cat,] = 1

                X = mail_data[text]
                Y = mail_data[cat]

                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=3)

                feature_extraction = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)

                X_train_features = feature_extraction.fit_transform(X_train)
                X_test_features = feature_extraction.transform(X_test)

                Y_train = Y_train.astype('int')
                Y_test = Y_test.astype('int')

                model = LinearSVC()
                model.fit(X_train_features, Y_train)


                prediction_on_training_data = model.predict(X_train_features)
                accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

                print(title + "Accuracy on the training data : ", accuracy_on_training_data)

                prediction_on_test_data = model.predict(X_test_features)
                accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

                print(title + "Accuracy on test data : ", accuracy_on_test_data)

                #Prediction on email

                input_email = [content]

                input_mail_features = feature_extraction.transform(input_email)
                prediction = model.predict(input_mail_features)

                if(prediction[0] == 1):
                    
                    print(f"{title} flagged the email as non threatning (HAM)")
                elif (prediction[0] == 0):
                    spam = True
                    print(f"{title} has detected the email as spam")
                else:
                    print("Clean email. Lucky this time haha")

            if spam == True:
                self.stats(1, 0)
                print("We marked the email as spam.")
            else:
                self.stats(0, 1)
                print("We did not find anything weird.")

    def advanced_analysis(self, content):
        path = 'input/data/hamnspam/'
        mails = []
        labels = []

        for label in ['spam/', 'ham/'] :
            
            f_name  = os.listdir(os.path.join(path,label))
            
            for name in f_name :
                f = open((path + label + name), 'r', encoding = 'latin-1')
                bolk = f.read()
                mails.append(bolk)
                labels.append(label)
            print("Appended all the different files.")
        df = pd.DataFrame({'emails' : mails, 'label':labels})
        print("Appending it to a variable.")
        from sklearn.preprocessing import LabelEncoder
        encoder = LabelEncoder()
        df['label'] = encoder.fit_transform(df['label'])

        df['emails'] = df['emails'].apply(lambda x : x.lower())
        df['emails'] = df['emails'].apply(lambda x:x.replace('\n',''))
        df['emails'] = df['emails'].apply(lambda x:x.replace('\t',''))
        print("Pretty cleaning our label, \"Email\".")


        lemm = WordNetLemmatizer()
        processed_text = []
        

        for i in range(len(df)) :
            
            text = re.sub('^a-zA-z',' ',df['emails'][i])
            words = text.split()
            words = [lemm.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
            text_p = ' '.join(words)
            processed_text.append(text_p)

        cv = CountVectorizer()
        X = cv.fit_transform(processed_text).toarray()
        y = df['label']

        print("Starting to train the data.")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
        print("Data trained.")

        from sklearn.naive_bayes import MultinomialNB
        model = MultinomialNB()

        model.fit(X_train,y_train)
        print("Using model.fit.")

        y_pred = model.predict(X_test)
        print("Making a test prediction..")

        # Check accuracy

        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(y_test, y_pred)
        print(accuracy)
        input_email = [content]

        print("Starting our normal prediction.")
        input_mail_features = cv.fit_transform(input_email).toarray()
        prediction = model.predict(input_mail_features)
        
        print(prediction)

    def stats(self, amount_of_spam = 0, amount_of_ham = 0):
        data = {}
        date = datetime.datetime.now()
        year = date.year
        month = date.month
        day = date.day
        if os.path.exists('output/statistics.json'):
            with open("output/statistics.json") as f:
                data = json.load(f)

            with open("output/statistics.json", 'w') as f:
                data[str(year)][str(month)][str(day)]['Spam'] += amount_of_spam
                data[str(year)][str(month)][str(day)]['Ham'] += amount_of_ham
                
                json.dump(data, f, indent=4)
        else:
            with open("output/statistics.json", 'w') as f:
                
                data = {
                    year:
                        {
                            month: {
                                day: {
                                    "Spam": amount_of_spam,
                                    "Ham": amount_of_ham
                                }
                            }
                        }

                }
                json.dump(data, f, indent=4)

    def linkStorage(self, link = None):
        if os.path.exists('output/links.json'):
            with open('output/links.json') as f:
                data = json.load(f)
            with open('output/links.json', 'w') as w:
                data['links'].append(link)
                json.dump(data, w, indent=4)
        else:
            with open('output/links.json', 'w') as w:
                data = {
                    "links": [link]
                }
                json.dump(data, w, indent=4)

    def subjectStorage(self, subject = None):
        if os.path.exists('output/subjects.json'):
            with open('output/subjects.json') as f:
                data = json.load(f)
            with open('output/subjects.json', 'w') as w:
                data['subjects'].append(subject)
                json.dump(data, w, indent=4)
        else:
            with open('output/subjects.json', 'w') as w:
                data = {
                    "subjects": [subject]
                }
                json.dump(data, w, indent=4)
    
    def getData(self, file):
        with open(file, encoding="utf-8") as f:
            data = f.read()
            victim = None
            victims = []
            hasHTML = False
            links = []


            if data.__contains__('Delivered-To: '):
                victim = data.split('Delivered-To: ')[1]
                victim = victim.split('Received:')[0]

            if data.__contains__('To'):
                c = data.split('To: ')[2]
                victims = c.split('Content-Type')[0]
            
            regex=r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"


            if data.lower().__contains__("www") or data.lower().__contains__('http'):
                matches = re.findall(regex, data)
                for match in matches:
                    self.linkStorage(match)
main = Main()

main.worker()