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
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
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

try:
    import tkinter
    from tkinter import END, SUNKEN
    import customtkinter
    from PIL import Image

except:
    print("Failed to import Tkinter or Customtkinter. View docs")
    exit(0)

#Original Source: https://www.youtube.com/watch?v=dVKNy4eA3tI

class GUI(object):
    def __init__(self):
        customtkinter.set_appearance_mode("light")  # Modes: "System" (standard), "Dark", "Light"
        customtkinter.set_default_color_theme("green")  # Themes: "blue" (standard), "green", "dark-blue"
        
        self.main = Main()

        self.app = customtkinter.CTk()
        self.app.geometry("1200x1440")
        self.app.title("NettVett - Spam Radar")
        p1 = tkinter.PhotoImage(file = 'input/GUI/icon_192x192.png')
        self.app.iconphoto(False, p1)

        self.logged_in = False

        self.password_icon = customtkinter.CTkImage(Image.open(os.path.join("input/GUI/", "password.png")), size=(26, 26))
        self.username_icon = customtkinter.CTkImage(Image.open(os.path.join("input/GUI/", "user.png")), size=(26, 26))
        
        self.links_48_icon = customtkinter.CTkImage(Image.open(os.path.join("input/GUI/", "link-48.png")), size=(26, 26))
        self.links_96_icon = customtkinter.CTkImage(Image.open(os.path.join("input/GUI/", "link-96.png")), size=(26, 26))

        self.malicious_icon = customtkinter.CTkImage(Image.open(os.path.join("input/GUI/", "malicious.png")), size=(36, 36))
        self.safe_icon = customtkinter.CTkImage(Image.open(os.path.join("input/GUI/", "safe.png")), size=(36, 36))

        self.error_icon = customtkinter.CTkImage(Image.open(os.path.join("input/GUI/", "error.png")), size=(36, 36))
        self.main_frame = customtkinter.CTkFrame(self.app, bg_color='#F8F6F7', fg_color='#F8F6F7')
        self.frame_1 = customtkinter.CTkFrame(self.main_frame, height=50, bg_color="transparent", fg_color='#FFFFFF', corner_radius=15)

        
        self.user_frame = customtkinter.CTkFrame(master=self.frame_1, bg_color="transparent")
        self.password_frame = customtkinter.CTkFrame(master=self.frame_1, bg_color="transparent")

        self.password = customtkinter.CTkEntry(master=self.password_frame,
                               placeholder_text="App-Password",
                               placeholder_text_color="#000000",
                               #text_font="none 10",
                               text_color="black",
                               font=("Helvetica", 14),
                               
                               width=250,
                               height=35,
                               border_width=2,
                               border_color= "#aba9a9",
                               bg_color="#e6e1e1",
                               fg_color= "#e6e1e1",
                               
                               corner_radius=5)

        self.user = customtkinter.CTkEntry(master=self.user_frame,
                               placeholder_text="Email.",
                               placeholder_text_color="#000000",
                               #text_font="none 10",
                               text_color="black",
                               font=("Helvetica", 14),
                               
                               width=250,
                               height=35,
                               border_width=2,
                               border_color= "#aba9a9",
                               bg_color="#e6e1e1",
                               fg_color= "#e6e1e1",
                               
                               corner_radius=5)

    def button_callback():
        print("Button click", combobox_1.get())


    def slider_callback(self, value):
        progressbar_1.set(value)


    def work(self):

        
        self.main_frame.pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=True, padx=50, pady=50)
        
        
        self.frame_1.pack(side=customtkinter.LEFT, fill=customtkinter.BOTH, expand=True, padx=15, pady=15)

        label_1 = customtkinter.CTkLabel(master=self.frame_1, text="Nettvett | Spam-Radar", font=("none", 36), text_color="#85BF41",justify=tkinter.LEFT)
        label_1.pack(pady=10, padx=10)

        
        self.user_frame.pack(pady=20, padx=35)

        

        

        user_image = customtkinter.CTkLabel(master=self.user_frame, text="",image=self.username_icon,
                               #bg_color="#e6e1e1",
                               #fg_color= "#e6e1e1",
                               corner_radius=5)
        user_image.pack(side="right", padx=5)
        self.user.pack(side="left", padx=10, pady=10)

        
        self.password_frame.pack(pady=0, padx=35)

        

        

        password_image = customtkinter.CTkLabel(master=self.password_frame, text="",image=self.password_icon,
        #border_color= "#aba9a9",
        #bg_color="#e6e1e1",
        #fg_color= "#e6e1e1",
        corner_radius=5)
        password_image.pack(side="right", padx=5)
        self.password.pack(side="left", padx=10, pady=10)

        self.login_status = customtkinter.CTkLabel(master=self.frame_1, text="Not connected.", text_color="gray")
        self.login_status.pack(padx=10, pady=0)

        login_btn = customtkinter.CTkButton(
        master = self.frame_1,
        text= "Login",
        command=self.login,
        #text_font="none 12 bold",
        text_color="white",
        hover= True,
        hover_color= "#81b867",
        height=40,
        width= 120,
        border_width=2,
        corner_radius=5,
        border_color= "#608a4d", 
        bg_color="#262626",
        fg_color= "#79ae61")

        login_btn.pack(pady=10, padx=10)


        read_frame = customtkinter.CTkFrame(self.app, height=500)
        read_frame.pack(padx=10, pady=35)
        
        read_emails_btn = customtkinter.CTkButton(
        master = read_frame,
        text= "Find Spam",
        command=self.read,
        #text_font="none 12 bold",
        text_color="white",
        hover= True,
        hover_color= "#81b867",
        height=40,
        width= 120,
        border_width=2,
        corner_radius=5,
        border_color= "#608a4d", 
        bg_color="#262626",
        fg_color= "#79ae61"
        )
        self.links = {}

        read_emails_btn.pack(pady=10, padx=10)
        with open("output/links.json") as f:
                #print(json.load(f))
                links = json.load(f)
        scrollbar = tkinter.Scrollbar(read_frame, orient=tkinter.VERTICAL)
        
        

        self.rating_label = customtkinter.CTkLabel(read_frame, text="We rated this as safe ", image=self.safe_icon, compound="right", font=("Poppins", 20))
        self.rating_label.pack(padx=0, pady=10)

        self.subject_label = customtkinter.CTkLabel(read_frame, text="New Year's Resolution: Let Domino's tip you 🤑 ", font=("Poppins", 16))
        self.subject_label.pack(padx=0, pady=1)

        self.from_label = customtkinter.CTkLabel(read_frame, text="\"Domino's Pizza\" <offers@e-offers.dominos.com> ", font=("Poppins", 14))
        self.from_label.pack(padx=0, pady=0)

        result_label = customtkinter.CTkLabel(read_frame, text="Links", image=self.links_96_icon, compound="left")
        result_label.pack(padx=0, pady=10)
        self.results = tkinter.Listbox(read_frame, bg="light blue", fg="white", selectmode=tkinter.BROWSE, highlightthickness=0, relief=SUNKEN, borderwidth=2, width=50, height=50, font=("Poppins", 20))
        
        self.results.configure(yscrollcommand=scrollbar.set)
        self.results.pack(side=tkinter.LEFT, fill=tkinter.BOTH, pady=0, padx=10)
        scrollbar.pack(side=tkinter.RIGHT, fill=tkinter.Y)

        self.app.mainloop()

    def read(self):
        if self.logged_in == True:
            malicious = self.main.worker(self.user.get(), self.password.get())
            
            self.subject_label.configure(text=self.main.subject)
            self.from_label.configure(text=self.main.from_)
        
            with open("output/links.json") as f:
                #print(json.load(f))
                
                links = json.load(f)
            
            for link in links['links']:
                self.results.insert(END, link)
            if malicious == True:
                self.rating_label.configure(text="We rated this as spam ", text_color="red", image=self.malicious_icon, compound="right")
            else:
                self.rating_label.configure(text="We rated this as clean ", text_color="green", image=self.safe_icon, compound="right")
        else:
            self.rating_label.configure(text="Error. You need to be logged in order to evaluate your emails. ", text_color="red", image=self.error_icon, compound="right")
            print("You need to be logged in.")

    def login(self):
        print("Accessed function login..")
        if self.main.login(self.user.get(), self.password.get()):
            self.logged_in = True
            self.login_status.configure(text="Connection secured.", text_color="green")
        else:
            self.logged_in = False
            self.login_status.configure(text="Connection failed.", text_color="red")

class Main(object):
    def __init__(self):
        self.root = os.path.abspath(os.curdir)
        self.config = self.root + "/config/conf.json"

        self.subject = ""
        self.from_ = ""

        self.mail = imaplib.IMAP4_SSL("imap.gmail.com")

    def login(self, username, password):
        try:
            rv, message = self.mail.login(username, password)
            return True
        except:
            return False
        #print(self.mail.login(username, password))


    def read_emails(self, user, passw):
        print("Trying to connect to imap.gmail.com")
        mb= imaplib.IMAP4_SSL("imap.gmail.com")
        print("Connected to the server.")
        rv, message = mb.login(user, passw)
        print("Logging into the account.")
        # 'OK', [b'LOGIN completed']
        rv, num_emails = mb.select('Inbox')
        # 'OK', [b'22']

        # Get unread messages
        rv, messages = mb.search(None, 'UNSEEN')
        # 'OK', [b'21 22']

        # Download a message
        typ, data = mb.fetch(b'21', '(RFC822)')

        # Parse the email
        msg = email.message_from_bytes(data[0][1])
        return msg
        print(msg['From'], ":", msg['Subject'])

        # Print the Plain Text (is this always the plain text?)
        print(msg.get_payload()[0].get_payload())

    def worker(self, user, password):
        data = {}
        spam = False
        scores = {}
        with open(self.config) as af:
            data = json.load(af)
            #print(data['datasets'])

        utils = utilities.Api()
        self.msg = self.read_emails(user, password)
        self.getData(self.msg)

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

            print('- ' + title + " Accuracy on the training data : ", accuracy_on_training_data)

            prediction_on_test_data = model.predict(X_test_features)
            accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

            print("- " + title + " Accuracy on test data : ", accuracy_on_test_data)

            #Prediction on email

            input_email = self.msg
            print(input_email)

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
            print("We marked the email as spam.\n")
            return True
        else:
            self.stats(0, 1)
            print("We did not find anything weird.\n")
            return False

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
                data['links'] = link
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
                data['subjects'] = subject
                json.dump(data, w, indent=4)
        else:
            with open('output/subjects.json', 'w') as w:
                data = {
                    "subjects": [subject]
                }
                json.dump(data, w, indent=4)
    
    def fromStorage(self, from_ = None):
        if os.path.exists('output/spam_emails.json'):
            with open('output/spam_emails.json') as f:
                data = json.load(f)
            with open('output/spam_emails.json', 'w') as w:
                data['spam_emails'] = from_
                json.dump(data, w, indent=4)
        else:
            with open('output/spam_emails.json', 'w') as w:
                data = {
                    "spam_emails": [from_]
                }
                json.dump(data, w, indent=4)

    def getData(self, content):
            data = content
            victim = None
            victims = []
            hasHTML = False
            links = []

            print(data.keys())
            if 'Delivered-To' in data.keys():
                victim = data['Delivered-To']

            if 'To' in data.keys():
                victims = data['To']

            if 'Subject' in data.keys():
                subject = data['Subject']
                self.subject = subject

            if 'From' in data.keys():
                from_ = data['From']
                new_from = str(from_).split('<')[1]
                new_from = new_from.split('>')[0]
                self.from_ = new_from
            
            regex=r"\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b"

            payload = data.get_payload()
            for content in payload:
                if content.__contains__("www") or content.__contains__('http'):
                    matches = re.findall(regex, content)
                    for match in matches:
                        print(match)
                        self.linkStorage(match)
            

main = Main()

#main.worker()
#main.read_emails("lukeproductions3@gmail.com", "mmwyktfkyqzauyif")
#Main().login("lukproductions3@gmail.com", 'asdk')

GUI().work()