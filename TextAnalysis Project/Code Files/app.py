from tracemalloc import stop
import streamlit as st

import numpy as np
import pandas as pd
import re

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

nltk.download('punkt')

nltk.download('stopwords')

sw=nltk.corpus.stopwords.words("english")


rad=st.sidebar.radio("Navigation",["Home","Spam or Ham Detection","Sentiment Analysis","Stress Detection","Hate and Offensive Content Detection","Sarcasm Detection"])
#upload_file = st.sidebar.file_uploader("Upload Your file here")
#0) Home Page

if rad=="Home":
    st.title("Hello, Welcome to the Complete Text Analysis WebApp :wave:")
    
    st.image("Complete Text Analysis Home Page.jpg")
    st.text(" ")
    
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    
    st.text("1. Spam or Ham Detection")
    st.text("2. Sentiment Analysis")
    st.text("3. Stress Detection")
    st.text("4. Hate and Offensive Content Detection")
    st.text("5. Sarcasm Detection")
    

    with st.container():
        st.write("----")
        st.header("Any Doubt? Get in touch with us")
        st.write("##")
        
        contact_form = """
        <form action="https://formsubmit.co/pranavjoshi6565@gmail.com" method="POST">
        <input type="hidden" name="_captcha" value="false">
        
        <input type="text" name="name" placeholder="Your name" required>
        
        <input type="email" name="email" placeholder="Your email" required>
        
        <textarea name="message" placeholder="Your message here" required></textarea>
        
        <button type="submit">Send</button>
        
        </form>
        """
        left_column, right_column = st.columns(2)
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st.empty()
        

        
    
    
    

#function to clean and transform the user input which is in raw format

def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    
    ps=PorterStemmer()
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)

#1) Spam Detection Prediction

tfidf1=TfidfVectorizer(stop_words=sw,max_features=20)
def transform1(txt1):
    txt2=tfidf1.fit_transform(txt1)
    return txt2.toarray()

df1=pd.read_csv("Spam Detection.csv")
df1.columns=["Label","Text"]
x=transform1(df1["Text"])
y=df1["Label"]

# training the model for spam detection

x_train1,x_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.1,random_state=0)
model1=LogisticRegression()
model1.fit(x_train1,y_train1)

#Spam Detection Analysis Page
    

if rad=="Spam or Ham Detection":
    st.header("Detect Whether A Text Is Spam Or Ham? Detect here ")
    sent1=st.text_area("Enter The Text")
    transformed_sent1=transform_text(sent1)
    vector_sent1=tfidf1.transform([transformed_sent1])
    prediction1=model1.predict(vector_sent1)[0]

    if st.button("Predict"):
        if prediction1=="spam":
            st.warning("Spam Text!!")
        elif prediction1=="ham":
            st.success("Ham Text!!")
            
    upload_file = st.file_uploader("Upload Your file here")
    
    with st.container():
                
        
        st.image("SpamImage.png") 
                
                
                
            
        
        st.subheader("Spam or Ham Analysis-")
        
        st.write("""This is a project I am working on while learning concepts of 
                data science and machine learning. 
                The goal here is to identify whether a text is spam or ham.
                We will take a dataset of unstructered csv messages and apply classification 
                techniques. We can later test the model for accuracy and performance on 
                unclassified text messages. Similar techniques can be applied to other
                NLP applications like sentiment analysis etc.""")
        
        st.text(" ")
        
    st.subheader("So What is Spam?")
    
    
    st.write("""Wikipedia describes Spam as “the use of electronic messaging systems to send unsolicited bulk messages, 
             especially advertising, indiscriminately.”  
             This is a commonly accepted definition in the industry, though there are 
             variations in how governments define and regulate spam.  
             For example, in the United States, any commercial email has to comply 
             with Spam regulations, even if the email is not sent in bulk.
            - The key word here is unsolicited. This means that you did not ask for messages from this 
            source. So if you didn’t ask for the mail it must be spam, Right? That is true, however 
            quite often people don’t realize that they are signing up for mailers when they download 
            free software, or sign up for a new service, or even when updating existing software. 
            - The best way to deal with spam is to forward the message to the system administrator.
            In 2003 the CAN-SPAM ACT was made law. This act defines the rules for advertisers and bulk mailers to follow. In order to legally send bulk mail and advertisements, 
            they are required to adhere to the following guidelines:
            The header of the commercial email (indicating the sending source, destination and routing information) doesn't contain materially false or materially misleading information;
            The subject line doesn't contain deceptive information;
            The email provides "clear and conspicuous" identification that it is an advertisement or solicitation;
            The email includes some type of return email address, which can be used to indicate that the recipient no longer wishes to receive spam email from the sender (i.e. to "opt-out");
            The email contains "clear and conspicuous" notice of the opportunity to opt-out of receiving future emails from the sender;
            The email has not been sent within 10 days after the sender received notice that the recipient no longer wishes to receive email from the sender (i.e. has "opted-out");
            The email contains a valid, physical postal address for the sender.
            """
            )
    
    st.text(" ")

    st.subheader("So What is Ham?")

    st.write("""
        The term ‘ham’ was originally coined by SpamBayes sometime around 2001 and is currently defined and understood to be “E-mail that is generally desired and isn't considered spam.”

    Desired? You may be saying to yourself “I do not desire this mail; how is this ham and why am I getting it? “ The answer is you requested it.

    There are two ways you could have signed up for this email.


    Directly- While downloading free software such as a browser or a game or signing up 
    for a new online service you were required to agree to and check the box agreeing to 
    their Terms of Service (TOS). Below or above the TOS were other checkboxes. 

    One said “Yes! I would like to receive information and offers from you and your partners.” 
    If you checked this box, then legally you asked for this email.

    Indirectly- This is the same scenario as Directly signing up except, 
    The box for the information and offers is pre-checked, 
    leaving it for you to uncheck the box if you do not want to be on their mail lists.

    Either way, once you are on a bulk mail list, they can legally send you the offers (and rarely any information worth anything) as long as they follow RFC Regulations.
    The good news is that if they follow RFC Rules, then it is easy to stop these emails. 
    All you have to do is simply “click to unsubscribe,” and the mail stops. 
    That is if they follow rules.

    Malicious spammers especially will take advantage of this and offer the same 
    format at the bottom of their emails linking the unsubscribe link to malicious downloads 
    and/or tracking cookies, etc...


    """
    )

    st.write("[For More Information Visit](https://www.kaggle.com/code/dejavu23/sms-spam-or-ham-beginner)")

#2) Sentiment Analysis Prediction 

tfidf2=TfidfVectorizer(stop_words=sw,max_features=20)
def transform2(txt1):
    txt2=tfidf2.fit_transform(txt1)
    return txt2.toarray()

df2=pd.read_csv("Sentiment Analysis.csv")
df2.columns=["Text","Label"]
x=transform2(df2["Text"])
y=df2["Label"]

# training the model for sentiment analysis

x_train2,x_test2,y_train2,y_test2=train_test_split(x,y,test_size=0.5,random_state=0)
model2=LogisticRegression()
model2.fit(x_train2,y_train2)

#Sentiment Analysis Page

if rad=="Sentiment Analysis":
    st.header("Detect The Sentiment Of The Text :smile: Detect here")
    sent2=st.text_area("Enter The Text")
    transformed_sent2=transform_text(sent2)
    vector_sent2=tfidf2.transform([transformed_sent2])
    prediction2=model2.predict(vector_sent2)[0]

    if st.button("Predict"):
        if prediction2==0:
            st.warning("It's a Negetive Text :cry:")
        elif prediction2==1:
            st.success("It's a Positive Text :smile: ")
            
    st.image("SentimentAnalysisImage.png")
    
    
    st.subheader("WHAT IS SENTIMENT ANALYSIS?")
    
    st.info("""
             
             Sentiment analysis, also referred to as opinion mining, is an approach to natural language processing (NLP)
             that identifies the emotional tone behind a body of text. 
             This is a popular way for organizations to determine and categorize opinions 
             about a product, service or idea. 
             Sentiment analysis involves the use of data mining, machine learning (ML), 
             artificial intelligence and computational linguistics to mine text for sentiment and subjective information such as whether it is expressing positive, negative or neutral feelings.
            Sentiment analysis systems help organizations gather insights into real-time customer sentiment, customer experience and brand reputation. Generally, these tools use text analytics to analyze online sources such as emails, blog posts, online reviews, customer support tickets, news articles, survey responses, case studies, web chats, tweets, forums and comments. Algorithms are used to implement rule-based, automatic or hybrid methods of scoring whether the customer is expressing positive words, negative words or neutral ones.
            In addition to identifying sentiment, sentiment analysis can extract the polarity or the amount of positivity and negativity, subject and opinion holder within the text. This approach is used to analyze various parts of text, such as a full document or a paragraph, sentence or subsentence.
            Vendors that offer sentiment analysis platforms include Brandwatch, Critical Mention, Hootsuite, Lexalytics, Meltwater, MonkeyLearn, NetBase Quid, Sprout Social, Talkwalker and Zoho. Businesses that use these tools to analyze sentiment can review customer feedback more regularly and proactively respond to changes of opinion within the market.



         """)
    
    st.image("SentimentAnalysis2Image.png")
    st.subheader("HOW SENTIMENT ANALYSIS CAN BOOST YOUR BRAND? ")
    st.write("When done right, sentiment analysis adds great value to a business. Sentiment analysis gives a more in-depth insight into how your customers feel and what they expect from your brand.")
    
    st.subheader("Monitor Social Media Mentions")
    st.info("""You can use sentiment analysis to monitor Facebook, Instagram, and Twitter posts. From the results, sentiment analysis helps you categorize and label the mentions in order of urgency. 
    You can then send the remarks to the support team for quick feedback.""")
    
    st.subheader("Track Customer Feedback to Improve Brand Perception")
    st.info("""
             Sentiment analysis can help you leverage customer reviews to improve a product, service, or brand. Sentiment analysis can help you step-up your customer service, 
             so customers are satisfied with every aspect of your brand.""")
    
    st.subheader("GAIN COMPETITOR SIGHTS")
    st.info("""
             You can use sentiment analysis to discover the negative comments about a competitor and find ways to fill the gaps.
            The positive mentions can help you learn what your competitors are doing well,
            which will help advance your strategies.
    
          """
            )
    
    
    
    
    
    st.write("[For More Information Click here](https://monkeylearn.com/sentiment-analysis/)")
   
  
    
#3) Stress Detection Prediction

tfidf3=TfidfVectorizer(stop_words=sw,max_features=20)
def transform3(txt1):
    txt2=tfidf3.fit_transform(txt1)
    return txt2.toarray()

df3=pd.read_csv("Stress Detection.csv")
df3=df3.drop(["subreddit","post_id","sentence_range","syntax_fk_grade"],axis=1)
df3.columns=["Text","Sentiment","Stress Level"]
x=transform3(df3["Text"])
y=df3["Stress Level"].to_numpy()

# stress detection tranining
x_train3,x_test3,y_train3,y_test3=train_test_split(x,y,test_size=0.1,random_state=0)
model3=DecisionTreeRegressor(max_leaf_nodes=2000)
model3.fit(x_train3,y_train3)



#Stress Detection Page
if rad=="Stress Detection":
    st.header("Detect The Amount Of Stress In The Text!!")
    sent3=st.text_area("Enter The Text")
    transformed_sent3=transform_text(sent3)
    vector_sent3=tfidf3.transform([transformed_sent3])
    prediction3=model3.predict(vector_sent3)[0]

    if st.button("Predict"):
        if prediction3>=0:
            st.warning("It's a Stressful Text!! :(")
        elif prediction3<0:
            st.success("It's not Stressful Text!! :)")
    
    st.subheader("What is Stress Detection ?")        
    st.image("Stress-Detection.jpg")
    
    st.write("""The study shows the high potential of ML algorithms in mental health. 
               Aldarwish et al used machine learning algorithms SVM and Naïve- Bayesian for Predicting stress from UGC- User Generated Content in Social media sites (Facebook, Twitter, Live Journal) They used social interaction stress datasets based on mood and negativism and BDI- questionnaire having 6773 posts, 2073 depressed, 4700 non-depressed posts (textual). They achieved an accuracy of 57% from SVM and 63% from Naïve- Bayesian. They also emphasized stress detection using big data techniques [29].
               Cho et al. presented the analysis of ML algorithms for diagnosing mental illness. 
               They studied properties of mental health, techniques to identify, their limitations, 
               and how ML algorithms are implemented. The authors considered SVM, GBM, KNN, Naïve Bayesian, 
               KNN, Random Forest. Deshpande and Rao presented an 
               emotion artificial intelligence technique to detect depression. The authors collected 10,000 
               Tweet Using Twitter API. They applied SVM and Naïve Bayes machine learning algorithms and 
               achieved F1 scores of 79% and 83% respectively. Zucco et al. presented a preliminary design 
               of an integrating Sentiment Analysis (SA) and Affective Computing (AC) methodologies for 
               depression conditions monitoring. 
               The authors described SA and AC analysis pipelines. 
               
               The literature for stress detection shows that the models used for 
               prediction need improvement. The mental health prediction and monitoring also need to be 
               combined with other health parameters such as eating, sleeping,
               physiological and other factors.""")
    
    st.image("Stress-detection3.png")
    
    st.subheader("Benefits of Stress Detection:-")
    
    st.markdown("""- Performance optimization: Stress can significantly impact cognitive functions, 
                memory, concentration, and decision-making abilities.""")
    
    st.markdown("""- Cost savings: Chronic stress can result in increased healthcare costs, absenteeism, 
                and decreased productivity""")
    
    st.markdown("""- Personal well-being: Stress detection can help individuals become more aware of their 
                stress levels and take proactive steps to manage them. By identifying stress early on, 
                individuals can implement strategies to reduce stress, improve their mental health, and 
                enhance overall well-being.""")



#4) Hate & Offensive Content Prediction
tfidf4=TfidfVectorizer(stop_words=sw,max_features=20)
def transform4(txt1):
    txt2=tfidf4.fit_transform(txt1)
    return txt2.toarray()

df4=pd.read_csv("Hate Content Detection.csv")
df4=df4.drop(["Unnamed: 0","count","neither"],axis=1)
df4.columns=["Hate Level","Offensive Level","Class Level","Text"]
x=transform4(df4["Text"])
y=df4["Class Level"]



# training hate and offensive model
x_train4,x_test4,y_train4,y_test4=train_test_split(x,y,test_size=0.1,random_state=0)
model4=RandomForestClassifier()
model4.fit(x_train4,y_train4)



#Hate & Offensive Content WebPage
if rad=="Hate and Offensive Content Detection":
    st.header("Detect The Level Of Hate & Offensive Content In The Text!!")
    sent4=st.text_area("Enter The Text")
    transformed_sent4=transform_text(sent4)
    vector_sent4=tfidf4.transform([transformed_sent4])
    prediction4=model4.predict(vector_sent4)[0]

    if st.button("Predict"):
        if prediction4==0:
            st.exception("Highly Offensive Text!!")
        elif prediction4==1:
            st.warning("Oops, It's a Offensive Text!!")
        elif prediction4==2:
            st.success("Its a Non Offensive Text!!")
            
    st.subheader("Hate and Offensive Meaning")
    st.image("hate speech cropped .png")
    st.write("""Hate speech detection is the task of detecting if communication
             such as text, audio, and so on contains hatred and or encourages violence
             towards a person or a group of people. This is usually based on prejudice 
             against 'protected characteristics' such as their ethnicity, gender, sexual 
             orientation, religion, age et al. Some example benchmarks are ETHOS and HateXplain. 
             Models can be evaluated with metrics like the F-score or F-measure.
    
            
            """)
    
    st.write("[Click here to read more](https://www.geeksforgeeks.org/hate-speech-detection-using-deep-learning/)")



#5) Sarcasm Detection Prediction
tfidf5=TfidfVectorizer(stop_words=sw,max_features=20)
def transform5(txt1):
    txt2=tfidf5.fit_transform(txt1)
    return txt2.toarray()

df5=pd.read_csv("Sarcasm Detection.csv")
df5.columns=["Text","Label"]
x=transform5(df5["Text"])
y=df5["Label"]


# training the sarcasm detection model
x_train5,x_test5,y_train5,y_test5=train_test_split(x,y,test_size=0.1,random_state=0)

model5=LogisticRegression()
model5.fit(x_train5,y_train5) 




#Sarcasm Detection WebPage
if rad=="Sarcasm Detection":
    st.header("Detect Whether The Text Is Sarcastic Or Not!!")
    sent5=st.text_area("Enter The Text")
    transformed_sent5=transform_text(sent5)
    vector_sent5=tfidf5.transform([transformed_sent5])
    prediction5=model5.predict(vector_sent5)[0]

    if st.button("Predict"):
        if prediction5==1:
            st.exception("Sarcastic Text!!")
        elif prediction5==0:
            st.success("Non Sarcastic Text!!")
            
    st.subheader("EXACTLY WHAT IS SARCASM DETECTION IN PYTHON?")
    st.image("sarcasm.jpg")
    st.info(""" SARCASM is a very important part of human speech. 
            It has existed since time immemorial. For example, a plane missing
            “What a great day wow!”. The true meaning of a sentence will darken its true 
            essence, that is, despair from the shadow of anger. 
            The meaning of this sentence can be expressed only through speech, 
            while writing it may cause confusion. It is just a pebble in the mountain 
            of problems with the increasing use of satire as it is a part of many fields
            like politics etc. The project aims to solve the problem of satire detection 
            using ML and neural models to increase understanding for it.
    
    
          """)
    
    
    
    st.write("To read more [Click here](https://www.sciencedirect.com/science/article/abs/pii/S2214785320368164)")
    
            
            


# if rad=="Sarcasm Detec":
#     st.header("Detect Whether The Text Is Sarcastic Or Not!!")
#     sent5=st.nav_bar("Enter The Text")
#     transformed_sent5=transform_text(sent5)
#     vector_sent5=tfidf5.transform([transformed_sent5])
#     prediction5=model5.predict(vector_sent5)[0]

#     if st.button("Predict"):
#         if prediction5==0:
#             st.exception("Sarcastic Text!!")
#         elif prediction5==0:
#             st.success("Non Sarcastic Text!!")
