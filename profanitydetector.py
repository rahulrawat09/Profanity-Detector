# Importing Libraries
import speech_recognition as sr 
import moviepy.editor as mp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import moviepy
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import SpeechToText.py



'''
Importing Video file
Converting video to audio file
'''
clip = mp.VideoFileClip(r"Sample2.mp4") 
clip.audio.write_audiofile(r"converted2.wav")




# Filter the video
import FilterVideo from filterVideo.py

FilterVideo filterVideo = FilterVideo(clip, bad_words)

filterVideo.filter();


# Speech Recognition API object
r = sr.Recognizer()

# Fetching audio file
audio = sr.AudioFile("converted2.wav")




'''
Using google speech recognition API
to extract text from audio file
'''
with audio as source:
  audio_file = r.record(source)
result = SpeechToText.convertSpeechToText(audio_file)

orginal_words = result.split(" ")




'''
Cleaning Text:
Removing redundant words like helping verbs
'''
nltk.download('stopwords')

ps = PorterStemmer()
words = []

for word in orginal_words:
    if not word in set(stopwords.words('english')):
        words.append(ps.stem(word))




'''
Importing file which consists all Profane words
Adding profane words in **** form to the list
'''
with open("list.txt") as inp:
    profane_words = list(inp)

extra_words = []
for word in profane_words:
    extra_words.append(word[0] + '*'*(len(word) - 1) + '\n')

for word in extra_words:
    profane_words.append(word)




'''
Creating a model to identify profane/bad words
Using python dictionary
'''
not_allowed = {}    

for word in profane_words:
    not_allowed[word] = True




'''
Processing text word by word
If the model identifies the bad word
then store it
'''

## Creating the Bag of Words model
# from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features = 1500)
# X = cv.fit_transform(corpus).toarray()
# y = dataset.iloc[:, 1].values

bad_words = {}

for word in words:
    
    if word + '\n' in not_allowed:
        
        if word not in bad_words:
            bad_words[word] = 0
        
        bad_words[word] += 1




# Total words caught
total_profane_words = sum(bad_words.values())


# Showing all profane words used in video
print("\n\n")
for word in bad_words:
    print("{} is used {} times!!!".format(word, bad_words[word]))
    
print("\n\n")




# Video review
print("Total words used: {}".format(len(words)))
print("Total profane words used: {}".format(total_profane_words))
print("Profane word percentage: {} %".format(str(round(total_profane_words/len(words)*100, 2))))