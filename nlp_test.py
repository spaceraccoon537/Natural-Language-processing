from nltk.corpus import twitter_samples#nltk.download("twitter_samples")
#nltk.download("punkt")
import re,string
from nltk.corpus import stopwords#nltk.download('stopwords')
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

stop_words=stopwords.words('english')
positive_tweets=twitter_samples.strings('positive_tweets.json')
negative_tweets=twitter_samples.strings('negative_tweets.json')
text=twitter_samples.strings('tweets.20150430-223406.json')
tweet_tokens=twitter_samples.tokenized('positive_tweets.json')


def lemmatize_sentence(tokens):
    lemmatizer=WordNetLemmatizer()
    lemmatized_sentence=[]
    for word,tag in pos_tag(tokens):
        if tag.startswith('NN'):
           pos='n'
        elif tag.startswith('VB'):
            pos='v'
        else:
            pos='a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word,pos))
    return lemmatized_sentence

##NNP: Noun, proper, singular
##NN: Noun, common, singular or mass
##IN: Preposition or conjunction, subordinating
##VBG: Verb, gerund or present participle
##VBN: Verb, past participle

def remove_noise(tweet_tokens,stop_words=()):

    cleaned_tokens=[]

    for token,tag in pos_tag(tweet_tokens):
        token=re.sub('http[s]?://(?:[a-zA-Z]|[$-_@.&+#]|[!*\(\),]|'\
                     '(?:%[0-9a-f-A-F][0-9a-fA-F]))+','',token)
        token=re.sub('(@[A-Za-z0-9_]+)','',token)

        if tag.startswith("NN"):
            pos='n'
        elif tag.startswith("VB"):
            pos='v'
        else:
            pos='a'
        lemmatizer=WordNetLemmatizer()
        token=lemmatizer.lemmatize(token,pos)

        if len(token)>0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


positive_tweets_tokens=twitter_samples.tokenized('positive_tweets.json')
negative_tweets_tokens=twitter_samples.tokenized('negative_tweets.json')

positive_cleaned_tokens_list=[]
negative_cleaned_tokens_list=[]

for tokens in positive_tweets_tokens:
    positive_cleaned_tokens_list.append(remove_noise(tokens,stop_words))

for tokens in negative_tweets_tokens:
    negative_cleaned_tokens_list.append(remove_noise(tokens,stop_words))

def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token

all_pos_words=get_all_words(positive_cleaned_tokens_list)
freq_dist_pos=FreqDist(all_pos_words)

def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token,True]for token in tweet_tokens)

positive_tokens_for_model=get_tweets_for_model(positive_cleaned_tokens_list)
negative_tokens_for_model=get_tweets_for_model(negative_cleaned_tokens_list)
    
positive_dataset=[(tweet_dict,"Positive")for tweet_dict in positive_tokens_for_model]
negative_dataset=[(tweet_dict,"Negative")for tweet_dict in negative_tokens_for_model]

dataset=positive_dataset+negative_dataset

random.shuffle(dataset)

train_data=dataset[:7000]
test_data=dataset[7000:]

classifier=NaiveBayesClassifier.train(train_data)

accuracy=classify.accuracy(classifier,test_data)

custom_tweet='I am so unhappy today!!'
custom_tokens=remove_noise(word_tokenize(custom_tweet))

print(classifier.classify(dict([token,True]for token in custom_tokens)))
