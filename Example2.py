from urllib.request import urlopen
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from heapq import nlargest
from collections import defaultdict

def parseArticlePage(url):

    page = urlopen(url).read().decode('utf8', 'ignore')

    #BeautifulSoup create a tree structure.
    soup = BeautifulSoup(page, "lxml")

    #Join the text of each articles
    text = ' '.join(map(lambda p: p.text, soup.find_all('article')))

    # Removing encoding characters
    return text.encode('ascii', errors='replace').decode("utf-8").replace("?", "")

def summarize(text, n):

    sents = sent_tokenize(text)

    assert n <= len(sents)

    _stopwords = set(stopwords.words('english') + list(punctuation))

    word_sent = word_tokenize(text.lower())
    word_sent = [word for word in word_sent if word not in _stopwords]

    freq = FreqDist(word_sent)

    # Find the rank of each sentence
    ranking = defaultdict(int)
    for i, sent in enumerate(sents):
        for w in word_tokenize(sent.lower()):
            if w in freq:
                ranking[i] += freq[w];

    # Top 4 most important sentences by frequency
    sents_idx = nlargest(n, ranking, key=ranking.get)

    # Sort the sentences by index
    return [sents[j] for j in sorted(sents_idx)]

articleUrl = "https://www.washingtonpost.com/entertainment/when-adam-schumann-went-to-war-he-didnt-foresee-its-horrors--or-a-movie-about-his-life/2017/10/19/ece4ebc2-aeec-11e7-9e58-e6288544af98_story.html?hpid=hp_hp-top-table-main_sa-thankyou-1232pm%3Ahomepage%2Fstory&utm_term=.dc38cff17472"

text = parseArticlePage(articleUrl)

print (summarize(text, 3))



