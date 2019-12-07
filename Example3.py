from urllib.request import urlopen
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from string import punctuation
from heapq import nlargest
import numpy as np

def getAllPosts(url, links):
    response = urlopen(url)
    soup = BeautifulSoup(response, "lxml")
    for a in soup.findAll('a'):
        try:
            url = a['href'];
            title = a['title'];
            if title == "Older Posts":
                print (title, url)
                links.append(url)
                getAllPosts(url, links) # recursive call for downloading all next pages
        except:
            title = ""
    return

def getText(testUrl):
    response = urlopen(testUrl)
    soup = BeautifulSoup(response, "lxml")
    divs = soup.findAll("div", { "class" : "post-body" })
    posts = []
    for div in divs:
        posts += map(lambda p:p.text.encode('ascii', errors='replace').decode("utf-8").replace("?", ""), div.findAll("li"))
    return posts

blogUrl = "http://doxydonkey.blogspot.co.uk/"
links = []
#getAllPosts(blogUrl, links)

links.append("http://doxydonkey.blogspot.co.uk/search?updated-max=2017-05-23T19:53:00-07:00&max-results=7")
links.append("http://doxydonkey.blogspot.co.uk/search?updated-max=2017-05-14T19:02:00-07:00&max-results=7&start=7&by-date=false")

posts = []
for link in links:
    posts += getText(link)

# Converts text to TF-IDF representation
vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english')

# All articles in the list are converted (number of articles * number of distinct words in all articles)
X = vectorizer.fit_transform(posts)

km = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=1, verbose=True)
# (1) number of groups
# (2) algo for choosing initial centroids
# (3) in case of non-convergence, max iteration

km.fit(X)

# Group each article to a label 0, 1, 2 and the number of article in each
# Example: (array([0, 1, 2]), array([ 4, 21, 13]))
arr = np.unique(km.labels_, return_counts=True)

# Clusters are represented by numbers. We need to go through the important words of the articles to associate a user-friendly name
text = {}
for i, cluster in enumerate(km.labels_):
    oneDocument = posts[i]
    if cluster not in text.keys():
        text[cluster] = oneDocument
    else:
        text[cluster] += oneDocument

_stopwords = set(stopwords.words('english') + list(punctuation) + ["million", "millions", "billion", "billions", "year"])

# Top keywords for each cluster
keywords = {}
counts = {}
for cluster in range(3):
    word_sent = word_tokenize(text[cluster].lower())
    word_sent = [word for word in word_sent if word not in _stopwords]
    freq = FreqDist(word_sent)
    keywords[cluster] = nlargest(100, freq, freq.get)
    counts[cluster]=freq

# Top keywords for each cluster which does not appear in the others
unique_keys = {}
for cluster in range(3):
    other_clusters = list(set(range(3)) - set([cluster]))
    keys_other_clusters = set(keywords[other_clusters[0]]).union(set(keywords[other_clusters[1]]))
    unique = set(keywords[cluster])-keys_other_clusters
    unique_keys[cluster]=nlargest(10, unique, key=counts[cluster].get)

print(unique_keys)

# For new articles, it comes back to a classification problem

article = "Facebook, buoyed by robust advertising sales, signaled that it will keep up its brisk pace of investments to attract users and advertisers. Spending climbed 82 percent in the second quarter as the social-media company increased hiring, poured money into data centers and boosted marketing. While that was more than double the rate of sales growth, Facebook still managed to top analysts’ revenue estimate, crossing the $4 billion mark for the first time. Net income shrank to $719 million from $791 million a year earlier, while the operating margin, a measure of profitability, narrowed to 31 percent from 48 percent. Revenue rose 39 percent to $4.04 billion. Yet Facebook also signaled that it wouldn’t let investments grow uncontrollably. The company forecast that expenses will rise 55 percent to 60 percent this year, compared with a previous range of 55 percent to 65 percent. Facebook is improving tools for advertisers and expanding the audience for its mobile applications beyond Facebook itself, including Messenger, Instagram and WhatsApp, which have yet to contribute meaningfully to revenue. Monthly active users for Facebook’s main social network jumped 13 percent to 1.49 billion, with 1.31 billion people logging in at least once a month via mobile. Shares of Menlo Park, California-based Facebook fell 3.4 percent in extended trading, after advancing 1.8 percent to $96.99 at the close in New York. The stock is up 24 percent this year. Facebook’s ability to keep adding users and keeping them engaged stands in stark contrast to Twitter, which is struggling to break past 300 million people. The total number of Facebook users who logged in daily rose to 968 million in June, the company said, slightly less than the 970.5 million projection of four analysts surveyed by Bloomberg."

# Represent the new article with TF-IDF algo
test = vectorizer.transform([article])

# Training phase
classifier = KNeighborsClassifier()
classifier.fit(X, km.labels_)
# Result: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=5, p=2, weights='uniform')

# Test phase
theme = classifier.predict(test)
# Result: [1]
