import nltk

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.collocations import *
from nltk.stem.lancaster import LancasterStemmer
from nltk.wsd import lesk
from string import punctuation

#######################################################################
# Tokenisation
#######################################################################

text="Mary had a little lamb. Her fleece was white as snow"

sents=sent_tokenize(text)
#Result: 'Mary had a little lamb.', 'Her fleece was white as snow"

words = [word_tokenize(sent) for sent in sents]
#Result: 'Mary', 'had', 'a', 'little', 'lamb', '.', 'Her', 'fleece', 'was', 'white', 'as', 'snow'

#######################################################################
# Stopwords removal
#######################################################################

#We have a corpus of existing words to be ignored in the NLTK library, to what we can add more words. In our case, we add punctuation.
customStopWords = set(stopwords.words('english')+list(punctuation))

wordsWOStopWords = [word for word in word_tokenize(text) if word not in customStopWords]
#Result: ['Mary', 'little', 'lamb', 'Her', 'fleece', 'white', 'snow']

#######################################################################
# N-Grams
#######################################################################

#Collocations, or N-grams are words which goes together.
#Constructs all the bigrams and add their frequencies, by checking consecutive words. This is an ordered list by bigrams. Trigram method exists too.
finder = BigramCollocationFinder.from_words(wordsWOStopWords)
sortedItems = sorted(finder.ngram_fd.items())
#Result: (('Her', 'fleece'), 1), (('Mary', 'little'), 1), (('fleece', 'white'), 1), (('lamb', 'Her'), 1), (('little', 'lamb'), 1), (('white', 'snow'), 1)]

#######################################################################
# Stemming
#######################################################################

text2="Mary closed on closing night when she was in the mood to close."

#There are specific algorithms to reduce these words to their root form
st = LancasterStemmer()
stemmedWords = [st.stem(word) for word in word_tokenize(text2)]
#Result: ['mary', 'clos', 'on', 'clos', 'night', 'when', 'she', 'was', 'in', 'the', 'mood', 'to', 'clos', '.']

#######################################################################
# Part of Speech
#######################################################################

#Tag values: http://www.nltk.org/book/ch05.html#tab-universal-tagset

tags = nltk.pos_tag(word_tokenize(text2))
#Result: [('Mary', 'NNP'), ('closed', 'VBD'), ('on', 'IN'), ('closing', 'NN'), ('night', 'NN'), ('when', 'WRB'), ('she', 'PRP'), ('was', 'VBD'), ('in', 'IN'), ('the', 'DT'), ('mood', 'NN'), ('to', 'TO'), ('close', 'VB'), ('.', '.')]

#######################################################################
# Word meaning disambiguation
#######################################################################

#Wordnet is a lexicon, we use it to retrieve the meaning of words
#In the following example, we retrieve the definitions for the word 'Bass'
for ss in wordnet.synsets('bass'):
    print(ss, ss.definition())

# LESK is an algorithm for Word Sense Disambiguation.
# Depending on the context of the sentence, the right definition is selected

sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"), 'bass')
#Result: Synset('bass.n.07')

sense2 = lesk(word_tokenize("This sea bass was really hard to catch"), 'bass')
#Result: Synset('sea_bass.n.01')
