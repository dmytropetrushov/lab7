# import necessary libraries
import nltk
from nltk.corpus import brown, conll2000, treebank

# Сomment the lines below to remove downloadable NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
nltk.download('conll2000')
nltk.download('treebank')
nltk.download('brown')

# define text to be analyzed
text = 'Machine Learning evolved from computer science that primarily studies the design of ' \
       'algorithms that can learn from experience. To learn, they need data that has certain ' \
       'attributes based on which the algorithms try to find some meaningful predictive patterns. ' \
       'Majorly, ML tasks can be categorized as concept learning, clustering, predictive ' \
       'modeling, etc. The ultimate goal of ML algorithms is to be able to take decisions without ' \
       'any human intervention correctly. Predicting the stocks or weather are a couple of ' \
       'applications of machine learning algorithms.'

# tokenize the text into words and then tag each word with its part of speech
tokenised = nltk.word_tokenize(text)
taged = nltk.pos_tag(tokenised)

# print the tagged words
print(taged)

# define a list of POS tags and print the definition of each tag using the upenn_tagset method
tags = ['CC', 'IN', 'NN', 'JJ', 'DT', 'CD']
for i in tags:
    nltk.help.upenn_tagset(i)

# print the raw data of conll2000 and treebank corpora
print(conll2000.raw())
print(treebank.raw())

# print the tagged sentences of conll2000 and brown corpora
print(conll2000.tagged_sents())
print(brown.tagged_sents())

# select a subset of sentences from the brown corpus
sent = brown.sents(categories='fiction')

# Task 2: Define regex patterns for different POS tags
patterns = [
    (r'\d ', 'CD'),
    (r'(this|these|those|that)$', 'DT'),
    (r'.*(ble|full)$', 'JJ'),
    (r'(CC)$', 'CC'),
    (r'(IN)$', 'IN'),
    (r'(NN|NNS|NNP|NNPS)$', 'NN')
]

# apply the regex patterns to the selected sentences from the brown corpus
regexp_tagger = nltk.RegexpTagger(patterns)
print(regexp_tagger.tag(sent[0]), '\n', regexp_tagger.tag(sent[1]))

# Task 3: Define regex patterns for Ukrainian POS tags
ukr_patterns = [
    (r'(у|на|з|поряд|в|до|по|без])$', 'ПРИЙМ'),
    (r'(.*[а|е|и|і|о|у|я|ю|є|ї])+$', 'ДІЄСЛ'),
    (r'(.*[ий|а|е|є|ій|ої|і|у])+$', 'ПРИКМ'),
    (r'(.*[а|е|є|і|о|у|я|ю])+$', 'ІМЕН')
]

# apply the regex patterns to a sample Ukrainian text
ukr_text = 'дивитись на  рухливі цятки вогню завороженний , на тремтячі його язички, що, ' \
           'коливаючись, щось безмовно розповідали й дорозі, й деревам, і ' \
           'чорному небу, — і все їх, мабуть, розуміло, бо ж слухало заворожено.  '
regexp_tagger1 = nltk.RegexpTagger(ukr_patterns)
print(regexp_tagger1.tag(nltk.word_tokenize(ukr_text)))
