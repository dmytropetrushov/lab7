import nltk
import re
from nltk.corpus import conll2000, treebank, brown

# Task 1: Tokenize and Part-of-Speech (POS) tag a given text
text = "Machine Learning evolved from computer science that primarily studies the design of algorithms that can learn " \
       "from experience. To learn, they need data that has certain attributes based on which the algorithms try to " \
       "find some meaningful predictive patterns. Majorly, ML tasks can be categorized as concept learning, " \
       "clustering, predictive modeling, etc. The ultimate goal of ML algorithms is to be able to take decisions " \
       "without any human intervention correctly. Predicting the stocks or weather are a couple of applications of " \
       "machine learning algorithms."
tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)

# Task 2: Print information about given POS tags
tags = ['CC', 'IN', 'NN', 'JJ']
for tag in tags:
    print(nltk.help.upenn_tagset(tag))

# Task 3: Print raw text from conll2000 and treebank corpora
print(conll2000.raw())
print(treebank.raw())

# Task 4: Print tagged sentences from conll2000 and brown corpora
print(conll2000.tagged_sents())
print(brown.tagged_sents())

# Task 5: Tag words in a given text based on predefined patterns
patterns = [
    (r'\d+', 'CD'),  # CD: cardinal number
    (r'\b(the|a|an)\b', 'DT'),  # DT: determiner
    (r'\b\w+(ous|able|ible|al|ful|ic|ish|ive|less|like|y|er|est)\b', 'JJ'),  # JJ: adjective
    (r'\b\w+ly\b', 'RB'),  # RB: adverb
    (r'\b\w+s\b', 'NNS'),  # NNS: plural noun
    (r'\b\w+ing\b', 'VBG'),  # VBG: gerund verb
    (r'\b\w+ed\b', 'VBD'),  # VBD: past tense verb
    (r'\b\w+\b', 'NN'),  # NN: singular noun
    (r'\b(can|will|should)\b', 'MD'),  # MD: modal auxiliary verb
]
regex_patterns = [(re.compile(p), tag) for (p, tag) in patterns]
sentences = brown.sents(categories='fiction')[:2]
tagged_sentences = [[(word, tag) if pattern.search(word) else (word, None) for pattern, tag in regex_patterns] for
                    sentence in sentences for word in sentence]
for tagged_sentence in tagged_sentences:
    print(tagged_sentence)

# Task 6: Tag words in a given Ukrainian text based on predefined patterns
ukr_patterns = [
    (r'(у|на|з|поряд|в|до|по|без])$', 'ПРИЙМ'),
    (r'(.*[а|е|и|і|о|у|я|ю|є|ї])+$', 'ДІЄСЛ'),
    (r'(.*[ий|а|е|є|ій|ої|і|у])+$', 'ПРИКМ'),
    (r'(.*[а|е|є|і|о|у|я|ю])+$', 'ІМЕН')
]

ukr_text = 'дивитись на  рухливі цятки вогню завороженний , на тремтячі його язички, що, ' \
           'коливаючись, щось безмовно розповідали й дорозі, й деревам, і ' \
           'чорному небу, — і все їх, мабуть, розуміло, бо ж слухало заворожено.  '
regexp_tagger1 = nltk.RegexpTagger(ukr_patterns)
print(regexp_tagger1.tag(nltk.word_tokenize(ukr_text)))
