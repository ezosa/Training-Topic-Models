import os
import string
import pickle
import random
import numpy as np
import json
from nltk.corpus import stopwords
from gensim import corpora
from gensim.models import LdaModel
from gensim.models import LdaSeqModel
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer

stop_fr = set(stopwords.words('french') + ['les','puis','cette','lui','avoir','dès','cest','qui','que',
                                           'tion','leurs','être','aux','quil','elles','quand','ils'])

exclude = set(string.punctuation) - set('\'')
lemmatizer = FrenchLefffLemmatizer()


def clean(doc):
    clean_stop = " ".join([i for i in doc.lower().split() if i not in stop_fr])
    clean_stop = clean_stop.replace('\'',' ')
    clean_punc = ''.join(ch for ch in clean_stop if ch not in exclude)
    clean_tokens = [lemmatizer.lemmatize(word) for word in clean_punc.split()]
    clean_tokens = [tok.lower() for tok in clean_tokens if len(tok) > 2 and tok not in stop_fr]
    return clean_tokens


def get_tfidf_score(articles):
    print("Getting TF-IDF scores")
    corpus = [" ".join(art) for art in articles]
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    scores = np.max(X,axis=0)
    words = vectorizer.get_feature_names()
    tfidf_dict = {words[i]:scores[i] for i in range(len(words))}
    return tfidf_dict


def get_freq_score(articles):
    print("Getting Frequency scores")
    corpus = [" ".join(art) for art in articles]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    X = X.toarray()
    scores = np.max(X, axis=0)
    words = vectorizer.get_feature_names()
    freq_dict = {words[i]:scores[i] for i in range(len(words))}
    return freq_dict


def prune_vocabulary(articles, vocab_len=10000):
    scores = get_tfidf_score(articles)
    valid_words = [w for _,w in sorted(zip(scores.values(), scores.keys()), reverse=True)]
    valid_words = valid_words[:vocab_len]
    articles_pruned = []
    print("Pruning articles")
    for art in articles:
        pruned_art = list(set(art).intersection(set(valid_words))) #[a for a in art if a in valid_words]
        articles_pruned.append(pruned_art)
    return articles_pruned


def prepare_dataset(with_adverts=True, with_classifieds=True):
    print("Preparing French articles")
    # paths = ['/wrk/group/newseye/corpora/bnf/Oeuvre_1910_1920/text_files/',
    #          '/wrk/group/newseye/corpora/bnf/Excelsior_1910_1920/text_files/']
    paths = ['/wrk/group/newseye/corpora/bnf/Oeuvre_1910_1920/text_files/']
    advert_label = 'Publicité'
    classified_label = 'Annonces'
    articles_dict = {}
    for path in paths:
        print("Newspaper: ", path)
        files = os.listdir(path)
        files.sort()
        for f in files:
            print(f)
            data = pickle.load(open(path+f,'rb'))
            for art in data:
                art_date = art['date'].split('-')[0]
                art_content = clean(art['content'])
                art_label = art['LABEL']
                if len(art_date) > 0 and len(art_content) > 50:
                    if art_date not in articles_dict:
                        articles_dict[art_date] = []
                    if (art_label==advert_label and with_adverts==True):
                        articles_dict[art_date].append(art_content)
                    elif (art_label==classified_label and with_classifieds==True):
                        articles_dict[art_date].append(art_content)
                    elif (art_label != advert_label and art_label != classified_label):
                        articles_dict[art_date].append(art_content)
    return articles_dict


def prepare_demonstrator_dataset():
    print("Preparing French articles from Demonstrator")
    path = "/wrk/group/newseye/corpora/demonstrator_data_dump/newseye_articles_all_languages.json"
    articles = []
    data = json.load(open(path,'rb'))
    for key in data:
        if data[key]['lang'] == 'fr':
            articles.append(data[key]['text'])
    return articles


def get_random_articles(articles, art_per_year=1000):
    years = sorted(list(articles.keys()))
    random_articles = []
    timeslices = []
    for year in years:
        articles_year = articles[year]
        n_articles = len(articles_year)
        if n_articles > art_per_year:
            indexes = [random.randint(0, n_articles-1) for _ in range(art_per_year)]
            sampled_articles = [articles_year[i] for i in indexes]
            random_articles.extend(sampled_articles)
            timeslices.append(art_per_year)
        else:
            random_articles.extend(articles_year)
            timeslices.append(n_articles)
    return random_articles, timeslices


def get_random_demonstrator_articles(articles, max_articles=10000):
    random_articles = []
    if len(articles) < max_articles:
        random_articles = articles
    else:
        indexes = np.array([random.randint(0, len(articles)) for _ in range(max_articles)])
        random_articles = [articles[i] for i in indexes]
    return random_articles


def train_lda(articles, filename, k = 10):
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(d) for d in articles]
    print("All articles: ", str(len(articles)))
    print("Topics: ", k)
    print("Vocab len: ", len(common_dictionary))
    print("Training LDA...")
    lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=k, passes=1000)
    model_file = filename
    lda.save(model_file)
    dict_filename = model_file + "_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_file + "_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))
    print("Saved trained LDA model as", model_file, "!")


def train_dtm(articles, timeslices, filename, k = 10):
    chain_var = 0.1
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(a) for a in articles]
    print("Training DTM for all articles")
    print("Articles:", len(articles))
    print("Topics:", k)
    print("Time slices: ", timeslices)
    print("Vocab len: ", len(common_dictionary))
    ldaseq = LdaSeqModel(corpus=common_corpus, time_slice=timeslices,
                         num_topics=k, id2word=common_dictionary,
                         chain_variance=chain_var)
    model_file = filename
    ldaseq.save(model_file)
    dict_filename = model_file+"_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_file+"_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))
    print("Saved DTM model as", model_file, "!")

