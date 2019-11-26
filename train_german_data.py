from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
from gensim.models import LdaModel, LdaSeqModel, LdaMulticore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os
import string
import pickle
import random
import re
from collections import Counter
from nltk import word_tokenize

stop_de = set(stopwords.words('german') + ['sind','ist','sei','sein','hat','haben','hatte','habe',
                                           'war','wurde','worden','werden','wird','habe','hätte',
                                           'wäre','kann','konnte','können','sollen','müssen','muß',
                                           'ein','eine','ver','mehr'])
exclude = set(string.punctuation +'?»«')
lemma = WordNetLemmatizer()


def filter_tokens(doc):
    tokens = word_tokenize(doc.lower())
    clean_stop = " ".join([i for i in tokens if i not in stop_de])
    clean_punc = ''.join(ch if ch not in exclude else ' ' for ch in clean_stop)
    clean_tokens = [lemma.lemmatize(word) for word in clean_punc.split()]
    clean_tokens = [tok for tok in clean_tokens if len(tok)>3 and tok not in stop_de]
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



def prune_vocabulary(articles, vocab_len=5000):
    scores = get_tfidf_score(articles)
    valid_words = [w for _,w in sorted(zip(scores.values(), scores.keys()), reverse=True)]
    valid_words = valid_words[:vocab_len]
    articles_pruned = []
    print("Pruning articles")
    for art in articles:
        pruned_art = list(set(art).intersection(set(valid_words))) #[a for a in art if a in valid_words]
        articles_pruned.append(pruned_art)
    return articles_pruned


def prepare_dataset():
    path = '/wrk/group/newseye/corpora/arbeiter_zeitung/'
    files = os.listdir(path)
    files.sort()
    articles = []
    dates = []
    vocab = set()
    total_size = 0
    for f in files:
        print(f)
        doc = open(path+f,'rb').read().decode('utf-8').split()
        date_pub = doc[1].split("-")[0]
        dates.append(date_pub)
        doc_str = " ".join(doc[2:])
        clean_doc = filter_tokens(doc_str)
        articles.append(clean_doc)
        vocab.update(set(clean_doc))
    total_size = np.sum([len(a) for a in articles])
    mean_art_len = np.mean([len(a) for a in articles])
    print("Start date: ", min(dates))
    print("End date: ", max(dates))
    print("Articles: ", len(articles))
    print("Mean article length: ", mean_art_len)
    print("Total no. of tokens: ", total_size)
    counts = Counter(dates)
    print("Vocab size:", len(vocab))
    print("Date distribution: ")
    years_sorted = list(counts.keys())
    years_sorted.sort()
    for year in years_sorted:
        print(year,":",counts[year])
    return articles, dates


def prepare_dataset_decade():
    path = '/wrk/group/newseye/corpora/arbeiter_zeitung/'
    files = os.listdir(path)
    files.sort()
    articles_dict = {}
    vocab = set()
    total_size = 0
    for f in files:
        print(f)
        doc = open(path+f,'rb').read().decode('utf-8').split()
        date_pub = int(doc[1].split("-")[0])
        decade = int(date_pub - (date_pub % 10))
        doc_str = " ".join(doc[2:])
        clean_doc = filter_tokens(doc_str)
        if decade not in articles_dict.keys():
            articles_dict[decade] = []
        articles_dict[decade].append(clean_doc)
        vocab.update(set(clean_doc))
    print("Start decade: ", min(articles_dict.keys()))
    print("End decade: ", max(articles_dict.keys()))
    print("Date distribution: ")
    for decade in sorted(list(articles_dict.keys())):
        print(decade,":",len(articles_dict[decade]))
    return articles_dict


def prepare_dataset_1918():
    path = '/wrk/group/newseye/corpora/onb_1918_text/'
    files = os.listdir(path)
    files = [f for f in files if ".txt" in f]
    files.sort()
    articles_dict = {}
    vocab = set()
    art_id = 0
    for f in files:
        print(f)
        date_pub = f.split("__")[2]
        doc = open(path+f,'rb').read().decode('utf-8')
        articles = re.split(r'#{100}', doc)
        for art in articles:
            clean_art = filter_tokens(art)
            # only include articles with more than 100 tokens
            if len(clean_art) > 200:
                vocab.update(set(clean_art))
                articles_dict[art_id] = {}
                articles_dict[art_id]['date'] = date_pub
                articles_dict[art_id]['text'] = clean_art
                art_id += 1
    print("No. of articles: ", len(articles_dict))
    print("Original vocab: ", len(vocab))
    return articles_dict


def slice_dataset_monthly_1918(articles_dict):
    print("Dividing dataset into months")
    articles_monthly = {}
    for art_id in articles_dict:
        art_text = articles_dict[art_id]['text']
        art_date = articles_dict[art_id]['date']
        art_month = int(art_date.split("-")[1])
        if art_month not in articles_monthly:
            articles_monthly[art_month] = []
        articles_monthly[art_month].append(art_text)
    print("Time slices:", len(articles_monthly))
    return articles_monthly


def slice_dataset_daily_1918(articles_dict):
    print("Dividing dataset into days")
    articles_daily = {}
    for art_id in articles_dict:
        art_text = articles_dict[art_id]['text']
        art_date = articles_dict[art_id]['date']
        art_day = int(art_date.split("-")[1]+art_date.split("-")[2])
        if art_day not in articles_daily:
            articles_daily[art_day] = []
        articles_daily[art_day].append(art_text)
    print("Time slices: ", len(articles_daily))
    return articles_daily


def take_random_articles(articles, dates, n_articles = 1500):
    max_art = len(articles)-1
    indexes = np.array([random.randint(0,max_art) for _ in range(n_articles)])
    art_array = np.array(articles)
    dates_array = np.array(dates)
    random_art = list(art_array[indexes])
    random_dates = list(dates_array[indexes])
    sorted_articles = [a for d, a in sorted(zip(random_dates, random_art))]
    random_dates.sort()
    return sorted_articles, random_dates


def train_lda(articles, n_topics):
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(d) for d in articles]
    print("Documents: ", str(len(articles)))
    print("Topics: ", n_topics)
    print("Training LDA...")
    lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=n_topics, passes=1000)
    model_file = "trained_models/arb_zeit/lda_1500"
    lda.save(model_file)
    dict_filename = model_file + "_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_file + "_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))
    print("Saved trained LDA model as", model_file, "!")


def train_lda_multicore(articles, n_topics, outfile="lda", workers=3):
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(d) for d in articles]
    print("Documents: ", str(len(articles)))
    print("Vocabulary: ", len(common_dictionary))
    print("Topics: ", n_topics)
    print("Training LDA...")
    lda = LdaMulticore(common_corpus, id2word=common_dictionary, num_topics=n_topics, workers=workers)

    model_file = "trained_models/"+outfile
    lda.save(model_file)
    dict_filename = model_file + "_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_file + "_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))
    print("Saved trained LDA model as", model_file, "!")


def train_dtm(articles, n_topics, outfile="dtm", dates=None, time_slices=None):
    if time_slices is None and dates is not None:
        counts = Counter(dates)
        time_slices = list(counts.values())
        print("Dates: ", counts)
    print("Time slices:", time_slices)
    chain_var = 0.1
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(a) for a in articles]
    ldaseq = LdaSeqModel(corpus=common_corpus, time_slice=time_slices,
                         num_topics=n_topics, id2word=common_dictionary,
                         chain_variance=chain_var)
    model_file = "trained_models/"+outfile
    ldaseq.save(model_file)
    dict_filename = model_file+"_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_file+"_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))

    print("Saved DTM model as", model_file, "!")

