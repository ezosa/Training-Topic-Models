from gensim import corpora
from gensim.models import LdaSeqModel
from gensim.models import LdaModel
from nltk.corpus import stopwords
from collections import Counter
from nltk import word_tokenize
import numpy as np
import os
import string
import pickle
import tarfile
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


exclude = set(string.punctuation)
stopwords_file = '../data/stopwords_fi_freq'
stop_list = open(stopwords_file, 'rb').read().decode('utf-8').split("\n")
sw = [s.split()[1].lower() for s in stop_list if len(s.split()) > 1]
stopwords_fi = set(stopwords.words('finnish'))
stopwords_fi.update(sw)
print("Finnish stopwords:", stopwords_fi)



def clean_document(doc):
    clean_punc = ''.join(ch if ch not in exclude else '' for ch in doc)
    clean_punc_tokens = word_tokenize(clean_punc)
    clean_stop = [tok for tok in clean_punc_tokens if tok not in stopwords_fi and len(tok) > 3]
    #clean_digits = [tok for tok in clean_stop if re.match(r'^([\s\d]+)$', tok) is None]
    return clean_stop


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
    scores = get_freq_score(articles)
    valid_words = [w for _,w in sorted(zip(scores.values(), scores.keys()), reverse=True)]
    valid_words = valid_words[:vocab_len]
    articles_pruned = []
    print("Pruning articles")
    for art in articles:
        pruned_art = list(set(art).intersection(set(valid_words))) #[a for a in art if a in valid_words]
        articles_pruned.append(pruned_art)
    return articles_pruned


def filter_rare_tokens(all_docs, min_count=100):
    print("Filtering tokens appearing less than", min_count, "times")
    all_tokens = [token for doc in all_docs for token in doc]
    counts = Counter(all_tokens)
    valid_tokens = [token for token in counts.keys() if counts[token] >= min_count]
    filtered_docs = []
    for i in range(len(all_docs)):
        doc = all_docs[i]
        filtered = [token for token in doc if token in valid_tokens and len(token) > 3]
        filtered_docs.append(filtered)
    return filtered_docs


def sample_articles_per_year(documents_sorted, years, docs_per_year=100):
    documents_sampled = []
    for y, year in enumerate(years):
        random_indexes = list(np.random.choice(range(0, len(documents_sorted[y])), docs_per_year, replace=False))
        documents_year = [documents_sorted[y][i] for i in random_indexes]
        documents_sampled.append(documents_year)
    new_timeslice = [len(documents_sampled[y]) for y in range(len(years))]
    documents_sampled_flat = [doc for documents_year in documents_sampled for doc in documents_year]
    return documents_sampled_flat, new_timeslice


def get_articles(filepath):
    suometar_bow = {}
    tar_files = os.listdir(filepath)
    print("Reading lemmatized articles from:", filepath)
    for tar_file in tar_files:
        tar = tarfile.open(filepath + tar_file, "r")
        for member in tar.getmembers():
            f = tar.extractfile(member)
            if f is not None:
                filename = member.name
                #print("Filename: ", filename)
                filename = filename.split("_")
                year = filename[1].split("-")[0]
                if year not in suometar_bow.keys():
                    #print("Add year: ", str(year))
                    suometar_bow[year] = []
                articles = f.read().decode("utf-8").lower()
                suometar_bow[year].append(clean_document(articles))
    # sanity check
    print("Unsorted years: ", str(suometar_bow.keys()))
    suometar_years = sorted(suometar_bow.keys())
    print("\nSorted years: ", str(suometar_years))
    return suometar_bow


def train_lda(articles, n_topics, model_filename):
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(d) for d in articles]
    print("Documents: ", str(len(articles)))
    print("Topics: ", n_topics)
    print("Vocab:", len(common_dictionary))
    print("Training LDA...")
    lda = LdaModel(common_corpus, id2word=common_dictionary, num_topics=n_topics, passes=1000)
    lda.save(model_filename)
    dict_filename = model_filename + "_dict.pkl"
    pickle.dump(common_dictionary, open(dict_filename, "wb"))
    dict_filename = model_filename + "_corpus.pkl"
    pickle.dump(common_corpus, open(dict_filename, "wb"))
    print("Saved trained LDA model as", model_filename, "!")


def train_dtm(articles, n_topics, timeslices, model_filename):
    # train DTM
    print("\nCreating common_corpus and common_dictionary")
    common_dictionary = corpora.Dictionary(articles)
    common_corpus = [common_dictionary.doc2bow(doc) for doc in articles]
    chain_var = 0.1
    print("Size of vocabulary: ", str(len(common_dictionary)))
    print("\nLDASeq Params:")
    print("time_slice = ", str(timeslices))
    print("n_topics = ", str(n_topics))
    print("chain_var = ", str(chain_var))
    print("\nStart training Ldaseq model")
    ldaseq = LdaSeqModel(corpus=common_corpus,
                         time_slice=timeslices,
                         num_topics=n_topics,
                         id2word=common_dictionary,
                         chain_variance=chain_var)

    model_file = model_filename
    ldaseq.save(model_file)
    print("***** Done training! Saved trained model as", model_file, "*****")
    # save common_dictionary
    f = open(model_file + "_dict.pkl", "wb")
    pickle.dump(common_dictionary, f)
    f.close()
    print("Saved common_dictionary!")
    # save common_corpus
    f = open(model_file + "_corpus.pkl", "wb")
    pickle.dump(common_corpus, f)
    f.close()
    print("Saved common_corpus!")
