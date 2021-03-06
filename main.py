from hazm import *
from operator import truediv

import pandas as pd
from pandas import ExcelWriter
from numpy import Inf
import numpy as np 
import math
from collections import Counter

import multiprocessing
import pickle
from gensim.models import Word2Vec
from numpy.linalg import norm


ALL_DOC_NUM = 80
# ALL_DOC_NUM = 7561
# ALL_DOC_NUM = 11437
# ALL_DOC_NUM = 50060

def normalize(txt):
    nmz = Normalizer()
    return nmz.normalize(txt)



# read excel file, normalize them and save to another file 
def read_normalize_write(read_file_name, write_file_name):

    df_read = pd.read_excel(read_file_name)
    # print("Column headings: ", df_read.columns)

    df_write = pd.DataFrame()

    for i in df_read.index:
        print("Normalizing; docID: ", i)

        # normalize content and save all data to a new file
        normalized_content = normalize(df_read['content'][i])

        df_new = pd.DataFrame({'id':[df_read['id'][i]],
                        'content':[normalized_content],
                        'topic': [df_read['topic'][i]],
                        'url': [df_read['url'][i]]})

        df_write = df_write.append(df_new)

    writer = ExcelWriter(write_file_name)
    df_write.to_excel(writer,'Sheet1',index=False)
    writer.save()



# read stop words from file and save them in a list
def load_stop_words(path):
    my_file = open(path, "r")
    content_list = my_file.readlines()
    my_set = set()
    for word in content_list:
        my_set.add(normalize(word.rstrip('\n')))
    return my_set



# preprocess our documentation; tokenization, stemming, removing stop words, lemmatization
def tokenizer(read_file_name):

    stemmer = Stemmer()
    lemmatizer = Lemmatizer()

    # stop_words_list = load_stop_words("../persian-stopwords/persian")
    # stop_words_list = load_stop_words("../persian-stopwords/chars")
    # stop_words_list = load_stop_words("../persian-stopwords/nonverbal")
    # stop_words_list = load_stop_words("../persian-stopwords/out")
    stop_words_list = []

    doc_word = {}

    df_read = pd.read_excel(read_file_name)

    for i in df_read.index:
    # for i in range (100):
        print("Preprocessing; docID: ", i)
        list = word_tokenize(df_read['content'][i])
        # print("Before: ", len(list))
        
        processed_list = []
        for word in list:
            # stemming
            tmp1 = stemmer.stem(word)
            # check for stop word, if not append to word list
            if tmp1 not in stop_words_list:
                # lemmatizing
                tmp2 = lemmatizer.lemmatize(tmp1)
                # append to list
                processed_list.append(tmp2)

        doc_word [i] = processed_list

        # print("After: ", len(processed_list))
        
    return doc_word



# get list of doc and all words in each of them and create positional index
def create_positional_index(doc_word):
    print("Creating positional index ...")

    positional_index = {}
    all_word = []
    i = 0
    LIMIT = Inf
    # LIMIT = 100

    for doc_Id in doc_word:
        if i < LIMIT:
            i += 1

            for index in range(len(doc_word[doc_Id])):
                term = doc_word[doc_Id][index]
                
                all_word.append(term)            

                if term in positional_index:
                    
                    # increase frequency
                    positional_index[term][0] += 1

                    # if term exist in doc ID
                    if doc_Id in positional_index[term][1]:
                        positional_index[term][1][doc_Id].append(index)
                            
                    else:
                        positional_index[term][1][doc_Id] = [index]

                else:

                    positional_index[term] = []
                    # set frequency to 1
                    positional_index[term].append(1)
                    positional_index[term].append({})
                    # add doc ID to postings list  
                    positional_index[term][1][doc_Id] = [index]

    # print("all term: ", len(all_word))
    # print("all unique term: ", len(remove_duplicate_preserve_order(all_word)))

    # print(positional_index)
    print("Done ...")
    return positional_index



# read excel file and save ID and title of news
def read_doc_title(read_file_name):
    df_read = pd.read_excel(read_file_name)

    doc_ID_title = {}
    for i in df_read.index:
        doc_ID_title [i] = [df_read["url"][i], df_read["topic"][i]]
    
    return doc_ID_title



# give reuslt list and remove duplicate answer, but preserve order
def remove_duplicate_preserve_order(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



def find_one_term(term, positional_index):
    query_answer_one = []

    if term in positional_index:
        result_list = positional_index[term][1].keys()
        for doc_ID in result_list:
            query_answer_one.append(doc_ID)

    return query_answer_one



def find_two_term_with_order(term1, term2, positional_index):

    query_answer_two = []
    # if both of them in one doc
    if term1 in positional_index and term2 in positional_index:
        list1 = positional_index[term1][1]
        list2 = positional_index[term2][1]

        for doc_ID1, position1 in list1.items():
            for doc_ID2, position2 in list2.items():
                if doc_ID1 == doc_ID2:
                    if position1[0] + 1 == position2[0]:
                        query_answer_two.append(doc_ID1)

    return query_answer_two



def find_two_term_without_order(term1, term2, positional_index):

    query_answer_two = []
    # if both of them in one doc
    if term1 in positional_index and term2 in positional_index:
        list1 = positional_index[term1][1]
        list2 = positional_index[term2][1]

        for doc_ID1, position1 in list1.items():
            for doc_ID2, position2 in list2.items():
                if doc_ID1 == doc_ID2:
                    if position1[0] + 1 != position2[0]:
                        query_answer_two.append(doc_ID1)

    return query_answer_two



# boolean retrieval
def boolean_search(positional_index, sentense):
    tokenized_query = word_tokenize(sentense)
    
    # print(len(tokenized_query))
    # print (tokenized_query)

    query_answer = []

    if (len(tokenized_query) == 1):
        query_answer_one = find_one_term(tokenized_query[0], positional_index)
        query_answer.extend(query_answer_one)


    elif len(tokenized_query) == 2:
        
        query_answer.extend(find_two_term_with_order(tokenized_query[0], tokenized_query[1], positional_index))
        query_answer.extend(find_two_term_without_order(tokenized_query[0], tokenized_query[1], positional_index))

        query_answer.extend(find_one_term(tokenized_query[0], positional_index))
        query_answer.extend(find_one_term(tokenized_query[1], positional_index))


    elif len(tokenized_query) == 3:
        query_answer_two1 = find_two_term_with_order(tokenized_query[0], tokenized_query[1], positional_index)
        query_answer_two2 = find_two_term_with_order(tokenized_query[1], tokenized_query[2], positional_index)
        
        # print(query_answer_two1)
        # print(query_answer_two2)

        query_answer_three = list(set(query_answer_two1) & set(query_answer_two2))
        query_answer.extend(query_answer_three)

        # print(query_answer_three)

        query_answer.extend(query_answer_two1)
        query_answer.extend(query_answer_two2)

        query_answer.extend(find_one_term(tokenized_query[0], positional_index))
        query_answer.extend(find_one_term(tokenized_query[1], positional_index))
        query_answer.extend(find_one_term(tokenized_query[2], positional_index))


    return query_answer



# create champion list from positional index
def create_champion_list(positional_index, r=50):
    print("Creating champion list ...")

    champion_list = {}

    for term in positional_index:
        champion_list[term] = []

        champion_list[term].append(len (positional_index[term][1]))
        champion_list[term].append({})

        # if number of doc for term is less than our limitation (r), put all doc id champion list for that term.
        if len (positional_index[term][1]) <= r:

            for doc_Id in positional_index[term][1]:
                champion_list[term][1][doc_Id] = len(positional_index[term][1][doc_Id])
        
        # if number of doc for term is more than our limitation (r), just put doc with the most term.
        else:
            term_freq_list = []
            for doc_Id in positional_index[term][1]:
                term_freq_list.append(len(positional_index[term][1][doc_Id]))

            # find term freq of word in r rank, and just save word with more than this freq
            term_freq_list.sort(reverse=True)
            max_term_freq = term_freq_list[r]

            for doc_Id in positional_index[term][1]:
                if len(positional_index[term][1][doc_Id]) >= max_term_freq:
                    champion_list[term][1][doc_Id] = len(positional_index[term][1][doc_Id])

    print("Done ...")
    return champion_list



# calculate length of all docs vector 
def calculate_lengths(champion_list):
    print("Calculating length ...")

    lengths = [0] * (ALL_DOC_NUM + 1)

    for term in champion_list:
        for doc_ID in champion_list[term][1]:
            lengths [doc_ID] += (1 + math.log10(champion_list[term][1][doc_ID])) * (math.log10(ALL_DOC_NUM/champion_list[term][0]))

    print("Done ...")
    return np.sqrt(lengths)



# ranked retrieval
def ranked_search(champion_list, sentense, lengths):
    tokenized_query = word_tokenize(sentense)
    unique_tokenized_query = Counter(tokenized_query)
    # print (unique_tokenized_query)

    scores = [0] * (ALL_DOC_NUM + 1)

    for term in unique_tokenized_query:
        # do not calculate score for term that are not in
        if term in champion_list:
            weight_tq = (1 + math.log10(unique_tokenized_query[term])) * (math.log10(ALL_DOC_NUM/champion_list[term][0]))
        
            posting = champion_list[term][1]

            for doc_ID in posting:
                weight_td = (1 + math.log10(posting[doc_ID])) * (math.log10(ALL_DOC_NUM/champion_list[term][0]))
                scores[doc_ID] += weight_tq * weight_td

    
    # divide all calculated score to their length
    normlized_scores = list(map(truediv, scores, lengths))

    # sort scores and return their index 
    list2 = sorted(range(len(normlized_scores)), key=lambda k: normlized_scores[k])
    list2.reverse()

    # print highest scores 
    # print (normlized_scores[list2[0]])
    # print (normlized_scores[list2[1]])
    # print (normlized_scores[list2[2]])

    return list2



# store training data with pickle 
def store_training_data(training_data, obj_file_name):

    data_file = open(obj_file_name, 'ab')
    pickle.dump(training_data, data_file)                     
    data_file.close()



# train model with our training data and save
def train_model(obj_file_name, model_file_name):

    # open training data file
    handle = open(obj_file_name, 'rb')
    training_data = pickle.load(handle)
    

    docs_num = len (training_data)
    tokens_num = sum([len(x) for x in training_data])
    print("docs: ", docs_num)
    print("tokens: ", tokens_num)

    cores = multiprocessing.cpu_count()
    w2v_model = Word2Vec(min_count = 1, window = 5, vector_size = 300, alpha = 0.03, workers = cores - 1)

    w2v_model.build_vocab(training_data)
    w2v_model_vocab_size = len(w2v_model.wv)
    print("unique vocabs: ", w2v_model_vocab_size)

    w2v_model.train(training_data, total_examples = w2v_model.corpus_count, epochs = 20)

    w2v_model.save(model_file_name)



# for all term in all docs, calculate tf idf weight and store in a list contains of a dictionary of {term: weight}
def calculate_tf_idf(doc_word_list, champion_list):
    docs_tf_idf = []

    for doc_ID, terms in doc_word_list.items():
        unique_terms = Counter(terms)

        term_weight_dic = {}
        
        for term in unique_terms:
            if term in champion_list:
                posting = champion_list[term][1]
                term_weight_dic[term] = (1 + math.log10(posting[doc_ID])) * (math.log10(ALL_DOC_NUM/champion_list[term][0]))

        docs_tf_idf.append(term_weight_dic)

    return docs_tf_idf



# calculate docs embedding
def embed (model_file_name, docs_tf_idf):

    w2v_model = Word2Vec.load(model_file_name)

    docs_embedding = []

    for doc in docs_tf_idf:
        doc_vec = np.zeros(300)
        weights_sum = 0
        for token, weight in doc.items():
            if token in w2v_model.wv:
                doc_vec += w2v_model.wv[token] * weight
                weights_sum += weight

        docs_embedding.append(doc_vec/weights_sum)

    return docs_embedding


# find similarity of two doc and return in a number between 0 to 1
def similarity(doc1, doc2):
    similarity_score = np.dot(doc1, doc2) / (norm(doc1) * norm(doc2))
    return (similarity_score + 1)/2



# get query sentence and calculate tf idf weight for its term
def calculate_query_tf_idf(sentense):
    print("query:", sentense)
    
    tokenized_query = word_tokenize(sentense[1:])
    unique_tokenized_query = Counter(tokenized_query)

    term_weight_dic = {}

    for term in unique_tokenized_query:
        if term in champion_list:
            term_weight_dic[term] = (1 + math.log10(unique_tokenized_query[term])) * (math.log10(ALL_DOC_NUM/champion_list[term][0]))

    return term_weight_dic



# compare query with all docs and find the most similiae ones
def enhanced_ranked_search(docs_embedding, query_embedding):

    scores = [0] * (ALL_DOC_NUM + 1)

    for i in range (0, ALL_DOC_NUM):
        scores[i] = similarity(docs_embedding[i], query_embedding)
    
    # sort scores and return their index 
    list2 = sorted(range(len(scores)), key=lambda k: scores[k])
    list2.reverse()

    # print highest scores 
    # print (normlized_scores[list2[0]])
    # print (normlized_scores[list2[1]])
    # print (normlized_scores[list2[2]])

    return list2



# find minimum euclidean distance between a doc and all centroids to calculate clusters.
def calculate_clusters(docs_embedding, centroids, k):
    clusters = {}
    for i in range(k):
        clusters[i] = []

    for i in range(len(docs_embedding)):
        euc_dist = []
        for j in range(k):
            euc_dist.append(np.linalg.norm(docs_embedding[i] - centroids[j]))
            # euc_dist.append(math.dist(data, centroids[j]))
        # Append the cluster of data to the dictionary
        clusters[euc_dist.index(min(euc_dist))].append([i, docs_embedding[i]])
        # print(clusters)
    return clusters



# calculate centroids 
def recalculate_centroids(centroids, clusters, k):
    for i in range(k):
        tmp = np.average(clusters[i], axis=0)
        # print('*******************************************')
        # print (tmp)
        # print(i)

        # temprory: handle Not A Number error
        if not np.isnan(tmp[1][0]):
            centroids[i] = tmp[1]
    
    return centroids



# run k means algorithm 
def run_k_means(k, docs_embedding):
    centroids = {}
    # initialize centroid with random numbers
    for i in range(k):
        centroids[i] = docs_embedding[i]
 
    clusters =  calculate_clusters(docs_embedding, centroids, k)
    centroids = recalculate_centroids(centroids, clusters, k)

    # recalculate clusters and then centroids.
    print("Recalculating clusters and centroids ...")
    for i in range(10):
        clusters =  calculate_clusters(docs_embedding, centroids, k)
        centroids = recalculate_centroids(centroids, clusters, k)
    print("Done ...")

    return centroids, clusters


# first select "b" of the most similar centroids and then rank result among these clusters.
def k_means_search(k_number, b, clusters, centroids, query_embedding, translate, category):

    print("category: ", category)

    centroids_scores = [0] * k_number
    for i in range (0, k_number):
        centroids_scores[i] = similarity(centroids[i], query_embedding)
    
    # sort scores and return their index 
    list2 = sorted(range(len(centroids_scores)), key=lambda k: centroids_scores[k])
    list2.reverse()

    list2_cat = []
    for i in list2:
        if translate[i] == category:
            list2_cat.append(i)

    # chosse the most similiar clusters from their centroids.
    selected_cluster_index = list2_cat[0: b]
    selected_cluster_item = []

    # concat all selected cluster
    for i in selected_cluster_index:
        selected_cluster_item += clusters[translate[i]]
        print(translate[i])

    scores = [0] * ALL_DOC_NUM
    for i in range (0, len(selected_cluster_item)):
        # scores[i] = similarity(selected_cluster_item[i][1], query_embedding)
        # print(i)
        # print(selected_cluster_item[i])
        scores[selected_cluster_item[i][0]] = similarity(selected_cluster_item[i][1], query_embedding)

    # sort scores and return their index 
    list3 = sorted(range(len(scores)), key=lambda k: scores[k])
    list3.reverse()

    return list3



# label clusters
def label_classes(clusters):
    translate = {}

    for key, value in clusters.items():
        
        topics = {}
        for doc_id in value:
            # docs in each cluster
            # print(doc_id[0])
            topic = doc_ID_title[doc_id[0]][1]
        
            if topic not in topics:
                topics[topic] = 1
            else:
                topics[topic] += 1

        # print(topics)
        label = max(topics, key=topics.get)
        translate[key] = label

        # selected label for cluster
        # print(label)

    # change old with new keys
    for old, new in translate.items():
        clusters[new] = clusters.pop(old)

    return translate


if __name__ == "__main__":

    # read_normalize_write('IR00_dataset_ph3/IR00_3_11k News.xlsx', 'IR00_dataset_ph3/IR00_3_11k News_normalized.xlsx')
    # read_normalize_write('IR00_dataset_ph3/IR00_3_17k News.xlsx', 'IR00_dataset_ph3/IR00_3_17k News_normalized.xlsx')
    # read_normalize_write('IR00_dataset_ph3/IR00_3_20k News.xlsx', 'IR00_dataset_ph3/IR00_3_20k News_normalized.xlsx')
    # importing the required modules
    # excl_list = []    
    # excl_list.append(pd.read_excel('IR00_dataset_ph3/IR00_3_11k News_normalized.xlsx'))
    # excl_list.append(pd.read_excel('IR00_dataset_ph3/IR00_3_17k News_normalized.xlsx'))
    # excl_list.append(pd.read_excel('IR00_dataset_ph3/IR00_3_20k News_normalized.xlsx'))
    # excl_merged = pd.concat(excl_list, ignore_index=True)
    # excl_merged.to_excel('IR00_dataset_ph3/IR00_3_48k News_normalized_merged.xlsx', index=False)

    # doc_word = tokenizer('IR1_7k_news_normalized.xlsx')
    # doc_word = tokenizer('IR00_dataset_ph3/IR00_3_48k News_normalized_merged.xlsx')
    # doc_word = tokenizer('IR00_dataset_ph3/IR00_3_11k News_normalized.xlsx')
    doc_word = tokenizer('IR00_dataset_ph3/IR00.xlsx')
    # data_file = open('tmp_50k_obj', 'ab')
    # pickle.dump(doc_word, data_file)          
    # data_file.close()
    # handle = open('tmp_50k_obj', 'rb')
    # doc_word = pickle.load(handle)

    doc_ID_title = read_doc_title('IR00_dataset_ph3/IR00.xlsx')
    # doc_ID_title = read_doc_title('IR00_dataset_ph3/IR00_3_48k News_normalized_merged.xlsx')
    # doc_ID_title = read_doc_title('IR00_dataset_ph3/IR00_3_11k News_normalized.xlsx')
    positional_index = create_positional_index(doc_word)

    # number_of_doc_for_each_term = 100
    number_of_doc_for_each_term = Inf
    champion_list = create_champion_list(positional_index, number_of_doc_for_each_term)

    lengths = calculate_lengths(champion_list)

    obj_file_name = 'my_training_data.obj'
    # model_file_name = "my_files/my_model.model"
   
    # store_training_data(list(doc_word.values()), obj_file_name)
    # train_model(obj_file_name, model_file_name)
   
    model_file_name = "word2vec_model_hazm/w2v_150k_hazm_300_v2.model"
    docs_tf_idf  = calculate_tf_idf(doc_word, champion_list)
    docs_embedding = embed (model_file_name, docs_tf_idf)

    
    # k means
    k, b = 5, 1
    # clusters: {cluster_id: [[doc_id, data], ...]}
    centroids, clusters = run_k_means(k, docs_embedding)
    # print(clusters)

    translate = label_classes(clusters)


    # answer queries
    while True:
        sentense = input("Enter your query: ")
        # query_answer = boolean_search (positional_index, sentense)
        # query_answer = ranked_search (champion_list, sentense, lengths)

        query_tf_idf = calculate_query_tf_idf(sentense.partition(' ')[2])
        query_embedding = embed (model_file_name, [query_tf_idf])
        # query_answer = enhanced_ranked_search (docs_embedding, query_embedding[0])

        query_answer = k_means_search(k, b, clusters, centroids, query_embedding[0], translate, sentense.partition(' ')[0])

        counter = 0
        # remove duplicate docID
        for item in remove_duplicate_preserve_order(query_answer):
            # doc_ID_title : {id : [url, topic]}
            print (counter+1, ") docID: ", item, "| title: ", doc_ID_title[item][0])
            counter += 1
            if counter == 10:
                break



# dont forget to choose stop_word_list in line 65 ...

# positional index example: 'word': [4, {80: [1432], 97: [132, 159, 357]}] 
# champion list example:    'word': [2, {80: 1, 97: 3}]
# champion list example:    'word': [doc freq, {doc id: term freq, doc id: term frerq}]

# I used this website for the k-means algorithm
# https://towardsdatascience.com/k-means-without-libraries-python-feb3572e2eef