from __future__ import print_function, unicode_literals
from re import L, search
import re
from hazm import *
from numpy import Inf

import pandas as pd
from pandas import ExcelWriter



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

        df_new = pd.DataFrame({'content':[normalized_content],
                        'url':[df_read['url'][i]],
                        'title': [df_read['title'][i]]})

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
        print("Before: ", len(list))
        
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

        print("After: ", len(processed_list))
        
    return doc_word



# get list of doc and all words in each of them and create positional index
def create_positional_index(doc_word):

    positional_index = {}
    all_word = []
    i = 0
    LIMIT = Inf

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
    return positional_index



# read excel file and save ID and title of news
def read_doc_title(read_file_name):
    df_read = pd.read_excel(read_file_name)

    doc_ID_title = {}
    for i in df_read.index:
        doc_ID_title [i] = df_read["title"][i]
    
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



def search(positional_index, sentense, doc_ID_title):
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


    counter = 0
    # remove duplicate docID
    for item in remove_duplicate_preserve_order(query_answer):
        print (counter+1, ") docID: ", item, "| title: ", doc_ID_title[item])
        counter += 1
        if counter == 10:
            break


if __name__ == "__main__":
    
    read_normalize_write('IR1_7k_news.xlsx', 'IR1_7k_news_normalized.xlsx')
    doc_word = tokenizer('IR1_7k_news_normalized.xlsx')
    doc_ID_title = read_doc_title('IR1_7k_news_normalized.xlsx')
    positional_index = create_positional_index(doc_word)

    while True:
        sentense = input("Enter your query: ")
        search (positional_index, sentense, doc_ID_title)
