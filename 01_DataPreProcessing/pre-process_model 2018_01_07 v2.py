# -*- coding: utf-8 -*-
from __future__ import division

#from __future__ import print_function
from time import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import pandas as pd
import os  # operating system commands

import math
from textblob import TextBlob as tb

# Test


####################################################################################################
####################################################################################################
#
# This script will serve as a pre-processor in text analytics, aimed to build framework for running topic models
# Therefore, it will perform several functions as follows:
#    1) load filedsfrom various DSI sources, save to a standardized format (organize the corpus JSON inputs)
#    2) build a corpus from standardized JSON input files (build the corpus)
#    3) run pre-processing scripts to clean-up the data and ensure its ready for input to topic models (perform pre-processing)
#    4) generate corpus statistics for contextualizing the data and understanding results (perform corpus statistics)
#    5) run basic python topic models for a quick scan of results before any advanced modeling in R (separate script)
#
####################################################################################################
####################################################################################################
#
# Code Map: List of Procedures / Functions
# - welcome
#
# == set of functions to build data corpus i.e., source, load, organize in corpus list ==
# - getInputFilesList
# - loadDSI # i.e., for each file in the input directory, load to the corpus
# - buildCorpus
# 
# == set of functions to perform pre-processing for topic modeling ==
# - performPreProcessing  - CURRENTLY A PLACEHOLDER
#       a) replace carrage returns, tabs, etc
#       b) replace characters not alphanumeric
#       c) replace single-character words with space
#       d) convert all characters to lowercase
#       e) replace selected stopwords with space
#       f) replace multiple blank characters with one blank character
# 
# == set of functions to perform topic modeling with sckikit-learn ==
# - useTF_LDA
# - getTF_Vectorizer
# - extractTF_Vector
# - fitLDA_Model
# - 
# - useTFIDF_NMF
# - getTFIDF_FeatureNames
# - fitNMF_Kullback_leibler_Model
# - fitNMF_Model
# - getTFIDF_Vectorizer
# - extractTFIDF_Vector
# - extractTFIDF_Vector
# 
# == set of basic functions ==
# - print_top_words  i.e., for each topic in model
# - 
# 
# == identify crucial parameters (these can be changed by the user)
#    i.e., obtainCorpusSizeSpecs ??
#
# 

####################################################################################################
####################################################################################################
#
# Procedure to welcome the user and identify the code
#
####################################################################################################
####################################################################################################


def welcome ():


    print
    print '******************************************************************************'
    print
    print 'Welcome to the Pre-Processor for the Capital Markets Structured Topic Model'
    print '  standardizing, normalizing and starting with a simple LDA method.'
    print 'Version 1.0, 11/18/2017, T. Danka'
    print 'For comments, questions, or bug-fixes, contact: troydanka2014@u.northwestern.edu'
    print ' ' 
    print 'This program builds a framework for implementing topic model applications'
    print 'It allows users to understand the activities in pre-processing, modeling, and visualizing topic models.'
    print
    print '******************************************************************************'
    print
    return()

####################################################################################################
####################################################################################################
#
# A collection of text analytics functions
#
####################################################################################################
####################################################################################################



def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)
    

####################################################################################################
####################################################################################################
#
# Procedures to initialize the dataCorpus from the raw source files, i.e., json files from scraper output and parsing
#
####################################################################################################
####################################################################################################

def loadDSI (DSI_filename):
    
    import json

    with open(DSI_filename) as json_file:  
        doc = json.load(json_file)
  
    return doc


def getInputFilesList (inputDirPath):
    
    ## This function will lookup all the json files in a directory and return as a list
    
    # initialize count of results files
    nfiles = 0
    #inputDirPath = 'C:\\Users\\troyd\\Dropbox\\Troy Files\\04 Education\\NU MSPA\\Thesis\\0A_Code\\Sandbox\\FullModel\\'
    temp_files_list = list()
    
    # identify all of the directory names in the input folder
    dir_names = [name for name in 
        os.listdir(inputDirPath)
        if os.path.isdir(os.path.join(inputDirPath, name))]
    print('\nInput Directory Names')   
    print(dir_names)
    
    # create a directory for the results... called results
    # will perform this in another function if necessary
    #os.mkdir('C:\\Users\\troyd\\Dropbox\\Troy Files\\04 Education\\NU MSPA\\Thesis\\0A_Code\\Sandbox\\FullModel\\' + 'results') 
    
    for input_dir in dir_names:
        # indentify all json files in this directory
        json_names = None  # initialize file name list for this directory
        json_names = [name for name in 
            os.listdir(os.path.join(inputDirPath, input_dir)) 
            if name.endswith('.json')]
        print('\nWorking on directory: ')
        print(input_dir)
        print(json_names)
        # work on files one at a time
        for input_file in json_names:
            # read in the html file
            this_dir = os.path.join(inputDirPath, input_dir)
            temp_files_list.append(os.path.join(this_dir, input_file))   
    
    return temp_files_list
    
    
def processCovariates(DSI_doc_output):
        
    # run a set up business rules to detect if a covariate condition exists
    
    #### Example Covariate Condiation ####
    ## 1. Does the DS contain narrative about regulation or "too big to fail"
    ## 2. Does the DS contain narrative about "data"
    ## 3. Does the DSI contain narrative about "AI/ML"
    
    ## The idea would be to add a covariate flag to the DSI set to true if pressent
    
    business_rule1_bool = False
    business_rule2_bool = False
    business_rule3_bool = False
    
    temp_string = DSI_doc_output["dsi_aggregated_content"]  #[0]  # gets the string from a 1 element list of the string
          
    # convert uppercase to lowercase
    temp_string = temp_string.lower()
    
    
    ## run business rule 1
    # if (DSI contains item in business_rule1) then set DSI_doc_output["dsi_cov_reg"] = TRUE, else FALSE
    
    business_rule1 = ['regulation', 'regulatory', 'rule', 'regulator']
    
    try:
        for businessrule in business_rule1:
            
            #if s.find("is") == -1:
            #    print "No 'is' here!"
            #else:
            #    print "Found 'is' in the string."   
            
            if temp_string.find(businessrule) == -1:
                
                business_rule1_bool = False
            else:
                business_rule1_bool = True
                
        DSI_doc_output["dsi_cov_reg"] = business_rule1_bool
                
    except:
        print("Problem with business_rule1 covariate")
    
    
    ## run business rule 2
    # if (DSI contains item in business_rule2) then set DSI_doc_output["dsi_cov_data"] = TRUE, else FALSE
    
    business_rule2 = ['data']
    
    try:
        for businessrule in business_rule2:
            
            if temp_string.find(businessrule) == -1:
                
                business_rule2_bool = False
            else:
                business_rule2_bool = True
                
        DSI_doc_output["dsi_cov_data"] = business_rule2_bool
                
    except:
        print("Problem with business_rule2 covariate")


    ## run business rule 3
    # if (DSI contains item in business_rule3) then set DSI_doc_output["dsi_cov_ai"] = TRUE, else FALSE
    
    business_rule3 = ['artificial intelligence', 'machine learning', 'data science']
    
    try:
        for businessrule in business_rule3:
            
            if temp_string.find(businessrule) == -1:
                
                business_rule3_bool = False
            else:
                business_rule3_bool = True
                
        DSI_doc_output["dsi_cov_ai"] = business_rule3_bool
                
    except:
        print("Problem with business_rule3 covariate")
    
    return DSI_doc_output
    
def organizeCorpusDSIs(inputDirPath):
    
    
    import json
    import re
    
    # Initialize temp variables 
    dataCorpus = list()
    temp_file_list = list()
    
    DSI_doc_output = dict() # initialize a dict for storing the model output DSI
    
    #inputDirPath = 'C:\\Users\\troyd\\Dropbox\\Troy Files\\04 Education\\NU MSPA\\Thesis\\0A_Code\\Sandbox\\FullModel\\'
    temp_file_list = getInputFilesList (inputDirPath)
    
    # Look through all identified files in input list and load the DSI file
    for DSI_filename in temp_file_list:
        
        ## determine the source to load file properly
        if re.search("FRB_Speech" , DSI_filename):
            
            print("Found a FRB Speech file:"+ DSI_filename)
            
            ## then load the DSI JSON file in DSI_filename
            DSI_doc = loadDSI(DSI_filename)
            
            ##
            
            ## map fields to the standardized model DSI dict
            
            ## PLACEHOLDER TO ADD A DSI ID
            
            DSI_doc_output["dsi_title"] = DSI_doc["speech_title"][0]
            DSI_doc_output["dsi_date"] = DSI_doc["speech_date"][0]   ## Need to work on a date time transformation
            DSI_doc_output["dsi_source"] = "FRB"
            DSI_doc_output["dsi_author"] = DSI_doc["speech_speaker"][0]
            DSI_doc_output["dsi_type"] = "FRB_Speech"
            DSI_doc_output["dsi_location"] = DSI_doc["speech_location"][0]
            DSI_doc_output["dsi_content_sponsor"] = "NA"
            DSI_doc_output["dsi_content"] = DSI_doc["speech_content"]
            DSI_doc_output["dsi_aggregated_content"] = DSI_doc_output["dsi_title"] + " " + DSI_doc["speech_content"] ## DSI_doc["speech_title"] + " "
                          
        elif re.search("TF_article" , DSI_filename):
            
            print("Found a TF Opinion file:"+ DSI_filename)  
               
            ## then load the DSI JSON file in DSI_filename
            DSI_doc = loadDSI(DSI_filename)
            
            ## PLACEHOLDER TO ADD A DSI ID
            
            DSI_doc_output["dsi_title"] = DSI_doc["article_title"][0]
            DSI_doc_output["dsi_date"] = DSI_doc["article_date"][0]   ## Need to work on a date time transformation
            DSI_doc_output["dsi_source"] = "TabbForum"
            DSI_doc_output["dsi_author"] = DSI_doc["article_author"][0]
            DSI_doc_output["dsi_type"] = "TF_OpinionAnalysis"
            DSI_doc_output["dsi_location"] = "www.tabbforum.com"
            
            if (len(DSI_doc["article_authorcompany"]) > 0):
                DSI_doc_output["dsi_content_sponsor"] = DSI_doc["article_authorcompany"][0]
            else:
                DSI_doc_output["dsi_content_sponsor"] = DSI_doc["article_authorcompany"]
                
            if (len(DSI_doc["article_content"]) > 0):
                DSI_doc_output["dsi_content"] = DSI_doc["article_content"][0]
            else:
                DSI_doc_output["dsi_content"] = DSI_doc["article_content"]
            
                 
            DSI_doc_output["dsi_aggregated_content"] = DSI_doc_output["dsi_title"] + " " + DSI_doc_output["dsi_content"] ## DSI_doc["speech_title"] + " " 
            
            ## then find then field by field process and map to output file field
       
        else:
            
            print("Found an unknown source file:"+ DSI_filename)
            
            
        DSI_doc_output = processCovariates(DSI_doc_output)  # Run "business rules" to identify covariates
        
        head, DSI_filename_short = os.path.split(DSI_filename)
        DSI_doc_output["dsi_filename"] = DSI_filename_short
        
        print("*** START FILENAME ***")
        print(DSI_filename_short)
        print("*** END FILENAME ***")

        saveDSI_tojson(DSI_filename_short, DSI_doc_output, False)
        
        dataCorpus.append(DSI_doc_output)
    
        
    return dataCorpus

def buildCorpus (inputDirPath):

    import json
    import re
    
    # Initialize temp variables 
    dataCorpus = list()
    temp_file_list = list()
    
    temp_file_list = getInputFilesList (inputDirPath)
    
    # Look through all identified files in input list and load the DSI file
    for DSI_filename in temp_file_list:
        
        DSI_doc = dict()
        
        head, DSI_filename_short = os.path.split(DSI_filename)
        
        if re.search("DSI_" , DSI_filename_short):
            
            try:
                #print(DSI_filename)
                DSI_doc = loadDSI(DSI_filename)
                dataCorpus.append(DSI_doc)
                #dataCorpus.append(DSI_doc["dsi_aggregated_content"])
            except:
                print("Error with file: "+DSI_filename)
                        
    return dataCorpus
    
def buildCorpusStatsDF(dataCorpus, file_name):
    
    ## The goal of this function is to consume a dataCorpus as list of JSON DSIs
    ## Then save certain features to a DF for purpose of generaging corpus statistics and visualizations
    
    # https://stackoverflow.com/questions/13784192/creating-an-empty-pandas-dataframe-then-filling-it
        
    print("Building corpus statistics data frame ...")
    
    
    ############## This code needs to be modified - just an example
    import datetime
    import pandas as pd
    import numpy as np
    
    labels = ['DSINUM','DSI_Type', 'DSI_TIMESTAMP', 'DSI_SOURCE', 'DSI_AUTHOR', 'DSI_TITLE', 'DSI_LOCATION', 'DSI_SPONSOR','DSI_COV_REG','DSI_COV_DATA','DSI_COV_AI']

    dsi_count = 0    
    dsi_list = list()
    dsi_record = list()
    
    for DSI_doc in dataCorpus:
        
        try:
            dsi_record = [dsi_count,DSI_doc["dsi_type"],DSI_doc["dsi_date"],DSI_doc["dsi_source"],DSI_doc["dsi_author"],DSI_doc["dsi_title"],DSI_doc["dsi_location"],DSI_doc["dsi_content_sponsor"],DSI_doc["dsi_cov_reg"],DSI_doc["dsi_cov_data"],DSI_doc["dsi_cov_ai"] ]  
                        
            #print(dsi_record)           
                            
        
        except:
            dsi_record = [dsi_count,
                        "",
                        "",
                        "",
                        "",
                        "",
                        "",
                        ""
                        ]
            #print(dsi_record)
        
        dsi_list.append(dsi_record)
        dsi_count = dsi_count + 1
        
        
    df = pd.DataFrame.from_records(dsi_list, columns=labels)
    
    # Confirm that the length of the dataCorpus is the length of the dataframe
    if (len(df) == len(dataCorpus)):
        print("Integrity Check: DSI statistics have been generaged for the FULL CORPUS")
    else:
        print("Integrity Check: DSI statistics have NOT been generaged for the FULL CORPUS")
    
    
    df.to_csv(file_name, encoding='utf-8')
    
    dataCorpus_DF = df
    
    return dataCorpus_DF


####################################################################################################
####################################################################################################
#
# Procedures to perform pre-processing of data on the dataCorpus
#
####################################################################################################
####################################################################################################

def performStopWordProcessing (doc_string):
    
    import re  # regular expressions 
    
    # Remove stop words
    # could be added to this list....
    stoplist = ['js','the','of','to','and','in','it','its',\
        'they','their','we','us','our','you','me','mine','my',\
        'for','by','with','within','about','between','from',\
        'as','for','an','what','who','how','when','where',\
        'whereas','is','are','were','this','that','if','or',\
        'not','nor','at','why','your','on','off',\
        'url','png','jpg','jpeg','gif','hover','em','px','pdf',\
        'header','footer','padding','before','after','ie','tm']
    
    # replace selected character strings/stop-words with space
    for i in range(len(stoplist)):
        stopstring = ' ' + stoplist[i] + ' '
        doc_string = re.sub(stopstring, ' ', doc_string)
    
    return doc_string


def performSingleCharacterProcessing (doc_string):
    
    import re  # regular expressions 
    
    # define list of codes to be dropped from document
    # carriage-returns, line-feeds, tabs
    codelist = ['\r', '\n', '\t']    
       
    # replace all characters not alphanumeric
    temp_string = re.sub('[^a-zA-Z]', '  ', doc_string)    
        
    # replace codes with space
    for i in range(len(codelist)):
        stopstring = ' ' + codelist[i] + '  '
        temp_string = re.sub(stopstring, '  ', temp_string)      
    
    # replace single-character words with space
    temp_string = re.sub('\s.\s', ' ', temp_string)
      
    return temp_string
    
def getNamedStringList():
    
    import csv
    
    with open('named_string_list.csv', 'rb') as f:
        reader = csv.reader(f)
        named_string_list = list(reader)
        
    
    return named_string_list
    
def transformNamedStrings(temp_string):
    
    import re  # regular expressions 
    
    ## load named strings list (i.e., from a named_string_list CSV file)
    #  named_string_list is a CSV file with two colums: col1 = named string, col2 = named string replacement
    
    named_string_list = getNamedStringList()
    
    
    for named_string_pair in named_string_list:    # loop through each "named string" in named strings list
        
        named_string = named_string_pair[0]
        named_string_replacement = named_string_pair[1] 
        
        temp_string = re.sub(named_string, named_string_replacement, temp_string)
            
        #### log the named string that was replaced
             
    
    return temp_string
    
    

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])
  
def getDataCorpusNGrams(dataCorpus, n):
    
    import nltk
    ngram_set = set()
        
    for dsi_doc in dataCorpus:
        
        temp_string = dsi_doc["dsi_aggregated_content"][0]  # gets the string from a 1 element list of the string_
        
        #temp_string_tokens = ['all', 'this', 'happened', 'more', 'or', 'less']
        temp_string_tokens = nltk.tokenize.word_tokenize(temp_string)
        
        temp_ngram_list = find_ngrams(temp_string_tokens, 2)
        
        ngram_set.update(temp_ngram_list)
    
    return ngram_set

    
def performPreProcessing (dataCorpus):

    # Bring in Python modules for file and text manipulation
    import os  # operating system commands
    import re  # regular expressions 

    dataCorpusPP = list()
     
    ## NOTE: This function uses a list of strings as corpus, eventually the corpus should have all vars in the model DSI    
                 
    for doc_string in dataCorpus:
        
        print(doc_string["dsi_filename"])
        
        temp_string = doc_string["dsi_aggregated_content"]  # gets the string from a 1 element list of the string
          
        # convert uppercase to lowercase
        temp_string = temp_string.lower()
    
        # Named strings should be transformed i.e., change U.S. to unitedstates or capital markets to capitalmarkets (will need some n-gram analysis)
        temp_string = transformNamedStrings(temp_string) 
        
        # Replace a single character stop items
        temp_string = performSingleCharacterProcessing(temp_string)
        
        # Remove stop-words from string
        temp_string = performStopWordProcessing(temp_string)      
        
        # replace multiple blank characters with one blank character
        temp_string = re.sub('\s+', ' ', temp_string)
        
        # Add cleaned-up doc to the new pre-processed data corpus
        
        doc_string["dsi_aggregated_contentPP"] = temp_string
        
        dataCorpusPP.append(doc_string)
        
        DSI_filename_lead = re.sub('.json', '', doc_string["dsi_filename"])
        
        DSI_filenamePP = DSI_filename_lead + "PP.json"
        
        saveDSI_tojson(DSI_filenamePP, doc_string, True)
           
                
    return dataCorpusPP

####################################################################################################
####################################################################################################
#
# Procedures to perform topic modeling on the dataCorpus using the scikit-learn Python scipts
#    This is intended to be an intro set of secripts to learn basics of topic modeling
#
####################################################################################################
####################################################################################################

def extractTFIDF_Vector(dataCorpus, tfidf_vectorizer):
    
    t0 = time()
    tfidf = tfidf_vectorizer.fit_transform(dataCorpus)
    print("done in %0.3fs." % (time() - t0))
    
    return tfidf

def getTFIDF_Vectorizer():
    
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
    
    return tfidf_vectorizer
    
def fitNMF_Model(tfidf):
    
    nmf = NMF(n_components=n_components, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
    
    return nmf
    
def fitNMF_Kullback_leibler_Model(tfidf):
    
    nmf = NMF(n_components=n_components, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)
    return nmf
    
def getTFIDF_FeatureNames(tfidf_vectorizer):
    
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    
    return tfidf_feature_names

def useTFIDF_NMF (dataCorpus):
    
    print("Extracting tf-idf features for NMF...")
    
    
    # Initialize the NMF vectorizer with data corpus
    tfidf_vectorizer = getTFIDF_Vectorizer()
    tfidf = extractTFIDF_Vector(dataCorpus, tfidf_vectorizer)
    
     
    # Fit the NMF model
    print("Fitting the NMF model (Frobenius norm) with tf-idf features, "
        "n_samples=%d and n_features=%d..."
        % (n_samples, n_features))
    
    t0 = time()
    nmf = fitNMF_Model(tfidf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (Frobenius norm):")
    tfidf_feature_names = getTFIDF_FeatureNames(tfidf_vectorizer)
    print_top_words(nmf, tfidf_feature_names, n_top_words)

    # Fit the NMF model - generalized Kullback-Leibler divergence
    print("Fitting the NMF model (generalized Kullback-Leibler divergence) with "
        "tf-idf features, n_samples=%d and n_features=%d..."
        % (n_samples, n_features))
    t0 = time()
    nmf = fitNMF_Kullback_leibler_Model(tfidf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in NMF model (generalized Kullback-Leibler divergence):")
    tfidf_feature_names = getTFIDF_FeatureNames(tfidf_vectorizer)
    print_top_words(nmf, tfidf_feature_names, n_top_words)
       
    return()


def fitLDA_Model():
    
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    
    return lda


def extractTF_Vector(dataCorpus, tf_vectorizer):
    
    t0 = time()
    tf = tf_vectorizer.fit_transform(dataCorpus)
    print("done in %0.3fs." % (time() - t0))
    
    return tf

def getTF_Vectorizer():
    
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')

    return tf_vectorizer

def useTF_LDA (dataCorpus):
    
    print("Extracting tf features for LDA...")
    
    # Initialize the NMF vectorizer with data corpus
    tf_vectorizer = getTF_Vectorizer()
    
    tf = extractTF_Vector(dataCorpus, tf_vectorizer)
         
    print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
    
    
    lda = fitLDA_Model()
    
    t0 = time()
    lda.fit(tf)
    print("done in %0.3fs." % (time() - t0))

    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words)
    
    return()



####################################################################################################
####################################################################################################
#
# A collection of worker-functions, designed to do specific small tasks
#
####################################################################################################
####################################################################################################


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
def saveDSI_tojson(DSI_Name, DSI_out, PP):
    
    import os
    import json
    
    #######################
    ## Start Initialize Inputs ##
    
    DSI_Name_input = DSI_Name
    DSI_Name_output = "DSI_"+DSI_Name
             
    ## End Initialize Inputs ##_+
    #######################
    
        # Set input directory from current working director
    realPath = os.path.realpath(os.getcwd())
    #inputDirPath = 'C:\\Users\\troyd\\Dropbox\\Troy Files\\04 Education\\NU MSPA\\Thesis\\0A_Code\\Sandbox\\FullModel\\'
    
    if (PP):
        outputDirPath = realPath + "\\DSI_InputsPP\\"
    else:
        outputDirPath = realPath + "\\DSI_Inputs\\"
    
    doc_out_filename_wDir = os.path.join(outputDirPath, DSI_Name_output)
    
    print(doc_out_filename_wDir)
        
    with open(doc_out_filename_wDir, 'w') as outfile:  
        json.dump(DSI_out, outfile)
        
    return
   
####################################################################################################
#**************************************************************************************************#
####################################################################################################


def main():

    # Define the global variables        
    
    global n_samples
    global n_features 
    global n_components # number of topics for the model
    global n_top_words # number of "top words" used to describe the topic

    #initialize global variables ** NOTE may move to functions later
    n_samples = 2000
    n_features = 1000
    n_components = 5  # used to be 10, lowered since cannot be more than number of docs => component equals doc
    n_top_words = 20

    # Set global variables to configure if code blocks should be run (i.e., each methodology step)
    RUN_ORGANIZE_CORPUS = True
    RUN_BUILD_CORPUS = True
    RUN_NGRAM_ANALYSIS = False
    RUN_PERFORM_PREPROCESSING = True  # Dependent on running buildDataCorpus (i.e., set dataCorpus input)
    RUN_PERFORM_TFIDF = False
    RUN_BUILD_CORPUS_STATS = True
    RUN_USE_TFIDF_NMF = False
    RUN_USE_TFIDR_LDA = False


    # This calls the procedure 'welcome,' which just prints out a welcoming message. 
    # All procedures need an argument list. 
    # This procedure has a list, but it is an empty list; welcome().

    welcome()
    
    # Set input directory from current working director
    realPath = os.path.realpath(os.getcwd())
    inputDirPath = realPath  
        
    ## Intended to grab the DSI source files and map to common format for corpus
    if (RUN_ORGANIZE_CORPUS):
        dataCorpus = organizeCorpusDSIs(inputDirPath)    
        print(len(dataCorpus))
    
    
    if (RUN_BUILD_CORPUS):
        dataCorpus = buildCorpus(inputDirPath)
        print(len(dataCorpus))
          
    if (RUN_NGRAM_ANALYSIS):     
       
        # Lets run the bi-gram analysis and save to a file    
        output_set = getDataCorpusNGrams(dataCorpus, 2)
        output_list = list(output_set)
        
        labels = ['NGRAM1','NGRAM2']
        output_set_df = pd.DataFrame.from_records(output_list, columns=labels)
        output_set_df.to_csv("Corpus_bi-gram_output.csv", encoding='utf-8')
        
      
    if (RUN_PERFORM_PREPROCESSING):
        dataCorpus = performPreProcessing(dataCorpus)

        dataCorpusContentList = list()
        for dsi in dataCorpus:
            temp_dsi = dsi["dsi_aggregated_contentPP"]  
            dataCorpusContentList.append(temp_dsi)  
    
        print(len(dataCorpusContentList))
    
    
        if (RUN_PERFORM_TFIDF):
            bloblist = list()
            for dsi in dataCorpus:
                temp_dsi = tb(dsi["dsi_aggregated_contentPP"])
                bloblist.append(temp_dsi) 
        
            blob_record = list()
            blob_list = list()
    
            for i, blob in enumerate(bloblist):
                print("Top words in document {}".format(i + 1))
                scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
                sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                for word, score in sorted_words[:25]:
                    print("\tWord: {}, TF-IDF: {}".format(word, round(score,5)))
                    blob_record = [i,word,score]
                    blob_list.append(blob_record)    
        
            import pandas as pd
            labels = ["DSINUM","WORD","TF-IDF"]        
            df = pd.DataFrame.from_records(blob_list, columns=labels)
            df.to_csv("CorpusRTM.csv", encoding='utf-8')
        
    ## Note to add function like buildCorput but from the 
    

    if (RUN_BUILD_CORPUS_STATS):
        dataCorpus_DF = buildCorpusStatsDF(dataCorpus, "dataCorpus_Statistics.csv")
        
        ## CONSIDER ADDING OTHER STATISTICS TO THE OUTPUT
        # - DSI WORD COUNT
        # - DSI WORD COUNT AFTER PRE-PROCESSING
        # - DSI NAMED STINGS FOUND
        # - DSI COVARIATE EXISTENCE
    
    if (RUN_USE_TFIDF_NMF):
        # Use tf-idf features for NMF.
        useTFIDF_NMF(dataCorpusContentList)
    
    if (RUN_USE_TFIDR_LDA):
        # Use tf features for LDA.
        useTF_LDA(dataCorpusContentList)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
                                              
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                
    
if __name__ == "__main__": main()

####################################################################################################
# End program
#################################################################################################### 


####################################################################################################
# Source reference tracking
####################################################################################################

#####  Topic modeling   #####  

# scikit-learn sample code
# Src: http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py
# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck
#         Chyi-Kwei Yau <chyikwei.yau@gmail.com>
# License: BSD 3 clause
