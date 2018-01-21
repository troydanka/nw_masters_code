require('RJSONIO')
require(RJSONIO) 

library(tm)
library(topicmodels)
library(data.tree)
library(jsonlite)
library(magrittr)
library(SnowballC)


#####    Set Working Directory #####
#setwd("WHATEVER-YOUR-WD-PATH-IS")
setwd("C:/Users/troyd/Dropbox/Troy Files/04 Education/NU MSPA/Thesis/0A_Code/Sandbox/FullModel_2017Corpus_MediumSize_1502/DSI_InputsPP")


#####    Read Corpus Source Files #####
#Create a list of all file names in directory
#Save File names to a variable

filenames <- list.files(pattern="DSI+.*json")
## Get names without ".CSV" and store in "names"
names <- substr(filenames, 1, 23)
print(filenames)

## Read in all json data files using a loop, same to variable same as original filename

### Initialize df to grab Tabb Forum json files
#mycorpus <- data.frame(
#  article_title=character(),
#  article_date = as.Date(character()),
#  article_content=character(),
#  docnum=numeric()
#)

### Initialize df to grab FRB json files
mycorpus <- data.frame(
  speech_title=character(),
  speech_date = as.Date(character()),
  speech_content=character(),
  docnum=numeric()
)

count_ref = 0
for(i in filenames){
  count_ref = count_ref + 1
  filepath <- file.path(paste(i))
  as.Node(fromJSON(filepath))
  print(i)
  ### routine to grab Tabb Forum json files
  #tmp_out_df <- as.Node(fromJSON(filepath)) %>% ToDataFrameTable(article_title = "article_title",
  #                                           #article_source ="article_source", 
  #                                           #article_link = "article_link", 
  #                                           #article_type = "article_type", 
  #                                           article_date = "article_date", 
  #                                           article_content = "article_content"
  #                                           #article_author = "article_author"
  #                                           )
  
  ### routine to grab FRB json files
  tmp_out_df <- as.Node(fromJSON(filepath)) %>% ToDataFrameTable(dsi_title = "dsi_title", #"speech_title",
                                                                 #article_source ="article_source", 
                                                                 #article_link = "article_link", 
                                                                 #article_type = "article_type", 
                                                                 dsi_date = "dsi_date", #speech_date", 
                                                                 dsi_aggregated_contentPP = "dsi_aggregated_contentPP" #speech_content"
                                                                 #article_author = "article_author"
  )
  

  tmp_out_df['docnum'] = count_ref
  assign(paste("Df", i, sep = "."), tmp_out_df)
  
  mycorpus <- rbind(mycorpus, tmp_out_df)
}


mycorpus
dim(mycorpus)
summary(mycorpus)
head(mycorpus)

## Define mapping for Tabb FOrum files
#m <- list(article_title = "article_title",
#          id = "docnum",
#          #article_source ="article_source", 
#          #article_link = "article_link", 
#          #article_type = "article_type", 
#          article_date = "article_date", 
#          content = "article_content"
#          #article_author = "article_author"
#          )

## Define mapping for FRB files
m <- list(dsi_title = "dsi_title",
          id = "docnum",
          #article_source ="article_source", 
          #article_link = "article_link", 
          #article_type = "article_type", 
          dsi_date = "dsi_date", 
          content = "dsi_aggregated_contentPP"
          #article_author = "article_author"
)

mycorpus.corpus <- tm::Corpus(tm::DataframeSource(mycorpus), readerControl = list(reader = tm::readTabular(mapping = m)))

mycorpus.corpus

mycorpus_topic.dtm <- tm::DocumentTermMatrix(mycorpus.corpus, control = list(stemming = FALSE, stopwords = FALSE,
                                                                             minWordLength = 2, removeNumbers = TRUE, removePunctuation = TRUE))

head(mycorpus_topic.dtm$Terms)
str(mycorpus_topic.dtm)
dim(mycorpus_topic.dtm)
inspect(mycorpus_topic.dtm)

term_tfidf <- tapply(mycorpus_topic.dtm$v/slam::row_sums(mycorpus_topic.dtm)[mycorpus_topic.dtm$i], mycorpus_topic.dtm$j, mean) *
  log2(tm::nDocs(mycorpus_topic.dtm)/slam::col_sums(mycorpus_topic.dtm > 0))
summary(term_tfidf)

## Keeping the rows with tfidf >= to the 0.00427
mycorpusreduced.dtm <- mycorpus_topic.dtm[,term_tfidf >= 0.022470]
summary(slam::col_sums(mycorpusreduced.dtm))



#install.packages('lda')
#install.packages("Rmpfr")
library(lda)

harmonicMean <- function(logLikelihoods, precision = 2000L) {
  library("Rmpfr")
  llMed <- median(logLikelihoods)
  as.double(llMed - log(mean(exp(-mpfr(logLikelihoods,
                                       prec = precision) + llMed))))
}


k <- 75
burnin <- 1000
iter <- 1000
keep <- 50
fitted <- topicmodels::LDA(mycorpusreduced.dtm, k = k, method = "Gibbs",control = list(burnin = burnin, iter = iter, keep = keep) )
## assuming that burnin is a multiple of keep
logLiks <- fitted@logLiks[-c(1:(burnin/keep))]

## This returns the harmomnic mean for k = 25 topics.
harmonicMean(logLiks)


############ START DETERMINE TIME TO RUN MODEL ########################


#seqk <- seq(2, 100, 1)
#burnin <- 1000
#iter <- 1000
#keep <- 50
#system.time(fitted_many <- lapply(seqk, function(k) topicmodels::LDA(mycorpusreduced.dtm, k = k,
#                                                                     method = "Gibbs",control = list(burnin = burnin,
#                                                                                                     iter = iter, keep = keep) )))

# extract logliks from each topic
#logLiks_many <- lapply(fitted_many, function(L)  L@logLiks[-c(1:(burnin/keep))])

# compute harmonic means
#hm_many <- sapply(logLiks_many, function(h) harmonicMean(h))


#install.packages('ggplot2')
#require(ggplot2)
#ldaplot <- ggplot(data.frame(seqk, hm_many), aes(x=seqk, y=hm_many)) + geom_path(lwd=1.5) +
#  theme(text = element_text(family= NULL),
#        axis.title.y=element_text(vjust=1, size=16),
#        axis.title.x=element_text(vjust=-.5, size=16),
#        axis.text=element_text(size=16),
#        plot.title=element_text(size=20)) +
#  xlab('Number of Topics') +
#  ylab('Harmonic Mean') +
#  annotate("text", x = 25, y = -150000, label = paste("The optimal number of topics is", seqk[which.max(hm_many)])) +
#  ggtitle(expression(atop("Latent Dirichlet Allocation Analysis of FRB Speeches", atop(italic("How many distinct topics in the abstracts?"), ""))))

#ldaplot
#seqk[which.max(hm_many)]

############ END DETERMINE TIME TO RUN MODEL ########################

# Time to Run the Model

system.time(mycorpus.model <- topicmodels::LDA(mycorpusreduced.dtm, 75, method = "Gibbs", control = list(iter=2000, seed = 0622)))

mycorpus.topics <- topicmodels::topics(mycorpus.model, 1)
## In this case  am returning the top 30 terms.

mycorpus.terms <- as.data.frame(topicmodels::terms(mycorpus.model, 30), stringsAsFactors = FALSE)
mycorpus.terms[1:15]



# Creates a dataframe to store the Lesson Number and the most likely topic
doctopics.df <- as.data.frame(mycorpus.topics)
doctopics.df <- dplyr::transmute(doctopics.df, LessonId = rownames(doctopics.df), Topic = mycorpus.topics)
doctopics.df$LessonId <- as.integer(doctopics.df$LessonId)

## Adds topic number to original dataframe of lessons
mycorpus.display <- dplyr::inner_join(mycorpus.display, doctopics.df, by = "LessonId")


topicTerms <- tidyr::gather(mycorpus.terms, Topic)
topicTerms <- cbind(topicTerms, Rank = rep(1:30))
topTerms <- dplyr::filter(topicTerms, Rank < 4)
topTerms <- dplyr::mutate(topTerms, Topic = stringr::word(Topic, 2))
topTerms$Topic <- as.numeric(topTerms$Topic)
topicLabel <- data.frame()
for (i in 1:20){
  z <- dplyr::filter(topTerms, Topic == i)
  l <- as.data.frame(paste(z[1,2], z[2,2], z[3,2], sep = " " ), stringsAsFactors = FALSE)
  topicLabel <- rbind(topicLabel, l)
  
}
colnames(topicLabel) <- c("Label")
topicLabel



theta <- as.data.frame(topicmodels::posterior(mycorpus.model)$topics)
head(theta[1:5])  # changed 5 to 3
dim(theta)


x <- as.data.frame(row.names(theta), stringsAsFactors = FALSE)
print(x)
colnames(x) <- c("LessonId")
x$LessonId <- as.numeric(x$LessonId)
theta2 <- cbind(x, theta)
print(theta2)
theta2 <- dplyr::left_join(theta2, FirstCategorybyLesson, by = "LessonId")
## Returns column means grouped by catergory
theta.mean.by <- by(theta2[, 2:27], theta2$Category, colMeans)  # changed 27 to 4
theta.mean <- do.call("rbind", theta.mean.by)

#install.packages("corrplot")
library(corrplot)
c <- cor(theta.mean)
corrplot(c, method = "circle")


theta.mean.ratios <- theta.mean
for (ii in 1:nrow(theta.mean)) {
  for (jj in 1:ncol(theta.mean)) {
    theta.mean.ratios[ii,jj] <-
      theta.mean[ii,jj] / sum(theta.mean[ii,-jj])
  }
}
topics.by.ratio <- apply(theta.mean.ratios, 1, function(x) sort(x, decreasing = TRUE, index.return = TRUE)$ix)

# The most diagnostic topics per category are found in the theta 1st row of the index matrix:
topics.most.diagnostic <- topics.by.ratio[1,]
head(topics.most.diagnostic)

##########################


#install.packages('stringi')
#install.packages('LDAvis')
#install.packages('dplyr')

topicmodels_json_ldavis <- function(fitted, corpus, doc_term){
  ## Required packages
  library(topicmodels)
  library(dplyr)
  library(stringi)
  library(tm)
  library(LDAvis)
  
  ## Find required quantities
  phi <- posterior(fitted)$terms %>% as.matrix
  theta <- posterior(fitted)$topics %>% as.matrix
  vocab <- colnames(phi)
  doc_length <- vector()
  for (i in 1:length(corpus)) {
    temp <- paste(corpus[[i]]$content, collapse = ' ')
    doc_length <- c(doc_length, stri_count(temp, regex = '\\S+'))
  }
  
  
  temp_frequency <- as.matrix(mycorpusreduced.dtm[,])  # updated with fix in https://stackoverflow.com/questions/43748874/inspect-termdocumentmatrix-to-get-full-list-of-words-terms-in-r
  freq_matrix <- data.frame(ST = colnames(temp_frequency),
                            Freq = colSums(temp_frequency))
  rm(temp_frequency)
  
  
  
  ## Convert to json
  json_lda <- LDAvis::createJSON(phi = phi, theta = theta,
                                 vocab = vocab,
                                 doc.length = doc_length,
                                 term.frequency = freq_matrix$Freq)
  
  return(json_lda)
}


mycorpos.json <- topicmodels_json_ldavis(mycorpus.model, mycorpus.corpus, mycorpusreduced.dtm)

# save to json file
#library("rjson")
write(mycorpos.json, "mycorpos.json")
json_read_mycorpos.json <- fromJSON("mycorpos.json")


serVis(mycorpos.json)
