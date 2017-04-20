setwd("C:/Users/1/James/Research/Projects/DTW_Similarity_Project/R/")
library(dtwclust) ; library(doSNOW) ; library(snow) ;library(RMySQL) ;require(doParallel)
library(data.table) ; library(lubridate) ; library(dplyr) ; library(readxl)
library(ggplot2) ; library(jmotif) ; library(doBy) ; library(dtwclust)
library(TSclust) ; library(dtwSat) ; library(quantmod) ;library(TSdist)
library(zoo) ; library(caret) ; library(splus2R) ; library(wordVectors)
library(qualV) ; library(PerformanceAnalytics) ; library(MTS) ; library(stats)
library(text2vec) ; library(parallel) ; library(mxnet) ; library(wordnet)
library(bnlearn) ; library(e1071) ; library(ROCR)
library(gbm) ; library(caret) ; library(ptw) ; library(FSelector)
library(NLP) ; library(openNLP) ; library(openNLPmodels.en) ; 
library(tm) ; library(stringr) ; library(gsubfn) ; library(plyr)

################################################ FUNCTIONS ################################################ 
setDict("C:/Program Files (x86)/WordNet/2.1/dict")

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

get_postag_features <- function(sickdataset,pad=20){
  word_token_annotator  <- Maxent_Word_Token_Annotator ()
  
  features <- foreach(i=1:dim(sick_dataset)[1]) %dopar% {
    tse1 <- space_tokenizer(as.character(sick_dataset$sentence_A[i]))
    tse2 <- space_tokenizer(as.character(sick_dataset$sentence_B[i]))
    y1 <- annotate(x, list(sent_token_annotator ,word_token_annotator))
    y1 <- annotate(x, list(sent_token_annotator ,word_token_annotator))
  }
}


get_synset_features <- function(sick_dataset,wordnet_dict,dist="dtw", pos="NOUN",pad=20){
  
  library(doParallel) ; library(snow) ; library(doSNOW) ;library(dtwclust)
  cl.tmp = makeCluster(rep("localhost",8), type="SOCK") 
  registerDoSNOW(cl.tmp) 
  
  #cl <- makeCluster(8)
  #registerDoParallel(cl)
  Dists <- data.frame() ; s1 <- data.frame(); s2 <- data.frame()
  # getting the synonyms here takes very long, first get a dictionary 
  Dists <- foreach (i = 1:10,.combine = "rbind",.packages = "dtw") %dopar% {
    #start.time <- Sys.time()
    tse1 <- strsplit(as.character(sick_dataset$sentence_A[i])," ")
    tse2 <- strsplit(as.character(sick_dataset$sentence_B[i])," ")
    
    # here I could get the column means and then get dtw of that instead
    #s1 <- as.matrix(foreach(j1 = 2:length(tse1[[1]]),.combine="rbind") %dopar%{
    #  wordnet_dict[which(wordnet_dict$names==tse1[[1]][j1]),][-1]
    #})
    #s2 <- as.matrix(foreach(j2 = 2:length(tse2[[1]]),.combine="rbind") %dopar%{
    #  wordnet_dict[which(wordnet_dict$names==tse2[[1]][j2]),][-1]
    #})    
    
    for (j1 in 1:length(tse1[[1]])) {s1 <- rbind(s1,wordnet_dict[which(wordnet_dict$names==tse1[[1]][j1]),][-1])}
    for (j2 in 1:length(tse2[[1]])) {s2 <- rbind(s2,wordnet_dict[which(wordnet_dict$names==tse2[[1]][j2]),][-1])}
    dtw(s1,s2)$dist
    #if (dist=="cos"){
    #  Dists <- rbind(cosineDists,cosine_dist(rbind(normalize(s1),normalize(s2))))
    #}
    #if(dist=="dtw"){
    #  Dists <- rbind(Dists,dtw(normalize(s1),normalize(s2))$dist)
    #}
    #s1 <- sapply(tse1[[1]], function(i) length(synonyms(i,pos = pos)))
    #s2 <- sapply(tse2[[1]], function(i) length(synonyms(i,pos = pos)))
    #sse1 <- if(length(s1) <= pad){c(s1,rep(0,abs(pad-length(s1))))} else (s1[1:pad])
    #sse2 <- if(length(s2) <= pad){c(s2,rep(0,abs(pad-length(s2))))} else (s2[1:pad])
    #cosine_dist(rbind(sse1,sse2))
    #end.time <- Sys.time()
    #time.taken <- end.time - start.time
    #print(time.taken)
  }
  registerDoSEQ()
  return (Dists)
}


if(initDict()) {
  filter <- getTermFilter("ExactMatchFilter", "hot", TRUE)
  terms <- getIndexTerms("ADJECTIVE", 5, filter)
  getSynsets(terms[[1]])
}

multivariate.dtw <- function(m1,m2,dist="euc"){
  if (dist == "euc"){
    euc.d <- dtw(m1,m2)$distance
  }
  else (dist == "euc"){
    
  }
}

peakyBlinders <- function (input, w = 2){
  env <- diff(sign(diff(input, na.pad = F)))
  peaks <- sapply(which(env < 0), function(j){
    
    p <- j - w + 1
    p <- ifelse(z > 0, z, 1)
    m <- j + w + 1
    m <- ifelse(m < length(input), m, length(input))
    ifelse(all(input[c(p : j, (j + 2) : m)] <= input[j + 1]), j + 1,numeric(0))
  })
  peaks <- unlist(peaks)
  peaks
}

# Classifiers
nb.model <- function(x,y){
  nb <- naiveBayes(y~.,data=x, laplace = 0.001,na.action = na.pass)
  nb.predicted.train<-predict(nb,x)
  nb.cm.train<-table(nb.predicted.train,y)
  nb.cm.train ; eval.metrics(nb.cm.train)
  #sts_tan <- tree.bayes(x,y)
}

gbm.model <- function(x,y){
  data <- cbind(x,y)
  colnames(data) <- c(paste0("X",as.character(1:dim(x)[2])),"y")
  objControl = trainControl(method = 'cv',number = 5,classProbs= TRUE)
  gbm.model <- train(data[,1:(length(data[1,])-1)],data$y,  method = "gbm",verbose = FALSE)
  gbm.predict <- predict(gbm.model,x)
  eval.metrics(table(Predicted=gbm.predict,Actual=data$y))
  pred.gbm <- prediction(as.numeric(gbm.predict),data$y) #create predictions
  auc.curve(pred.gbm)
}

lstm <- function(X,y){
  mx.lstm(X, eval.data = NULL, num.lstm.layer=3, seq.len=1, num.hidden=200,
          num.embed=425, num.label=len(y), batch.size=16, input.size, ctx = mx.ctx.default(),
          num.round = 10, update.period = 1, initializer = mx.init.uniform(0.01),
          dropout = 0.2, optimizer = "sgd")
}

conv_net <- function(X,y){
  #https://www.r-bloggers.com/image-recognition-tutorial-in-r-using-deep-convolutional-neural-networks-mxnet-package/
  # My input shape should be (128, 20), then embedding should get (128, 1, 20, 128), which is 327680; but I always got
  
  
  # 1st convolutional layer
  conv_1 <- mx.symbol.Convolution(data = X1reshaped, kernel = c(2, 25), num_filter = 20)
  tanh_1 <- mx.symbol.Activation(data = conv_1, act_type = "tanh")
  pool_1 <- mx.symbol.Pooling(data = tanh_1, pool_type = "max", kernel = c(2, 15), stride = c(1, 5))
  # 1st fully connected layer
  flatten <- mx.symbol.Flatten(data = pool_1)
  fc_1 <- mx.symbol.FullyConnected(data = flatten, num_hidden = 100)
  tanh_3 <- mx.symbol.Activation(data = fc_1, act_type = "tanh")
  # 2nd fully connected layer
  fc_2 <- mx.symbol.FullyConnected(data = tanh_3, num_hidden = 40)
  # Output. Softmax output since we'd like to get some probabilities.
  NN_model <- mx.symbol.LinearRegressionOutput(data = fc_2)
  #NN_model <- mx.symbol.SoftmaxOutput(data = fc_2)
  
  # Set seed for reproducibility
  mx.set.seed(100)
  
  # Device used. CPU in my case.
  devices <- mx.cpu()
  
  # Training
  #-------------------------------------------------------------------------------
  
  # Train the model
  model <- mx.model.FeedForward.create(NN_model,
                                       X = X,
                                       y = y,
                                       ctx = devices,
                                       num.round = 480,
                                       array.batch.size = 40,
                                       learning.rate = 0.01,
                                       momentum = 0.9,
                                       eval.metric = mx.metric.accuracy,
                                       epoch.end.callback = mx.callback.log.train.metric(100))
  
  
}

knn.classifier <- function(X.train,y.train,X.test,y.test){
  for (i in 1:20){
    fit <- knnreg(as.matrix(X.train),y.train, k = i)
    y.train.pred <- predict(fit, X.train)
    y.test.pred <- predict(fit, X.test)
    print(cor(y.test, y.test.pred))
    print(cor(y.test, y.test.pred,method="spearman"))
    print(mean((y.test - y.test.pred)^2))
  }
}

knn_predict <- function(test_data, train_data, k_value){
  pred <- c()  #empty pred vector 
  #LOOP-1
  for(i in c(1:nrow(test_data))){   #looping over each record of test data
    eu_dist =c()          #eu_dist & eu_char empty  vector
    eu_char = c()
    good = 0              #good & bad variable initialization with 0 value
    bad = 0
    
    #LOOP-2-looping over train data 
    for(j in c(1:nrow(train_data))){
      
      #adding euclidean distance b/w test data point and train data to eu_dist vector
      eu_dist <- c(eu_dist, euclideanDist(test_data[i,], train_data[j,]))
      
      #adding class variable of training data in eu_char
      eu_char <- c(eu_char, as.character(train_data[j,][[6]]))
    }
    
    eu <- data.frame(eu_char, eu_dist) #eu dataframe created with eu_char & eu_dist columns
    
    eu <- eu[order(eu$eu_dist),]       #sorting eu dataframe to gettop K neighbors
    eu <- eu[1:k_value,]               #eu dataframe with top K neighbors
    
    #Loop 3: loops over eu and counts classes of neibhors.
    for(k in c(1:nrow(eu))){
      if(as.character(eu[k,"eu_char"]) == "g"){
        good = good + 1
      }
      else
        bad = bad + 1
    }
    
    # Compares the no. of neighbors with class label good or bad
    if(good > bad){          #if majority of neighbors are good then put "g" in pred vector
      
      pred <- c(pred, "g")
    }
    else if(good < bad){
      #if majority of neighbors are bad then put "b" in pred vector
      pred <- c(pred, "b")
    }
    
  }
  return(pred) #return pred vector
}

euclideanDist <- function(a, b){
  d = 0
  for(i in c(1:(length(a)-1) ))
  {
    d = d + (a[[i]]-b[[i]])^2
  }
  d = sqrt(d)
  return(d)
}


# Preprocessing

# here the similarity of all pairs of sentences is passed 
# ie. optimal path between all pairs multivariate sentence embeddings. 
get_dtw_clusters <- function(tensor, plotting = F,type="dba"){
  
  if (type=="dba"){
    
    # Create and register parallel workers
    cl <- makeCluster(8, type = "SOCK")
    registerDoSNOW(cl)
    # Parallel backend detected automatically
    hc <- dtwclust(tensor, k = 20L,
                   distance = "dtw_basic", centroid = "dba",
                   seed = 9421, control = list(trace = TRUE, window.size = 20L))
    
    stopCluster(cl)
    registerDoSEQ()
    return(hc$clusters)
  }
  else{
    # Create and register parallel workers
    cl <- makeCluster(8, type = "SOCK")
    registerDoSNOW(cl)
    pc.tadp <- dtwclust(tensor, type = "tadpole", k = 20L,
                        dc = 1.5, control = ctrl)
    stopCluster(cl)
    registerDoSEQ()
    return(pc.tadp)
  }
  
  if (plotting ==T){
    ## Modifying some plot parameters
    plot(hc, labs.arg = list(title = "DBA Centroids", x = "time", y = "series"))
  }
}

sentence2vec <- function(sentences,model){
  Clean_String <- function(string){
    # Lowercase
    temp <- tolower(string)
    #' Remove everything that is not a number or letter (may want to keep more 
    #' stuff in your actual analyses). 
    temp <- stringr::str_replace_all(temp,"[^a-zA-Z\\s]", " ")
    # Shrink down to just one white space
    temp <- stringr::str_replace_all(temp,"[\\s]+", " ")
    # Split it
    temp <- stringr::str_split(temp, " ")[[1]]
    # Get rid of trailing "" if necessary
    indexes <- which(temp == "")
    if(length(indexes) > 0){
      temp <- temp[-indexes]
    } 
    return(temp)
  }
  # Calculate the number of cores
  #no_cores <- detectCores() - 4
  #cl <- makeCluster(no_cores,type="SOCK")
  #registerDoParallel(no_cores)
  
  #vecs <- foreach(i=1:length(sentences), .combine = 'c')  %dopar%  Clean_String(sentences[i])
  #head(vecs)
  #clean <- foreach(i=1:length(vecs), .combine = 'c')   %:%
  #          foreach(j=1:length(i), .combine = 'rbind')   %dopar% {
  #            print (j)
  #            try(model[j])
  #        }
  #stopCluster(cl)
  
  clean <- foreach(i=1:length(sentences), .combine = 'c')  %:%
    foreach(j=1:length(i), .combine = 'rbind')  %dopar% {
      print (j) ; model[j]
    }
  
  return(clean)
}

Clean_String <- function(string){
  # Lowercase
  temp <- tolower(string)
  #' Remove everything that is not a number or letter (may want to keep more 
  #' stuff in your actual analyses). 
  temp <- stringr::str_replace_all(temp,"[^a-zA-Z\\s]", " ")
  # Shrink down to just one white space
  temp <- stringr::str_replace_all(temp,"[\\s]+", " ")
  # Split it
  temp <- stringr::str_split(temp, " ")[[1]]
  # Get rid of trailing "" if necessary
  indexes <- which(temp == "")
  if(length(indexes) > 0){
    temp <- temp[-indexes]
  } 
  return(temp)
}

#' function to clean text
Clean_Text_Block <- function(text){
  if(length(text) <= 1){
    # Check to see if there is any text at all with another conditional
    if(length(text) == 0){
      cat("There was no text in this document! \n")
      to_return <- list(num_tokens = 0, unique_tokens = 0, text = "")
    }else{
      # If there is , and only only one line of text then tokenize it
      clean_text <- Clean_String(text)
      num_tok <- length(clean_text)
      num_uniq <- length(unique(clean_text))
      to_return <- list(num_tokens = num_tok, unique_tokens = num_uniq, text = clean_text)
    }
  }else{
    # Get rid of blank lines
    indexes <- which(text == "")
    if(length(indexes) > 0){
      text <- text[-indexes]
    }  
    # Loop through the lines in the text and use the append() function to 
    clean_text <- Clean_String(text[1])
    for(i in 2:length(text)){
      # add them to a vector 
      clean_text <- append(clean_text,Clean_String(text[i]))
    }
    # Calculate the number of tokens and unique tokens and return them in a 
    # named list object.
    num_tok <- length(clean_text)
    num_uniq <- length(unique(clean_text))
    to_return <- list(num_tokens = num_tok, unique_tokens = num_uniq, text = clean_text)
  }
  return(to_return)
}

read_sick_data <- function(root = "C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Datasets/Sentence Similarity/dataset-sts/data/sts/sick2014/",trainpath="SICK_train.txt",testpath="SICK_test_annotated.txt"){
  training <- read.csv(paste(root,trainpath, sep=""),sep="\t",fill = T,header = T)
  testing <- read.csv(paste(root,testpath, sep=  ""), sep = "\t", fill = T,header=T)
  dataset <- rbind(data.frame(training),data.frame(testing))
  cleanline <- ""
  for (i in 1:length(dataset[,4])){cleanline <- rbind(Clean_String(dataset[i,4]),cleanline)} 
  return (dataset)
}

read_para_data <- function(root ="C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Datasets/Sentence Similarity/dataset-sts/data/para/msr/",trainpath="msr-para-train.tsv",testpath="msr-para-test.tsv",valpath="msr-para-val.tsv"){
    training <- read.csv(paste(root,trainpath, sep=""),sep="\t",fill = T,header = T,quote = "")
    colnames(training) <- c("quality","id1","id2","Sentence1","Sentence2")
    validating <- read.csv(paste(root,valpath, sep=  ""), sep = "\t", fill = T,header=T,quote = "")
    colnames(validating) <- c("quality","id1","id2","Sentence1","Sentence2")
    testing <- read.csv(paste(root,testpath, sep=  ""), sep = "\t", fill = T,header=T,quote = "")
    colnames(testing) <- c("quality","id1","id2","Sentence1","Sentence2")
    dataset <- rbind(data.frame(training),data.frame(validating),data.frame(testing))
    cleanline <- ""
    for (i in 1:length(dataset[,4])){cleanline <- rbind(Clean_String(dataset[i,4]),cleanline)} 
    for (i in 1:length(dataset[,5])){cleanline <- rbind(Clean_String(dataset[i,5]),cleanline)} 
    return (dataset)
}

read_anssel_data <- function(root ="C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Datasets/Sentence Similarity/dataset-sts/data/anssel/yodaqa/",
                             trainpath="large2470-training.csv",testpath="large2470-test.csv",valpath="large2470-val.csv"){
  training <- read.csv(paste(root,trainpath, sep=""),sep=",",fill = T,header = T)
  validating <- read.csv(paste(root,valpath, sep=  ""), sep = ",", fill = T,header=T)
  testing <- read.csv(paste(root,testpath, sep=  ""), sep = ",", fill = T,header=T)
  dataset <- rbind(data.frame(training),data.frame(validating),data.frame(testing))
  cleanline <- ""
  for (i in 1:length(dataset[,4])){cleanline <- rbind(Clean_String(dataset[i,1]),cleanline)} 
  for (i in 1:length(dataset[,5])){cleanline <- rbind(Clean_String(dataset[i,3]),cleanline)} 
  return (dataset)
}

# https://cran.r-project.org/web/packages/text2vec/vignettes/glove.html
get_embeddings <- function(type="glove",Corpus = "wiki"){
  if (type=="glove"){
    if (Corpus=="tweets"){
      embedding.path <-  "C:/Users/1/James/grctc/GRCTC_Project/Classification/Data/Embeddings/Glove/"
      tweets <- readLines(paste(embedding.path,"glove.6B.50d.txt",sep=""))
    }
    if (Corpus=="wiki"){
      text8_file = "C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Implementation/text8"
      # this is actually wiki
      tweets = readLines(text8_file, n = 1, warn = FALSE)
    }

    # Create iterator over tokens
    tokens <- space_tokenizer(tweets)
    # Create vocabulary. Terms will be unigrams (simple words).
    it = itoken(tokens, progressbar = FALSE)
    vocab <- create_vocabulary(it)
    # Use our filtered vocabulary
    vectorizer <- vocab_vectorizer(vocab, 
                                   # don't vectorize input
                                   grow_dtm = FALSE, 
                                   # use window of 5 for context words
                                   skip_grams_window = 5L)
    tcm <- create_tcm(it, vectorizer)
    glove = GlobalVectors$new(word_vectors_size = 25, vocabulary = vocab, x_max = 10)
    glove$fit(tcm, n_iter = 10)
    word_vectors <- glove$get_word_vectors()
    return(word_vectors)
  }
  else{
    # https://github.com/bmschmidt/wordVectors
    embedding.path <-  "C:/Users/1/James/grctc/GRCTC_Project/Classification/Data/Embeddings/word2vec/"
    word2vec <- read.vectors(paste(embedding.path,"GoogleNews-vectors-negative300.bin",sep=""))
    return(word2vec)
  }
}

word2vec_euc <- function(sick_dataset,pad=15){
  
    foreach (i = 1:10,.combine='rbind') %dopar%{
      tse1 <- space_tokenizer(as.character(sick_dataset$sentence_A[i]))
      tse2 <- space_tokenizer(as.character(sick_dataset$sentence_B[i]))
      s1 <-foreach (j = 1:length(tse1[[1]]),.combine='rbind') %dopar%{
        sse1 <- model[rownames(model)==tse1[[1]][j],]
      }
      s2<- foreach (j = 1:length(tse2[[1]]),.combine='rbind') %dopar%{
        sse2 <- model[rownames(model)==tse2[[1]][j],]
      }
      sse1 <- ifelse(dim(sse1)[1]<= pad,rep(0,abs(pad-dim(sse1)[1])),sse1[1:pad,])
      sse2 <- ifelse(dim(sse2)[1]<= pad,rep(0,abs(pad-dim(sse2)[1])),sse2[1:pad,])
      dists <- dist(t(s1), t(s2))
      colSums(as.matrix(dists))
  }
}


get_embedded_features <- function(sick_dataset,model,type="glove"){
  if (type=="word2vec"){
    sent_emb1 <- foreach (i = 1:dim(sick_dataset)[1],.combine='rbind') %dopar%{
      tse1 <- space_tokenizer(as.character(sick_dataset$sentence_A[i]))
      foreach (j = 1:length(tse1[[1]]),.combine='rbind') %dopar%{
        sse1 <- model[rownames(model)==tse1[[1]][j],]
      }
    }
    sent_emb2 <- foreach (i = 1:dim(sick_dataset)[1],.combine='rbind') %dopar%{
      tse2 <- space_tokenizer(as.character(sick_dataset$sentence_B[i]))
      foreach (j = 1:length(tse2[[1]]),.combine = 'c') %dopar%{
        sse2 <- model[rownames(model)==tse2[[1]][j],]
      }
    }
  }
  if (type=="glove"){
    sent_emb1 <- foreach (i = 1:dim(sick_dataset)[1],.combine='rbind') %dopar%{
      tse1 <- space_tokenizer(as.character(sick_dataset$sentence_A[i]))
      foreach (j = 1:length(tse1[[1]]),.combine='c') %dopar%{
        sse1 <- model[rownames(model)==tse1[[1]][j],,drop=F]
      }
    }
    sent_emb2<- foreach (i = 1:dim(sick_dataset)[1],.combine='rbind') %dopar%{
      tse2 <- space_tokenizer(as.character(sick_dataset$sentence_B[i]))
      foreach (j = 1:length(tse2[[1]]),.combine = 'c') %dopar%{
        sse2 <- model[rownames(model)==tse2[[1]][j],,drop=F]
      }
    }
  }
  return(list(sentence1=sent_emb1,sentence2=sent_emb2))
}

list.depth <- function(this, thisdepth = 0) {
  # http://stackoverflow.com/a/13433689/1270695
  if(!is.list(this)) {
    return(thisdepth)
  } else {
    return(max(unlist(lapply(this, list.depth, thisdepth = thisdepth+1))))    
  }
}

cosine.sim <- function(a,b) crossprod(a, b)/sqrt(crossprod(a) * crossprod(b))

cos.sim <- function(ix) 
{
  A = X[ix[1],]
  B = X[ix[2],]
  return( sum(A*B)/sqrt(sum(A^2)*sum(B^2)) )
}   

dtw_features <- function(sick_dataset){
  
  sse1_all <- data.frame() ; sse2_all <- data.frame()
  dtw_all <- list() ; dtw_sentence1_to_sentence2 <- list() ; dtw_sentence2_to_sentence1 <- list()
  pad <- 15 ; s2s1 <- data.frame() ; s1s2 <- data.frame()
  # train 1 - 7542, test - 7543 - 
  for (x in 1:2000){#round(dim(sick_dataset)[1]*0.8)) {
    print (x)
    tse1 <- space_tokenizer(as.character(sick_dataset$sentence_A[i]))
    tse2 <- space_tokenizer(as.character(sick_dataset$sentence_B[i]))
    
    sent_emb1 <- foreach (j = 1:length(tse1[[1]]),.combine='rbind') %dopar%{
      sse1 <- model[rownames(model)==tse1[[1]][j],]
    }
    
    sent_emb2 <- foreach (j = 1:length(tse2[[1]]),.combine = 'rbind') %dopar%{
      sse2 <- model[rownames(model)==tse2[[1]][j],]
    }
    sse1_all <- list(sse1_all,as.matrix(sent_emb1))
    sse2_all <- list(sse2_all,as.matrix(sent_emb2))
    
    # this section introduces the optimal alignment between pairs
    dtw_pairs <- foreach (i = 1:dim(as.matrix(sent_emb1))[1],.combine = 'rbind') %dopar%{
      foreach (j = 1:dim(as.matrix(sent_emb2))[1],.combine = 'cbind') %dopar%{
        # sentence 1 word i distance to each word in sentence 2 for the rows 
        dtw(as.matrix(sent_emb1)[,i], as.matrix(sent_emb2)[,j])$distance
      }
    }
    # maybe means but not sure
    dtw_sentence1_to_sentence2[[x]] <- rowMeans(dtw_pairs)
    dtw_sentence2_to_sentence1[[x]] <- colMeans(dtw_pairs)
    dtw_all[[x]] <- dtw_pairs
    
    if (length(dtw_sentence2_to_sentence1[[x]])<= pad){
      s2s1 <- rbind(s2s1,c(dtw_sentence2_to_sentence1[[x]],rep(0,abs(pad-length(dtw_sentence2_to_sentence1[[x]])))))
      s1s2 <- rbind(s1s2,c(dtw_sentence1_to_sentence2[[x]],rep(0,abs(pad-length(dtw_sentence1_to_sentence2[[x]])))))
    }
    else{
      length(dtw_sentence2_to_sentence1[[x]]) <- pad
      length(dtw_sentence1_to_sentence2[[x]]) <- pad
      s2s1 <- rbind(s2s1,dtw_sentence2_to_sentence1[[x]])
      s1s2 <- rbind(s1s2,dtw_sentence1_to_sentence2[[x]])
    }
  }
  return(cbind(s1s2,s2s1))
}


# Evaluation 

auc.curve <- function (pred, name = NULL){
  perf <- performance(pred,"tpr","fpr")
  plot(perf,col="black",lty=3, lwd=3,main=paste("ROC curve"))
  auc <- performance(pred,"auc")
  auc <- unlist(slot(auc, "y.values"))
  maxauc<-max(round(auc, digits = 2))
  minauc <- paste(c("AUC  = "),maxauc,sep="")
  legend(0.5,0.4,c(minauc,"\n"),cex=1.5,box.col = "white",bty = "n")
}

eval.metrics <- function(conf.matrix){
  
  diag = diag(conf.matrix) # number of correctly classified instances per class 
  rowsums = apply(conf.matrix, 1, sum) # number of instances per class
  colsums = apply(conf.matrix, 2, sum) # number of predictions per class
  precision = diag / colsums 
  recall = diag / rowsums 
  accuracy = sum(diag)/sum(conf.matrix)
  f1 = 2 * precision * recall / (precision + recall)
  return (data.frame(precision, recall, f1,accuracy)) 
}

########################################## ----  SICK MAIN ----- ########################################################

model <- get_embeddings(type = "word2vec")
glove_vecs <- get_embeddings(type="glove", Corpus = "wiki")
main <- "C:/Users/1/James/Research/Projects/DTW_Similarity_Project/Datasets/Sentence Similarity/dataset-sts/data/"
sick_datapath <- paste0(main,"sts/sick2014/")
sick_dataset <- read_sick_data(root = sick_datapath,trainpath="SICK_train.txt",testpath="SICK_test_annotated.txt")
sick_sent_emb_1 <- sentence2vec(sick_dataset[,3],model = model)
sick_sent_emb_2 <- sentence2vec(sick_dataset[,4],model = model)
registerDoSEQ()

# Here I need to present the dtw optimal alignment values (OAVs) between sentence pairs in a matrix which is then 
# Approach : For an n x n matrix of OAVs, get the columnwise mean
# Result : This results in a single vector representation of similarity between sentence which is then passed to k-NN to predict relatedness score

X <- get_embedded_features(sick_dataset = sick_dataset, model = model)
y <- sick_dataset$relatedness_score
Xtrain <- data.frame(X=X.train,y=y.train) ; Xtest <- data.frame(X=X.test,y=y.test)  

y.train <- sick_dataset$relatedness_score[1:4500] ; y.test <- sick_dataset$relatedness_score[4501:dim(X[[1]])[1]]
X.train <- X[[1]][1:4500,] ; X.test <- X[[1]][4501:dim(X[[1]])[1],]

X.train.cosine <- foreach(i=1:dim(X.train)[1], .combine=c) %dopar% cosine.sim(X[[1]][i,],X[[2]][i,])
X.test.cosine <- foreach(i=(dim(X.train)[1]+1):dim(X[[1]])[1], .combine=c) %dopar% cosine.sim(X[[1]][i,],X[[2]][i,])

X.train.dtw <- foreach(i=1:dim(X.train)[1], .combine=c) %dopar% dtw(X[[1]][i,],X[[2]][i,],)$dist
X.test.dtw <- foreach(i=(dim(X.train)[1]+1):dim(X[[1]])[1], .combine=c) %dopar% dtw(X[[1]][i,],X[[2]][i,])$dist


# 61.67 % of the terms have at least 1 synonym for NOUN's, 24.91 for VERBs , 26.9 % adjectives and 8.86 % adverbs
# altogether spanning 78.71 % of all terms 
dict <- lapply(space_tokenizer(paste((as.character(unlist(sick_dataset$sentence_A,sick_dataset$sentence_B))), collapse = ' ')),unique)
clean.dict <- Clean_String(dict)
mylist <- list()
noun.items <- sapply(clean.dict,function (key) {length(synonyms(key,pos = "NOUN"))})
verb.items <- sapply(clean.dict,function (key) {length(synonyms(key,pos = "VERB"))})
adverb.items <- sapply(clean.dict,function (key) {length(synonyms(key,pos = "ADVERB"))})
adj.items <- sapply(clean.dict,function (key) {length(synonyms(key,pos = "ADJECTIVE"))})
all.items <- as.numeric(noun.items)+as.numeric(verb.items)+as.numeric(adverb.items)+as.numeric(adj.items)
wn.data <- data.frame(names=unlist(clean.dict),nouns=noun.items,verbs=verb.items,adverbs=adverb.items, adjectives=adj.items,all=all.items)
wn.syns.cos.features <- get_synset_features(sick_dataset = sick_dataset, wordnet_dict = wn.data,dist = "cos")
wn.syns.dtw.features <- get_synset_features(sick_dataset = sick_dataset, wordnet_dict = wn.data,dist = "dtw")

is.nan.data.frame <- function(x)
  do.call(cbind, lapply(x, is.nan))
wn.syns.dtw.features[is.nan(wn.syns.dtw.features)] <- 0

X.train.dtw.syns <- wn.syns.dtw.features[1:4500,]
X.test.dtw.syns <- wn.syns.dtw.features[4501:length(dim((wn.syns.dtw.features)[1])),]

X.train.ptw <- foreach(i=1:dim(X.train)[1], .combine=c) %dopar% ptw(X[[1]][i,],X[[2]][i,])$value
X.test.ptw <- foreach(i=(dim(X.train)[1]+1):dim(X[[1]])[1], .combine=c) %dopar% ptw(X[[1]][i,],X[[2]][i,])$value

train.weights <- information.gain(y~.,data=Xtrain) ; test.weights <- information.gain(y~.,data=Xtest)
X.train.ig <- X.train[,which(weights!=0)] ;  X.test.ig <- X.test[,which(weights!=0)]

X1reshaped <- array(X[[1]],dim=c(9427,1,15,25))
X2reshaped <- array(X[[2]],dim=c(9427,1,15,25))
X.train.features <- cbind(X.train.dtw,X.train.cosine,X.train.ptw,X.train.dtw.syns) 
X.test.features <- cbind(X.test.dtw,X.test.cosine,X.test.ptw,X.test.dtw.syns)
dim(train_array) <- c(15, 300, 1, ncol(train_x))


knn.classifier(X.train.features,y.train,X.test.features,y.test)

#list.depth(dtw_sentence1_to_sentence2)
#dts12s2 <- do.call(c, unlist(dtw_sentence1_to_sentence2, recursive=FALSE))
#dts22s1 <- do.call(c,unlist(do.call(c, unlist(dtw_sentence2_to_sentence1, recursive=FALSE)),recursive = F))
#dtw_all_new <- unlist(dtw_all, recursive = TRUE, use.names = TRUE)
#nb.model(s1s2,dataset$V5[1:dim(s1s2)[1]])
#gbm.model(s1s2,dataset$V5[1:dim(s1s2)[1]])

######################################## ----- SICK DATASET CLASSIFICATION ---- ################################################

X <- cbind(s2s1,s1s2)

########################################## ----  PARA MAIN ----- ########################################################

para_datapath <- paste0(main,"para/msr/")
para_dataset <- read_para_data(root = para_datapath, trainpath = "msr-para-train.tsv",
                          testpath = "msr-para-test.tsv",valpath="msr-para-val.tsv")

########################################## ----  ANSWER SELECTION MAIN ----- ########################################################

answer_selection_path <- paste0(main,"anssel/yodaqa/")
answer_selection_dataset <- read_anssel_data(root = answer_selection_path, trainpath = "large2470-training.csv",
                          testpath = "large2470-test.csv",valpath="large2470-val.csv")


