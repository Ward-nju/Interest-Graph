word_index <- read.table("/home/hungfei/桌面/paperdata/word_index.txt",head=FALSE,sep = ',')
word_index <- rt[order(rt$V1),]

filename="/home/hungfei/桌面/paperdata/result/lda-seq/topic-002-var-e-log-prob.dat"
num_seq=11
a = scan(filename)
b = matrix(a, ncol=num_seq, byrow=TRUE)
colnames(b) <- c("t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11")
#对该主题的各时间列，按概率由高到低排序，取top10词汇作为该主题该时间段的描述 
c=t(apply(b,2,order,decreasing=T))[,1:10]

word_matrix<-matrix(nrow = num_seq,ncol = 10)
prob_matrix<-matrix(nrow = num_seq,ncol = 10)

for(i in 1:dim(word_matrix)[1]){for(j in 1:dim(word_matrix)[2]) {word_matrix[i,j]=word_index[c[i,j],2]}}

