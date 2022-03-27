posts_df <- read_excel("./RQ2/dataset_lda_topics.xlsx")
posts_df$FavoriteCount <- ifelse(is.na(posts_df$FavoriteCount),0,posts_df$FavoriteCount)

#############popularity
sub1 <- subset(posts_df, (configuration == 1 & PostTypeId == 1), select=c(CommentCount,FavoriteCount,ViewCount,Score))
sub1$category <- "configuration"

sum2<- summarize(group_by(sub1,category)
                 ,FavoriteCount=mean(FavoriteCount)
                 ,ViewCount=mean(ViewCount)
                 ,Score=mean(Score)
                 
)

write.csv(sum2,'configuration.csv')  




    
#`version control`



################# unanswered questions
sub1 <- subset(posts_df, (configuration == 1 & PostTypeId == 1 & AnswerCount ==0), select=c(OwnerUserId))
sub1$category <- "configuration"
sum2 <- sub1 %>% 
  group_by(category) %>% 
  summarise(num = n())

print(sum2$num)


################# without accepted answer questions
sub1 <- subset(posts_df, (configuration == 1 & PostTypeId == 1 &  is.na(AcceptedAnswerId)), select=c(OwnerUserId))
sub1$category <- "configuration"
sum2 <- sub1 %>% 
  group_by(category) %>% 
  summarise(num = n())

print(sum2$num)


################# hours to accepted answer
sub1 <- subset(posts_df, (configuration == 1 & PostTypeId == 1 &  !is.na(AcceptedAnswerId)), select=c(hours_between))
sub1$category <- "configuration"
sum2<- summarize(group_by(sub1,category)
                 ,hours_between=median(hours_between)
                 
)

print(sum2$hours_between)
########################LALL
sub1 <- subset(posts_df, ( PostTypeId == 1 ), select=c(FavoriteCount,ViewCount,Score))
sum_t<- summarize(group_by(sub1)
                  ,FavoriteCount=mean(FavoriteCount)
                  ,ViewCount=mean(ViewCount)
                  ,Score=mean(Score)
)


sub1 <- subset(posts_df, ( PostTypeId == 1 & AnswerCount ==0), select=c(OwnerUserId))
sub1$category <- "all"
sum2 <- sub1 %>% 
  group_by(category) %>% 
  summarise(num = n())

print(sum2$num)

sub1 <- subset(posts_df, ( !is.na(AcceptedAnswerId)), select=c(hours_between))
sum_t<- summarize(group_by(sub1)
                 ,hours_between=median(hours_between)
                 
)
print(sum_t$hours_between)


######################## unasnswered

sub1 <- subset(posts_df, (PostTypeId == 1 &  is.na(AcceptedAnswerId) & AnswerCount ==0 ), select=c(Title,CommentCount))

co <-subset(sub1, (CommentCount>0), select=c(Title,CommentCount))


sub1$category <- "all"


sum2<- summarize(group_by(sub1,category)
                 ,meanC=mean(CommentCount)
                 ,medianC=median(CommentCount)
                 
)


