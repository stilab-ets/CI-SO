posts_df <- read_excel("./RQ2/dataset_lda_topics.xlsx")
count=distinct(posts_df,OwnerUserId)
length(count$OwnerUserId)


#Users who post only questions
sub1 <- distinct(subset(posts_df, (PostTypeId == 1), select=c(OwnerUserId)),OwnerUserId)
sub2 <- distinct(subset(posts_df, (PostTypeId == 2), select=c(OwnerUserId)),OwnerUserId)
only_questions <- subset(sub1, !(OwnerUserId %in% sub2$OwnerUserId))
length(only_questions$OwnerUserId)

#Users who post questions & accepted answers
sub2 <- distinct(subset(posts_df, (!is.na(AcceptedAnswerId)), select=c(OwnerUserId)),OwnerUserId)

intersect <- subset(only_questions, (OwnerUserId %in% sub2$OwnerUserId))
length(intersect$OwnerUserId)

#Users who post questions & non-accepted answers
sub2 <- distinct(subset(posts_df, (is.na(AcceptedAnswerId) & PostTypeId == 1), select=c(OwnerUserId)),OwnerUserId)
intersect <- subset(only_questions, (OwnerUserId %in% sub2$OwnerUserId))
length(intersect$OwnerUserId)


#Users who post only accepted answers
accept_all <- distinct(subset(posts_df, (!is.na(AcceptedAnswerId)), select=c(OwnerUserId)),OwnerUserId)


sub2 <- distinct(subset(posts_df, (is.na(AcceptedAnswerId)&  PostTypeId == 1), select=c(OwnerUserId)),OwnerUserId)
only_accept <- subset(accept_all, !(OwnerUserId %in% sub2$OwnerUserId))
length(only_accept$OwnerUserId)

#Users who post accepted & non-accepted answers
sub2 <- distinct(subset(posts_df, is.na(AcceptedAnswerId)&  PostTypeId == 1, select=c(OwnerUserId)),OwnerUserId)
intersect <- subset(accept_all, (OwnerUserId %in% sub2$OwnerUserId))
length(intersect$OwnerUserId)

#Users who post only non-accepted answers
total_not_accept <- distinct(subset(posts_df, (is.na(AcceptedAnswerId) &  PostTypeId == 1), select=c(OwnerUserId)),OwnerUserId)
sub2 <- distinct(subset(posts_df, (!is.na(AcceptedAnswerId) ), select=c(OwnerUserId)),OwnerUserId)
notin <- subset(total_not_accept, !(OwnerUserId %in% sub2$OwnerUserId))
length(notin$OwnerUserId)


#Users who post non-accepted answers & questions

sub2 <- distinct(subset(posts_df, PostTypeId == 1 , select=c(OwnerUserId)),OwnerUserId)
intersect  <- subset(total_not_accept, (OwnerUserId %in% sub2$OwnerUserId))
length( intersect$OwnerUserId)




#Users who post non-accepted & accepted answers
sub2 <- distinct(subset(posts_df,!is.na(AcceptedAnswerId) , select=c(OwnerUserId)),OwnerUserId)
intersect  <- subset(total_not_accept, (OwnerUserId %in% sub2$OwnerUserId))
length( intersect$OwnerUserId)




####################################################################################
#Number of questions created by a user
users <- subset(posts_df, (PostTypeId == 1))

sum <- users %>% 
  group_by(OwnerUserId) %>% 
  summarise(num = n())

sum2 <- sum %>% 
  group_by(num) %>% 
  summarise(num2 = n())
write.csv(sum2,'Number of questions created by a user.csv')        
######################################################################################
#Number of Accepted Answers created by a user
users <- subset(posts_df, (!is.na(AcceptedAnswerId)))

sum <- users %>% 
  group_by(OwnerUserId) %>% 
  summarise(num = n())

sum2 <- sum %>% 
  group_by(num) %>% 
  summarise(num2 = n())
write.csv(sum2,'Number of Accepted Answers created by a user.csv') 


######################################################################################
#Number of Non-Accepted Answers created by a user
users <- subset(posts_df, (is.na(AcceptedAnswerId) & PostTypeId == 1))

sum <- users %>% 
  group_by(OwnerUserId) %>% 
  summarise(num = n())

sum2 <- sum %>% 
  group_by(num) %>% 
  summarise(num2 = n())
write.csv(sum2,'Number of Non-Accepted Answers created by a user.csv') 

######################################################################################
## count of question per year
sum <- subset(posts_df, PostTypeId == 1 ) %>% 
  group_by(year) %>% 
  summarise(questions = n())
## count of accepted answers
sum <- subset(posts_df, PostTypeId == 1 & !is.na(AcceptedAnswerId)) %>% 
  group_by(year) %>% 
  summarise(accepeted_answers = n())
## count of involved developers per year
sum <- subset(posts_df, !is.na(OwnerUserId)) %>% 
  group_by(year,OwnerUserId) %>% 
  summarise(nbr_questions_per_dev = n())

sum2 <- sum %>% 
  group_by(year) %>% 
  summarise(nbr_dev = n())

dist <- posts_df %>% 
  group_by(OwnerUserId) %>% 
  summarise(nbr_dev = n())


sum2 <-subset(sum2 , year>2008 & year<2020  )
ggplot(sum2, aes(x=year, y=nbr_dev)) + scale_x_continuous(breaks = seq(2009, 2019, 1))+
  geom_bar(stat="identity", width=.5, position = "dodge", aes (fill = "#FF6666")) + 
  theme(axis.text.x = element_text(angle=65, vjust=0.6) ,legend.position="bottom",text = element_text(size = 14))+
  labs(x="Year", y ="Number of distinct users", fill="")+ theme(legend.position="none")
