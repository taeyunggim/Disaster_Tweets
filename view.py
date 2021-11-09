import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 파일 읽기 
df = pd.read_csv('./data/train.csv')
print(df.sample(5))

# location 열 제거
tweet = df.drop(['location'], axis=1)
tweet = df.drop(['id'], axis=1)

# 결측값 제거
tweet = tweet.dropna(axis=0)

# 출력
print(tweet.sample(5))

# barplot target값 출력
x = tweet.target.value_counts()
sns.barplot(x.index,x)
plt.gca().set_ylabel('samples')
plt.show()

# 한 트윗당 글자 수의 분포
# 주로 120 ~ 140
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len = tweet[ tweet['target']==1 ]['text'].str.len()
ax1.hist(tweet_len,color='red')
ax1.set_title('disaster tweets')
tweet_len = tweet[ tweet['target']==0 ]['text'].str.len()
ax2.hist(tweet_len,color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Characters in tweets')
plt.show()

# 한 트윗당 단어 개수
# 주로 10 ~ 20
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
tweet_len = tweet[ tweet['target']==1 ]['text'].str.split().map(lambda x: len(x))
ax1.hist(tweet_len,color='red')
ax1.set_title('disaster tweets')
tweet_len = tweet[ tweet['target']==0 ]['text'].str.split().map(lambda x: len(x))
ax2.hist(tweet_len,color='green')
ax2.set_title('Not disaster tweets')
fig.suptitle('Words in a tweet')
plt.show()

# 한 트윗당 단어의 길이
# 주로 5 ~ 6
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
word = tweet[ tweet['target']==1 ]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('disaster')
word = tweet[ tweet['target']==0 ]['text'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Not disaster')
fig.suptitle('Average word length in each tweet')
plt.show()