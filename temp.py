# import re

# # str =  ' Pledge US$ 10 or more'

# # temp = max(re.findall(r'\d+', str))
# # print(temp)

# pledge = ["The Pledge US$ 1 or more. Standard Goodies. This isn't a reward, just some info on the rewards.. Starting at $20 every reward includes the At Night I Fly album and the chance to submit questions for the 'Questions for Spoon' episode/segments we'll do throughout the season.. Estimated delivery. Feb 2020. 0 backers. Pledge amount. 1. $. Continue. Other payment options. Pledge US$ 10 or more. Questions for Spoon. For a special 'Questions for Spoon' episode we'll be taking fan questions. Questions about growing up in the desert, living in prison, meeting Mother Teresa (it's true).. Incase your question isn't included in the edited episode we'll also send the unedited interview where I (matthew) ask Spoon your questions. First 20 questions get a guaranteed answer, after that I'll just be picking the most interesting ones so keep 'em coming!. Estimated delivery. Mar 2020. 3 backers. Pledge amount. 10. $. Continue. Other payment options. Pledge US$ 20 or more. At Night I Fly the album. The music plus extra features from the podcast. Matthew Schneeman was a musician in his past life and learned how to compose for radio. Weird electronic stuff that's all based in traditional American ballads. Think chill Randy Newman covers on the youtube Lo-fi channels.. If we get past out $5,000 goal At Night I Fly will also be commissioning pieces specifically for the podcast from currently and formally incarcerated people. Die Jim Crow is a record label Spoon and Matthew have ties to.. Estimated delivery. Feb 2020. 5 backers. Pledge amount. 20. $. Continue. Other payment options. Pledge US$ 30 or more. Signed Poem by Spoon. I'll print out from and letters addressed to you and Spoon will sign the poems and mail them from Solano. I haven't picked one out (i'll let Spoon decide which would be most fitting.) but here are some poems if you're curious: http://realnessnetwork.blogspot.com/. Estimated delivery. Feb 2020. 4 backers. Pledge amount. 30. $. Continue. Other payment options. Pledge US$ 30 or more. Copy of Signed Poem by Spoon. (Same as other pledge, I just had a typo in it. Kickstarter, for good reason, doesn't let you edit rewards after the campaign starts.). I'll mail some poems to Spoon. He'll sign and send them on to you from Solano. I haven't picked one out (i'll let Spoon decide which would be most fitting.) but here are some poems if you're curious: http://realnessnetwork.blogspot.com/. We'll also include the album of music from the show.. Estimated delivery. Feb 2020. 3 backers. Pledge amount. 30. $. Continue. Other payment options. Pledge US$ 50 or more. Copy of memoir By Heart. By Heart is a beautiful memoir written by Spoon and his mentor Judith Tannenbaum. New Village Press is DONATING copies of the book for this campaign. I, producer Matthew, used the book as a resource to understand Spoon, his work, and chronology. New Village Press giving us some copies for free is amazing.. The book is available for $20 on the New Village Press website if you'd like to pay them full price and then give this kickstarter whatever pledge you were planning. If you're going to listen to At Night I Fly the podcast I can't recommend the book enough.. Estimated delivery. Feb 2020. Ships to. Anywhere in the world. 7 backers. Limited (43 left of 50). Shipping destination. Select a country:. Select a country:. Pledge amount. 50. $. Continue. Other payment options. Pledge US$ 100 or more. Call or Letter from Spoon. Letter or call from Spoon. Whichever you prefer. If you want a call it'll cost about 20 cents a minutes because Spoon has to use a prison phone service. Ask him questions, let him know what you like to dislike about the show, whatever! He knows a lot about poetry and can recommend some great authors or poems.. Estimated delivery. Feb 2020. 3 backers. Pledge amount. 100. $. Continue. Other payment options. Pledge US$ 200 or more. Sponcer. Are you a prison advocacy show? Want to Sponsor an episode? That would be great! You don't even have to be a prison rights organization. You could just be charitable, that's great too.. We will also include all the previous rewards offered.. Estimated delivery. Feb 2020. 0 backers. Limited (30 left of 30). Pledge amount. 200. $. Continue. Other payment options. Pledge US$ 500 or more. Full Sponsor. Are you an organization that promotes the arts in prison? Or any advocacy in prison? Or you're just charitable? For $500 bucks we'll team up for the entire season to support each others work.. Additionally, we will give you all the rewards we offer.. Estimated delivery. Feb 2020. 1 backer. Pledge amount. 500. $. Continue. Other payment options"]

# min_amt = []
# max_amt = []
# mean = []
# pledge_count = []

# temp_set = []
# amt_set = []
# check = ['Pledge','or mor']

# for val in pledge:
#     temp_set = []
#     desc = val.split('.')
#     for line in desc:
#         if all([x in line for x in check]):
#             temp_set.append(line)
#     for str in temp_set:
#         amt = min(re.findall(r'\d+', str))
#         amt_set.append(amt)
#     min_amt.append(min(amt_set))
#     max_amt.append(max(amt_set))
#     # mean.append(sum(amt_set)/len(amt_set))
#     pledge_count.append(len(temp_set))

# print(temp_set)
# print(min_amt)
# print(max_amt)
# print(pledge_count)

import pandas as pd

df = pd.read_csv('Final Data.csv')

X = df.drop(['class','country','spotlight','staff_pick'], axis=1)
Y = df['class']

col_list = X.columns()
d = X.shape[1]

rel_CF = [0 for i in range(d)]
print(rel_CF)
count = 0

def IG(col_1, col_2):
    from sklearn.feature_selection import mutual_info_classif
    gain = mutual_info_classif(col_1,col_2)
    return gain  

for m in  range(d):
    mu = df[col_list[m]].mean()

    n_1 = df[df['class'] == 0].count()
    mu_1 = df[col_list[m]][df['class'] == 0].mean()
    var_1 = df[col_list[m]][df['class'] == 0].var()

    n_2 = df[df['class'] == 1].count()
    mu_2 = df[col_list[m]][df['class'] == 1].mean()
    var_2 = df[col_list[m]][df['class'] == 1].var()

    inter_class = n_1*(mu_1-mu)**2 + n_2*(mu_2-mu)**2
    intra_class = (n_1-1)*var_1 + (n_2-1)*var_2

    fscore = inter_class / intra_class
    igain = IG(df[[col_list[m]]],df['class'])
    print('fscore : ', fscore)
    print('gain : ', igain)

    rel_CF[count] = (1/(2*(fscore + igain)))
    count += 1

print(rel_CF)