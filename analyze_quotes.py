import json
import numpy
import pandas as pd
import re
import spacy
from nltk.util import ngrams
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

nlp = spacy.load("en_core_web_sm")
# python -m spacy download en_core_web_sm

sns.set(style="whitegrid")

## Constants

# source = "Reddit"
source = "Medium"


## Functions
def clean_quote(quote_str):
    quote_str = quote_str.lower()
    quote_str = re.sub('“|”|\.|,|\?|—|!|\(|\)|#|"|-|\t', "", quote_str)
    quote_str = re.sub("[ ]{2,}", " ", quote_str)
    quote_str = quote_str.strip()

    return quote_str


def plot_bar(table, col1_name, col2_name):

    barchart_df = pd.DataFrame(table, columns=[col1_name, col2_name])

    sns.barplot(x=col2_name, y=col1_name, data=barchart_df)

    plt.show()


## read data
file_path = "./data/quotes-data-alltime-clean.json"

if source == "Reddit":
    file_path = "./data/reddit-r-quote-top-75.json"


with open(file_path, "r") as f:

    all_qoutes = json.load(f)


## Clean data
if source == "Medium":

    all_qoutes_pd = pd.DataFrame(all_qoutes)

    all_qoutes_pd = all_qoutes_pd.rename(columns={"top_quote": "quote"})
    all_qoutes_pd = all_qoutes_pd.drop("author_name", axis=1)
    all_qoutes_pd = all_qoutes_pd.drop("link", axis=1)
    all_qoutes_pd = all_qoutes_pd.drop("post_image", axis=1)
    all_qoutes_pd = all_qoutes_pd.drop("title", axis=1)
    all_qoutes_pd = all_qoutes_pd.drop("primary_topic", axis=1)

    all_qoutes_pd = all_qoutes_pd[all_qoutes_pd["quote"] != ""]

    all_qoutes_pd["claps"] = [
        re.sub("claps|K", "", item) for item in all_qoutes_pd["claps"]
    ]
    all_qoutes_pd["claps"] = all_qoutes_pd["claps"].apply(pd.to_numeric) * 1000

    all_qoutes_pd["quote_clean"] = all_qoutes_pd["quote"]
    all_qoutes_pd["quote_clean"] = all_qoutes_pd["quote_clean"].apply(clean_quote)

else:

    quotes_column = []

    for item in all_qoutes:
        quotes_column.append(clean_quote(item["data"]["title"]))

    all_qoutes_pd = pd.DataFrame(quotes_column, columns=["quote_clean"])


print(all_qoutes_pd)


## add text to spacy
all_quotes_str = " ".join(all_qoutes_pd["quote_clean"])

doc = nlp(all_quotes_str)

del all_quotes_str


## plot top words
all_words = [
    token.text
    for token in doc
    if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"
]

## aggregate and count words
top_entities = Counter(all_words).most_common(20)

plot_bar(top_entities, "Words", "Count")


## show examples of quotes which includes the word "people"
quotes_with_words_first = all_qoutes_pd.loc[
    all_qoutes_pd["quote_clean"].str.find("people") != -1
]["quote_clean"].values.tolist()[1:5]

print(quotes_with_words_first)


## Plot first words
first_words = [word.split(" ")[0] for word in all_qoutes_pd["quote_clean"]]

top_first_words = Counter(first_words).most_common(20)

plot_bar(top_first_words, "First Word", "Count")


## show examples of quotes which starts with the word "the"
quotes_with_words_first = all_qoutes_pd.loc[
    all_qoutes_pd["quote_clean"].str.find("the") == 0
]["quote_clean"].values.tolist()[1:5]

print(quotes_with_words_first)


## Plot last words
last_words = [word.split(" ")[-1] for word in all_qoutes_pd["quote_clean"]]

top_last_words = Counter(last_words).most_common(20)

plot_bar(top_last_words, "Last Word", "Count")


## show examples of quotes which ends with the word "it"
quotes_with_words_last = all_qoutes_pd.loc[
    all_qoutes_pd["quote_clean"].str.endswith("it")
]["quote_clean"].values.tolist()[1:5]

print(quotes_with_words_last)


## Plot Trigrams
trigrams_list = [
    list(ngrams(quote.split(" "), 3)) for quote in all_qoutes_pd["quote_clean"]
]

all_trigrams = []
for ngram_list in trigrams_list:

    for ngram_tuple in ngram_list:
        all_trigrams.append(" ".join(ngram_tuple))

# print(all_trigrams)

del trigrams_list

top_trigrams = Counter(all_trigrams).most_common(20)


plot_bar(top_trigrams, "Trigrams", "Count")


## show examples of quotes which include trigram "you have to"
quotes_with_trigrams = all_qoutes_pd.loc[
    all_qoutes_pd["quote_clean"].str.find("you have to") != -1
]["quote_clean"].values.tolist()[1:5]

print(quotes_with_trigrams)


## Calculate & plot the optimal number of words
words_per_quote = [len(quote.split(" ")) for quote in all_qoutes_pd["quote_clean"]]


ax = sns.distplot(words_per_quote, kde=False)

ax.set(xlabel="Number of words", ylabel="Number of quotes")

plt.show()


## Find PoS patterns

all_pos_tokens = []
all_text_tokens = []

for token in doc:
    # ignore punctuation
    if token.pos_ == "PUNK":
        continue
    # print(token.pos_, token.text)

    all_pos_tokens.append(token.pos_)
    all_text_tokens.append(token.text)

## generate 4-grams from PoS tags
all_pos_ngrams = list(ngrams(all_pos_tokens, 4))
all_text_ngrams = list(ngrams(all_text_tokens, 4))


pos_dict = {}
all_pos_ngrams_redable = []
i = 0

## generate a dict with pos ngram and example text from the quotes
for pos_ngram in all_pos_ngrams:

    ngram_key = "_".join(pos_ngram)

    if pos_dict.get(ngram_key) is None:
        pos_dict[ngram_key] = [all_text_ngrams[i]]
    else:
        pos_dict.get(ngram_key).append(all_text_ngrams[i])

    i = i + 1

    # just of the sake of charting
    all_pos_ngrams_redable.append(ngram_key)


top_pos_ngrams = Counter(all_pos_ngrams_redable).most_common(20)


plot_bar(top_pos_ngrams, "4-grams", "Count")

## show example text for top 10 PoS ngrams
for ngram_key, count in top_pos_ngrams[0:10]:
    print(ngram_key)
    print(pos_dict.get(ngram_key)[0:3])

