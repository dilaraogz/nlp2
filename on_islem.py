import pandas as pd
import re
import snowballstemmer as sn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# sayısal verilerin kaldırılması
def remove_numeric(value):
    bfr = [item for item in value if not item.isdigit()]
    bfr = "".join(bfr)
    return bfr

def remove_emoji(value):
    bfr = re.compile("[\U00010000-\U0010ffff]",flags=re.UNICODE)
    bfr = bfr.sub(r'',value)
    return bfr


# tek karakterli ifadelerin kaldırılması
def remove_single_chracter(value):
    return re.sub(r'(?:^| )\w(?:$| )','',value)



# noktalama işaretlerinin kaldırılması
def remove_noktalama(value):
    return re.sub(r'[^\w\s]','',value)



# linklerin kaldırılması
def remove_link(value):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',value)


# linklerin kaldırılması
def remove_username(value):
    return re.sub('(@[^\s]+)','',value)


def remove_hashtag(value):
    return re.sub(r'#[^\s]+','',value)


stop_word = "acaba altmış altı ama ancak arada aslında ayrıca az bana bazı belki ben benden beni benim beri beş bile bin bir birçok biri birkaç birkez birşey birşeyi biz bize bizden bizi bizim böyle böylece bu buna bunda bundan bunlar bunları bunların bunu bunun burada çok çünkü da daha dahi de defa değil diğer diye doksan dokuz dolayı dolayısıyla dört edecek eden ederek edilecek ediliyor edilmesi ediyor eğer elli en etmesi etti ettiği ettiğini gibi göre halen hangi hatta hem henüz hep hepsi her herhangi herkesin hiç hiçbir için iki ile ilgili ise işte itibaren itibariyle kadar karşın katrilyon kendi kendilerine kendini kendisi kendisine kendisini kez ki kim kimden kime kimi kimse kırk milyar milyon mu mü mı nasıl ne neden nedenle nerde nerede nereye niye niçin o olan olarak oldu olduğu olduğunu olduklarını olmadı olmadığı olmak olması olmayan olmaz olsa olsun olup olur olursa oluyor on ona ondan onlar onlardan onları onların onu onun otuz oysa öyle pek rağmen sadece sanki sekiz seksen sen senden seni senin siz sizden sizi sizin şey şeyden şeyi şeyler şöyle şu şuna şunda şundan şunları şunu tarafından trilyon tüm üç üzerine var vardı ve veya ya yani yapacak yapılan yapılması yapıyor yapmak yaptı yaptığı yaptığını yaptıkları yedi yerine yetmiş yine yirmi yoksa yüz zaten"
stop_word = stop_word.split()


def stem_word(value):
    stemmer = sn.stemmer("turkish")
    value = value.lower()
    value = stemmer.stemWords(value.split())
    stop_words = stop_word
    
    value = [item for item in value if not item in stop_words]
    value = ' '.join(value)
    return value


def pre_processing(value):
    return [remove_numeric(remove_emoji
                          (remove_single_chracter
                           (remove_link
                            (remove_hashtag
                             (remove_username
                              (stem_word(word))))))) for word in value.split()]



def remove_space(value):
    return [item for item in value if item.strip()]

def bag_of_words(value):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(value)
    return X.toarray().tolist()

def tfidt(value):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(value)
    return X.toarray().tolist()


def word2vec(value):
    model = Word2Vec.load("data/word2vec.model")
    bfr_list = []
    bfr_len = len(value)

    for k in value:
        bfr = model.wv.key_to_index[k]
        bfr = model.wv[bfr]
        bfr_list.append(bfr)

    bfr_list = sum(bfr_list)
    bfr_list = bfr_list / bfr_len
    return bfr_list.tolist()
