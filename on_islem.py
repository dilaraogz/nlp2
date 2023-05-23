import pandas as pd
import re
import snowballstemmer


#sayısal değerlerin kaldırılması
def remove_numeric(value):
    bfr=[item for item in value if not item.isdigit()]
    bfr="".join(bfr)
    return bfr

#emojilerin kaldırılması
import re 
def remove_emoji(value):
    bfr=re.compile("[\U00010000-\U0010fffF]",flags=re.UNICODE)
    bfr=bfr.sub(r'',value)
    return bfr

#tek karakterli ifadelerin kaldırılması
def remove_single_character(value):
     return re.sub(r'(?:^| )\w(?:$| )','',value)

#noktalama ifadelerinin kaldırılması
def remove_noktalama(value):
    return re.sub(r'[^\w\s]', '', value)

#linkleri kaldırma 
def remove_link(value):
    return re.sub(r'http\S+|www\S+', '', value)

#hashtagleri kaldırma 
def remove_hashtag(value):
    return re.sub(r'#\w+', '', value)

#kullanıcı adlarınının kaldırılması
def remove_username(value):
    return re.sub(r'@\w+', '',value )

def stem_word(value):
   
     stemmer=snowballstemmer.stemmer("turkish")#tüekçe kullanım için
     value=value.lower() #küçük harfe çevirmek için
     value=stemmer.stemWords(value.split())
     stop_words = ["acaba", "ama", "aslında", "az", "bazı", "belki", "ben", "biri", "birkaç", "birşey", "biz", "bu", 
        "çok","çünkü", "da", "daha", "de", "defa", "diye", "eğer"
        "en", "gibi", "hem", "hep", "hepsi", "her", "hiç", "için", "ile", "ise", 
        "kez", "ki", "kim", "mı", "mu", "mü", "nasıl", "ne", "neden",
        "nerde", "nerede", "nereye", "niçin", "niye", "o", "sanki", "şey", "siz", "şu", 
        "tüm", "ve", "veya", "ya", "yani",
        "bir","iki","üç","dört","beş","altı","yedi","sekiz","dokuz","on"]
     value=' '.join(value)
     return value

#Tüm fonksiyonları çalıştırma
def pre_processing(value):
    return [remove_numeric(remove_emoji
                          (remove_single_character
                           (remove_noktalama
                            (remove_link
                             (remove_hashtag
                              (remove_username
                               (stem_word(word)
                               ))))))) for word in value.split()]

#boslukların kaldırılması
def remove_space(value):
    return[item for item in value if item.strip()]

