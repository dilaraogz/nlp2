### kütüphaneleri tanımlayınız. ### 
import pandas as pd
import re
import snowballstemmer as sn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

### tanımlanan fonksiyonlar da pass'lı ifadeler eksiktir. Bu fonksiyon içeriklerini doldurunuz ###

# numerik karakterlerin kaldırılması
def remove_numeric(value):
    bfr = [item for item in value if not item.isdigit()]
    bfr = "".join(bfr)
    return bfr
# emojilerin kaldırılması
def remove_emoji(value):
    bfr = re.compile("[\U00010000-\U0010ffff]",flags=re.UNICODE)
    bfr = bfr.sub(r'',value)
    return bfr

#noktalama işaretlerinin kaldırılması
def remove_single_chracter(value):
    return re.sub(r'(?:^| )\w(?:$| )','',value)

#tek karakterli ifadelerin kaldırılması
def remove_noktalama(value):
    return re.sub(r'[^\w\s]','',value)

#linklerin kaldırılması 
def remove_link(value):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',value)

# hashtaglerin kaldırılması
def remove_hashtag(value):
    return re.sub(r'#[^\s]+','',value)

# kullanıcı adlarının kaldırılması
def remove_username(value):
    return re.sub('(@[^\s]+)','',value)


stop_word = "acaba altmış altı ama ancak arada aslında ayrıca az bana bazı belki ben benden beni benim beri beş bile bin bir birçok biri birkaç birkez birşey birşeyi biz bize bizden bizi bizim böyle böylece bu buna bunda bundan bunlar bunları bunların bunu bunun burada çok çünkü da daha dahi de defa değil diğer diye doksan dokuz dolayı dolayısıyla dört edecek eden ederek edilecek ediliyor edilmesi ediyor eğer elli en etmesi etti ettiği ettiğini gibi göre halen hangi hatta hem henüz hep hepsi her herhangi herkesin hiç hiçbir için iki ile ilgili ise işte itibaren itibariyle kadar karşın katrilyon kendi kendilerine kendini kendisi kendisine kendisini kez ki kim kimden kime kimi kimse kırk milyar milyon mu mü mı nasıl ne neden nedenle nerde nerede nereye niye niçin o olan olarak oldu olduğu olduğunu olduklarını olmadı olmadığı olmak olması olmayan olmaz olsa olsun olup olur olursa oluyor on ona ondan onlar onlardan onları onların onu onun otuz oysa öyle pek rağmen sadece sanki sekiz seksen sen senden seni senin siz sizden sizi sizin şey şeyden şeyi şeyler şöyle şu şuna şunda şundan şunları şunu tarafından trilyon tüm üç üzerine var vardı ve veya ya yani yapacak yapılan yapılması yapıyor yapmak yaptı yaptığı yaptığını yaptıkları yedi yerine yetmiş yine yirmi yoksa yüz zaten"
stop_word = stop_word.split()

#kök indirgeme ve stop words işlemleri
def stem_word(value):

    stemmer = sn.stemmer("turkish")
    value = value.lower()
    value = stemmer.stemWords(value.split())
    stop_words = stop_word
    
    value = [item for item in value if not item in stop_words]
    value = ' '.join(value)
    return value

# ön işlem fonksiyonlarının sırayla çağırılması
def pre_processing(value):
    return [remove_numeric(remove_emoji
                          (remove_single_chracter
                           (remove_link
                            (remove_hashtag
                             (remove_username
                              (stem_word(word))))))) for word in value.split()]

# Boşlukların kaldırılması
def remove_space(value):
    return [item for item in value if item.strip()]






# word2vec model oluşturma ve kaydetme
def word2vec_create(value):
    model = Word2Vec(sentences = value.tolist(),vector_size=100,window=5,min_count=1)
    model.save("data/word2vec.model")

# word2vec model yükleme ve vektör çıkarma
def word2vec_analysis(value):
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

# word2vec model güncellenir.
def word2vec_update(value):
    model = Word2Vec.load("data/word2vec.model")
    model.build_vocab(value.tolist(),update=True)
    model.save("data/word2vec.model")


if __name__ == '__main__':
   
    # veri temizlemesi için örnek veri kümemiz okunur.
    df_1 = pd.read_csv("data/nlp.csv",index_col=0)


    ### tanımlanan df_1 içerisinde Text sütununu ön işlem fonksiyonlarından geçirerek Text_2 olarak df_1 içerisinde yeni bir sütun oluşturun. ###
    df_1["Text_2"] = df_1["Text"].apply(pre_processing)
    df_1["Text_2"] = df_1["Text_2"].apply(remove_space)

    ### df_1 içerisinde Text_2 sütununda boş liste kontrolü ###
    df_1[df_1["Text_2"].str[0].isnull()]

    df_1_index = df_1[df_1["Text_2"].str[0].isnull()].index
    df_1 = df_1.drop(df_1_index)
    df_1 = df_1.reset_index()

    del df_1["index"]

    df_1[df_1["Text_2"].str[0].isnull()]
    
    ### word2vec model oluşturma ###
    
    model = Word2Vec(sentences = df_1["Text_2"].tolist(), vector_size=100, window=5, min_count=1)
    model.save("data/word2vec.model")
    
    # df_1 dataframe mizi artık kullanmaycağımızdan ram de yer kaplamaması adına boş bir değer ataması yapıyoruz.
    df_1 = {}

    #############################################################################################################################################

    # sınıflandırma yapacağımız veri okunur.
    df_2 = pd.read_csv("data/metin_siniflandirma.csv",index_col=0)

    ### tanımlanan df_2 içerisinde Text sütununu ön işlem fonksiyonlarından geçirerek Text_2 olarak df_2 içerisinde yeni bir sütun oluşturun. ###
    df_2 = pd.read_csv("data/metin_siniflandirma.csv", index_col=0)
    df_2["Text_2"] = df_2["Text"].apply(pre_processing)
    df_2["Text_2"] = df_2["Text_2"].apply(remove_space)

    ### df_2 içerisinde Text_2 sütununda boş liste kontrolü ###
    df_2_index = df_2[df_2["Text_2"].str[0].isnull()].index
    df_2 = df_2.drop(df_2_index)
    df_2 = df_2.reset_index()
    del df_2["index"]
    df_2.head()
    
    ### sınıflandırma yapacağımız df_2 içerisinde bulunan Text_2 sütun verisini word2vec verisinde güncelleyin. ### 
    model = Word2Vec.load("data/word2vec.model")
    model.build_vocab(df_2["Text_2"].tolist(),update=True)
    model.save("data/word2vec.model")

    ### Text_2 sütun üzerinden word2vec adında bu modeli kullanarak yeni bir sütun yaratın
    df_2["word2vec"] = df_2["Text_2"].apply(word2vec_analysis)


    ### word2vec sütunumuzu train test olarak bölün ###
    msg_train, msg_test, label_train, label_test = train_test_split(df_2["word2vec"].tolist(), df_2["Label"].tolist(), test_size=0.2, random_state=42)


    ### svm pipeline oluştur, modeği eğit ve test et ###
    svm = Pipeline([('svm',LinearSVC())])
    svm.fit(msg_train,label_train)
    y_pred_class = svm.predict(msg_test)

    
    ### accuracy ve f1 score çıktısını print ile gösterin. ###
    print("svm accuracy score:", accuracy_score(label_test,y_pred_class))
    print("svm accuracy score:", f1_score(label_test,y_pred_class,average="weighted"))