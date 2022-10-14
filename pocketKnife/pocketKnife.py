import pandas as pd
import numpy as np

import multiprocessing
from multiprocessing import Pool
from functools import partial
import tqdm

import spacy
spacy.prefer_gpu()
from spacy.tokens.doc import Doc
import scipy.sparse as sp

# https://stackoverflow.com/questions/33945261/how-to-specify-multiple-return-types-using-type-hints
from typing import Iterable, List, Union, Tuple
from typing_extensions import Literal
from pathlib import Path
import pickle
from collections import Counter

# Criação do modelo de linguagem SBERT
# https://www.sbert.net/examples/applications/semantic-search/README.html
from sentence_transformers import SentenceTransformer, util, LoggingHandler

import logging
import time

from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%d-%m-%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# todo
# arquivos com listas longas daqui [back]
# add remove nomes próprios bem no inicio [sim]

nlp = spacy.load("pt_core_news_lg")
nlp.max_length=9000000000
matcher = spacy.matcher.Matcher(nlp.vocab)

class SaveLoad():
    def __init__(self,
        # home_path: str = '/home/jupyter', # no colab
        # data_subfolder: str = 'data/raw', # no colab
        home_path: str,
        data_subfolder: str,
        models_subfolder: str) -> None:
        
        self.folder_home_path = Path(home_path)
        self.folder_data_path = self.folder_home_path / data_subfolder
        self.folder_models_path = self.folder_home_path / models_subfolder
        self.clean_df_path = self.folder_data_path / 'df_clean.parquet'
        self.embedder_path = self.folder_models_path / 'embedder_sbert.dat'
        self.corpus_embedding_path = self.folder_models_path / 'corpus_embedding.dat'
        self.pickle_data_path = self.folder_data_path
        self.pickle_model_path = self.folder_models_path
    

    def to_pickle(self, obj: object, filename: str) -> None:
        path = self.folder_models_path / filename
        start = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        end = time.time()
        logging.info(f'Object saved to {path} in {end - start} seconds')
    
    def from_pickle(self, filename: str) -> object:
        path = self.folder_models_path / filename
        start = time.time()
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        end = time.time()
        logging.info(f'Object loaded from {path} in {end - start} seconds')
        return obj

    def to_parquet(self, df: pd.DataFrame, filename: str) -> None:
        path = self.folder_data_path / filename
        start = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        end = time.time()
        logging.info(f'Dataframe saved to {path} in {end - start} seconds')
    
    def from_parquet(self, filename: str, columns: Union[list, None] = None) -> pd.DataFrame:
        path = self.folder_data_path / filename
        start = time.time()
        df = pd.read_parquet(path, columns=columns)
        end = time.time()
        logging.info(f'Dataframe loaded from {path}, with {len(df)} rows, in {end - start} seconds')
        return df
    
    def from_csv(self, filename: str) -> pd.DataFrame:
        path = self.folder_data_path / filename
        start = time.time()
        df = pd.read_csv(path, sep="\-\|\-", encoding='ISO-8859-1', engine='python')
        end = time.time()
        logging.info(f'Dataframe loaded from {path} in {end - start} seconds')
        return df
    
    def save_embedder_and_corpus_embeddings(self,
        embedder: SentenceTransformer,
        corpus_embeddings: np.ndarray,
        used_embedder: str,
        dataset_filter: str) -> None:
            self.to_pickle(embedder, f'{used_embedder}_{dataset_filter}_embedder.dat')
            self.to_pickle(corpus_embeddings, f'{used_embedder}_{dataset_filter}_corpus_embeddings.dat')

    def load_embedder_and_corpus_embeddings(self,
        used_embedder: str,
        dataset_filter: str) -> Tuple[np.ndarray, SentenceTransformer]:
            embedder = self.from_pickle(f'{used_embedder}_{dataset_filter}_embedder.dat')
            corpus_embeddings = self.from_pickle(f'{used_embedder}_{dataset_filter}_corpus_embeddings.dat')
            return corpus_embeddings, embedder

# qtd_cores = multiprocessing.cpu_count() - 1
qtd_cores = multiprocessing.cpu_count()

################################################################################################
## Expressão regular para a remoção de campos de marcação de tempo                            ##
hour_patterns = r"(\d{1,2}:\d{1,2})|(\d{1,2}h\d{1,2})|(\d{1,2}x\d{1,2})|(\d{1,2}h)|(\d{1,2})" ##
################################################################################################

##################################################################################
## Somente as palavras cujas tags estiverem fora desta lista serão consideradas ##
## Universal POS tags: https://universaldependencies.org/u/pos/                 ##
##################################################################################
list_remove_tags = [
    # "PRON",
    "PUNCT",
    "CCONJ",
    "DET",
    # "ADP",
    "INTJ",
    "NUM",
    "SPACE",
    # "VERB",
    "AUX",
    "PART",
    "SYM",
    "X",
]


def tfidf_path(i, data_subfolder: Path):
    return data_subfolder / f'X_df_tfidf_25_perc_{i}.parquet'


def cleanString(series: str) -> str:
    """Remove os caracteres indesejados contidos em replacement_dict.

    Converte dataframe[column] para minúsculo, antes de comparar.

    Parameters
    ----------
        series: string
            Texto

    Returns
    ----------
        series: string
            Texto limpo.

    See also
    ----------
    Ref replacement_dict:
    https://ajuda.locaweb.com.br/wiki/caracteres-acentuados-hospedagem-de-sites/
    """
    from unidecode import unidecode_expect_ascii
    replacement_dic = {
        "&aacute;": "a",
        "&eacute;": "e",
        "&iacute;": "i",
        "&oacute;": "o",
        "&uacute;": "u",
        "&atilde;": "a",
        "&otilde;": "o",
        "&uuml;": "u",
        "&ntilde;": "n",
        "&amp;": "&",
        "&ccedil;": "c",
        "&acirc;": "a",
        "&ecirc;": "e",
        "&ocirc;": "o",
        "&agrave;": "a",
        "&nbsp;": " ",
        # "ã": "a",
        # "á": "a",
        # "â": "a",
        # "à": "a",
        # "é": "e",
        # "ê": "e",
        # "í": "i",
        # "ó": "o",
        # "õ": "o",
        # "ô": "o",
        # "ú": "u",
        # "ü": "u",
        # "ç": "c",
        "<br>": " ",
        "<br />": " ",
        "</strong>": " ",
        "<strong>": " ",
        "\r": " ",
        "\n": " ",
        "<ul>": " ",
        "</ul>": " ",
        "<li>": " ",
        "</li>": " ",
        "<p>": " ",
        "</p>": " ",
        "<b>": " ",
        "</b>": " ",
        "&ordm;": " ",
        "<u>": " ",
        "</u>": " ",
        '<span style="color: #000000;">': " ",
        '<span style="font-size: medium;">': " ",
        "</span>": " ",
        "&bull": " ",
        "/": " ",
        ".": " ",
        "-": " ",
        "+": " ",
        ":": " ",
        ";": " ",
        "?": " ",
        "*": " ",
        "|": " ",
        "•": " ",
        "°": " ",
        "@": " ",
        "#": " ",
        "$": " ",
        "%": " ",
        "&": " ",
        "(": " ",
        ")": " ",
        "_": " ",
        "=": " ",
        "[": " ",
        "]": " ",
        "{": " ",
        "}": " ",
        "ª": " ",
        "º": " ",
        "<": " ",
        ">": " ",
        "www": " ",
        "~": " ",
        # "§": " ", # avaliar se remove ou não, por enquanto remove
        "§": " ",
        "xxxxxx": " ",
    }
    for x, y in replacement_dic.items():
        series = str(series).lower()
        series = series.replace(x, y)
        series = unidecode_expect_ascii(series)
    return series


def get_stopwords() -> list:
    """Retorna um dicionário com a lista de stopwords para português.

    Exemplo:
    {
        'a':1
        'o':1
        'para':1
    }

    Parameters
    ----------
        none

    Returns
    ----------
        stopwords: list of strings
            Lista de stopwords.
    """
    stopwords = [
        "de",
        "a",
        "á",
        "o",
        "ó",
        "que",
        "quê",
        "e",
        "é",
        "do",
        "dó",
        "dê",
        "da",
        "em",
        "um",
        "para",
        "pará",
        "com",
        "não",
        "uma",
        "os",
        "no",
        "nó",
        "se",
        "sé",
        "na",
        "por",
        "mais",
        "as",
        "dos",
        "como",
        "mas",
        "ao",
        "ele",
        "das",
        "à",
        "seu",
        "sua",
        "suã",
        "ou",
        "quando",
        "muito",
        "nos",
        "já",
        "eu",
        "também",
        "só",
        "sô",
        "pelo",
        "pêlo",
        "pela",
        "péla",
        "até",
        "at",
        "tc",
        "isso",
        "ela",
        "entre",
        "depois",
        "sem",
        "mesmo",
        "aos",
        "seus",
        "quem",
        "nas",
        "me",
        "mé",
        "esse",
        "eles",
        "você",
        "voc",
        "essa",
        "num",
        "nem",
        "suas",
        "meu",
        "às",
        "ás",
        "minha",
        "numa",
        "pelos",
        "elas",
        "qual",
        "nós",
        "lhe",
        "deles",
        "essas",
        "esses",
        "pelas",
        "este",
        "dele",
        "tu",
        "te",
        "vocês",
        "vos",
        "lhes",
        "meus",
        "minhas",
        "teu",
        "tua",
        "teus",
        "tuas",
        "nosso",
        "nossa",
        "nossos",
        "nossas",
        "dela",
        "delas",
        "esta",
        "estes",
        "estas",
        "aquele",
        "aquela",
        "aqueles",
        "aquelas",
        "isto",
        "aquilo",
        "estou",
        "está",
        "estamos",
        "estão",
        "estive",
        "esteve",
        "estivemos",
        "estiveram",
        "estava",
        "estávamos",
        "estavam",
        "estivera",
        "estivéramos",
        "esteja",
        "estejamos",
        "estejam",
        "estivesse",
        "estivéssemos",
        "estivessem",
        "estiver",
        "estivermos",
        "estiverem",
        "hei",
        "há",
        "havemos",
        "hão",
        "houve",
        "houvemos",
        "houveram",
        "houvera",
        "houvéramos",
        "haja",
        "hajamos",
        "hajam",
        "houvesse",
        "houvéssemos",
        "houvessem",
        "houver",
        "houvermos",
        "houverem",
        "houverei",
        "houverá",
        "houveremos",
        "houverão",
        "houveria",
        "houveríamos",
        "houveriam",
        "sou",
        "somos",
        "são",
        "era",
        "éramos",
        "eram",
        "fui",
        "foi",
        "fomos",
        "foram",
        "fora",
        "fôramos",
        "seja",
        "sejamos",
        "sejam",
        "fosse",
        "fôssemos",
        "fossem",
        "for",
        "formos",
        "forem",
        "ser",
        "serei",
        "será",
        "seremos",
        "serão",
        "seria",
        "seríamos",
        "seriam",
        "ter",
        "tenho",
        "tem",
        "temos",
        "tém",
        "tinha",
        "tínhamos",
        "tinham",
        "tive",
        "teve",
        "tivemos",
        "tiveram",
        "tivera",
        "tivéramos",
        "tenha",
        "tenhamos",
        "tenham",
        "tivesse",
        "tivéssemos",
        "tivessem",
        "tiver",
        "tivermos",
        "tiverem",
        "terei",
        "terá",
        "teremos",
        "terão",
        "teria",
        "teríamos",
        "teriam",
    ]
    stopwords = {sw: 1 for sw in stopwords}
    return stopwords


def get_tokens_clean(text: str) -> list:
    import re  # Regular Expression
    """Remove stopwords e demais caracteres.

    Cria tokens a partir de um texto e remove as stopwords.

    Parameters
    ----------
        text: string

    Returns
    ----------
        tokens: list of strings
            Texto sem as palavras stopwords e outros caracteres indesejados.

    """
    text = str(text)
    text = text.lower()
    text = re.sub(r"[\.,\(\);\[\]\:]", " ", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    tokens = [token for token in text.split() if token not in get_stopwords()]
    return tokens


def plot_heatmap(df: pd.DataFrame) -> None:
    """Gera um gráfico mapa de calor

    Parameters
    ----------
        df: pd.DataFrame

    Returns
    ----------
        none
            plota gráfico de mapa de calor.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(7, 5))
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis")


def do_wordcloud(
    tokens_list: list,
    titles_list: list,
    nrows,ncols, main_title: str="Gráfico",
    figsize: tuple=(15, 10)
) -> None:
    """Gerar uma wordcloud

    Visualização de palavras cujo o tamanho é dimensionado conforme a
    frequência de cada palavra.

    Args:
    ----------
        tokens: list of string
            Palavras para formarem base do Wordcloud
        title: string
            Título de Gráfico

    Returns:
    ----------
        none
            Plota gráfico de mapa de calor.
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    if(len(tokens_list) != len(titles_list)):
        raise ValueError("tokens_list and titles_list must have the same length")

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        tight_layout=True, figsize=figsize
    )
    fig.set_label(main_title)
    # ax1, ax2, ax3, ax4 = axes.flatten()
    for i, ax in enumerate(axes.flatten()):
        tokens = tokens_list[i]
        title = titles_list[i]
        wordcloud = WordCloud(
            width=800,
            height=800,
            background_color="white",
            stopwords=get_stopwords(),
            min_font_size=10,
        ).generate_from_frequencies(tokens)

        ax.imshow(wordcloud)
        ax.set_title(title)
        plt.axis("off")
        plt.tight_layout(pad=0)
    plt.show()
    return


def do_bar_chart(tokens_list: list, titles_list: list, nlargest: int) -> None:
    """Gera o gráfico de barras verticais

    Visualização de palavras cujo o tamanho é dimensionado conforme a
    frequência de cada palavra.

    Parameters
    ----------
        tokens: list of string
            Palavras para formarem base do Wordcloud
        title: string
            Título de Gráfico

    Returns
    ----------
        none
            Plota gráfico de barra
    """
    import matplotlib.pyplot as plt

    if(len(tokens_list) != len(titles_list)):
        raise ValueError("tokens_list and titles_list must have the same length")

    # plt.rcParams['axes.labelsize'] = 30
    # plt.rcParams['axes.titlesize'] = 16
    plt.figure(figsize=(15, 5))
    for i in range(len(tokens_list)):
        tokens = tokens_list[i]
        title = titles_list[i]
        plt.subplot(1, 2, i+1)
        pd.Series(tokens).nlargest(nlargest).sort_values(ascending=False).plot(kind="bar")
        plt.title(title)
        plt.xlim([-0.5, nlargest])
    plt.show()


def remove_pos_tags_list(
    text_list: list, list_remove_tags: list
) -> pd.DataFrame:
    """Remoção de palavras marcadas com 'Pos-Tags' contidas em list_remove_tags

    Parameters
    ----------
        text_list: list of string
            Lista de palavras marcadas com o spaCy Pos-Tag.
        list_remove_tags: list
            Lista contendo as tags que serão removidas.
    Returns
    ----------
        df: pd.DataFrame
            Pandas DataFrame coluna x linhas com as palavras cujas tags não
            estão contidas em list_remove_tags.
    """
    words_list = []
    for doc_iter in (nlp(txt_iter) for txt_iter in text_list):
        words = str()
        for word in doc_iter:
            if word.pos_ not in list_remove_tags:
                if words == "":
                    words += word.text
                words += " " + word.text
        words_list.append(words)
    df = pd.DataFrame(words_list)
    return df


def do_clean_low_freq(txt: str, n: int) -> pd.DataFrame:
    """Remove palavras que aparecem com 'n' frequencia

    Parameters
    ----------
        txt: str
        n: int
            Nro. inteiro da quantidade mínima de frequência das palavras.
    Returns
    ----------
        df: pd.DataFrame
            Pandas DataFrame coluna x linhas com as palavras cujas frequências
            no texto são menores que 'n'.
    """
    from collections import defaultdict

    frequency = defaultdict(int)
    for text in txt:
        for token in text:
            frequency[token] += 1
    txt = [[token for token in text if frequency[token] > n] for text in txt]
    return pd.DataFrame([txt])


def show_text_tagger(
    text: str, nlp: spacy.language.Language
) -> spacy.tokens.Token:
    """Imprime palavra do token e Pos-Tag associada

    Parameters
    ----------
        text: str
        nlp: spacy.language.Language

    Returns
    ----------
        t: spacy.tokens.Token
    """
    text = nlp(text)
    text_list = list()
    txt = str()
    for t in text:
        txt += t.text + "/" + t.pos_ + " "
    text_list.append(txt)
    return text_list


def tokenization(text: str, nlp: spacy.language.Language) -> pd.DataFrame:
    """Tokenização de texto

    Parameters
    ----------
        text: str
        nlp: spacy.language.Language

    Returns
    ----------
        pd.DataFrame
    """
    # nlp.max_length = 9000000 # old_nlp.max_length
    nlp.max_length = 100000000
    doc = nlp(text)
    list_txt = list()
    for token in doc:
        list_txt.append(token.text)
    return pd.DataFrame(list_txt)


def lemmatize_text(
    text_list: list, nlp: spacy.language.Language
) -> pd.DataFrame:
    """Lemmatização de texto

    Parameters
    ----------
        text_list: list of Strings
        nlp: spacy.language.Language

    Returns
    ----------
        df: pd.DataFrame
    """
    nlp.max_length = 100000000
    words_lemma_list = list()
    for i, text in enumerate(text_list):
        words_lemma = str()
        text = nlp(str(text))
        for word in text:
            words_lemma = words_lemma + " " + word.lemma_
        words_lemma_list.append(words_lemma)
    df = pd.DataFrame(words_lemma_list)
    return df


def do_bow_tfidf(
    corpus: np.ndarray,
    ngram_range: tuple,
    max_features: Union[int, None] = None,
    max_df: Union[int, float, None] = 1.0,
    min_df: Union[int, float, None] = 1
) -> Tuple[list, sp.csr_matrix]:
    """Treina um Vectorizer e retorna um bag of words

    Parameters
    ----------
        corpus: numpy.ndarray
        ngram_range: tuple
        max_features: int or None, default None
        max_df: int or float, default 1.0
        min_df: int or flaot, default 1

    Returns
    ----------
        vectorizer.get_feature_names(): list
        bow: sparse matrix

    See Also
    ----------
    pocketnife_nlp.do_bow_inverse_ratio_tfidf: fórmula alternativa de calcular bag of words.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range
        , max_features=max_features
        , max_df=max_df
        , min_df=min_df
    )
    bow = vectorizer.fit_transform(corpus)
    return vectorizer.get_feature_names(), bow


def bow_inverse_ratio_tfidf_word_agg(
    corpus: np.ndarray,
    ngram_range: tuple,
    max_features: Union[int, None] = None,
    max_df: Union[int, float, None] = 1.0,
    min_df: Union[int, float, None] = 1
) -> Tuple[list, sp.csr_matrix]:
    """Treina um Vectorizer e retorna um bag of words

    Diferente da função pocketnife_nlp.do_bow_tfidf esta apresenta uma fórmula customizada
    para calcular o Bag of Words sobre n_grams do tipo (2,2) ou maiores e menos frequentes 
    como, por exemplo, sintagmas nominais (noun phrases).

    Parameters
    ----------
        corpus: numpy.ndarray
        ngram_range: tuple
        max_features: int or None, default None
        max_df: int or float, default 1.0
        min_df: int or float, default 1

    Returns
    ----------
        vectorizer.get_feature_names(): list
        bow: sparse matrix

    See Also
    ----------
    pocketnife_nlp.do_bow_tfidf: método padrão de calcular bag of words.
    ref: https://stackoverflow.com/questions/27488446/
    how-do-i-get-word-frequency-in-a-corpus-using-scikit-learn-countvectorizer
    """
    from sklearn.feature_extraction.text import (
        CountVectorizer,
        TfidfVectorizer,
    )
    from math import log

    cv = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
    )
    cv_fit = cv.fit_transform(corpus, None)
    tf = cv_fit.toarray().sum(axis=0)

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
    )
    vectorizer.fit(corpus)
    idf = vectorizer.idf_
    bow = (1 / tf) / (idf / log(10))
    return vectorizer.get_feature_names(), bow


def add_rules(
    matcher: spacy.matcher.Matcher,
    optional_new_parameters: Union[
        Iterable[Iterable[str]],
        None
    ] = None) -> spacy.matcher.Matcher:
    """Cria padrões de matcher para a extração dos sintagmas nominais

    Parameters
    ----------
        matcher: spacy.matcher.Matcher
            Matcher sem regras

    Returns
    ----------
        matcher: spacy.matcher.Matcher
            Matcher com regras

    See Also
    ----------
    ref: Sequências de correspondência de tokens, com base em padrão de regras
        ref #1: https://spacy.io/usage/rule-based-matching
        ref #2: https://spacy.io/api/matcher
        ref #3: https://spacy.io/api/annotation#pos-tagging
        ref #4: https://universaldependencies.org/u/pos

    """
    old_patterns = [
        ## sintagmas nominais nível 5
        [
            "NOUN",
            "NOUN",
            "NOUN",
            "NOUN",
            "ADJ",
        ],  ## nova regra - existem ocorrências de termos importantes com esta regra
        ## sintagmas nominais nível 4
        [
            "NOUN",
            "NOUN",
            "NOUN",
            "ADJ",
        ],  ## nova regra - existem ocorrências de termos importantes com esta regra
        ## sintagmas nominais nível 3
        [
            "NOUN",
            "ADJ",
            "ADJ",
        ],  ## nova regra - existem ocorrências de termos importantes com esta regra
        [
            "NOUN",
            "NOUN",
            "ADJ",
        ],  ## nova regra - existem ocorrências de termos importantes com esta regra
        ["NOUN", "ADJ", "NOUN"],
        ["NOUN", "ADJ", "PROPN"],
        ["PROPN", "NOUN", "ADJ"],
        ["PROPN", "ADJ", "NOUN"],
        ## sintagmas_nominais nível 2
        ["PROPN", "ADJ"], 
        # ["PROPN", "PROPN"],
        # ["PROPN", "NOUN"],
        ["NOUN", "ADJ"],
        # ["NOUN", "NOUN"],
        # ["NOUN", "PROPN"],
        ## sintagmas nominais nível 1
        # ["NOUN"],
        # ["ADJ"],
        # ["PROPN"],
    ]

    patterns = [  
        ## sintagmas nominais nível 3
        ["PROPN","ADP","NOUN"],
        ["PROPN","ADJ","NOUN"],
        ["NOUN","ADJ","NOUN"],

        ## sintagmas nominais nível 2
        ["PROPN","PROPN"],
        ["PROPN","VERB"],
        ["NOUN","VERB"],
        ["NOUN","ADJ"],
        ["PROPN","ADJ"],
        ["NOUN","PROPN"],    
        ["NOUN","NOUN"],
        ["PROPN","NOUN"],

        ## sintagmas nominais nível 1
        ["NOUN"],
        ["PROPN"],
        ["ADJ"],
        ["VERB"],
    ]
    # patterns = old_patterns

    if optional_new_parameters:
        patterns = optional_new_parameters

    # Iteração que coloca em formato de dicionário os padrões PoS-Tag contidos em patterns
    patterns_list = list()
    for i, p in enumerate(patterns):
        pattern = [
            {"POS": pi} for pi in patterns[i]
        ]  # iteração que cria o dicionário a partir de patterns
        matcher.add(
            "id-" + str(i), [pattern]
        )  # cria uma lista de listas de dicionários
        rules = "id-" + str(i) + str([pattern])
        patterns_list.append(rules)
    #     print(patterns_list)
    return matcher

matcher = add_rules(matcher)

def extract_patterns(
    text_list: list,
    corpus_and_bow: Literal['corpus','bow'],
    analyze_mode: bool = False,
) -> pd.DataFrame:
    """Função de extração de termos conforme padrões de match

    As regras de sintagmas nominais nível 5 são preferíveis,
    porque expressam um sentido e contexto mais completo e
    na sequencia os sintagmas nível 4, 3, 2 e 1.

    Parameters
    ----------
        text_list: list
        matcher: spacy.matcher.Matcher
            Matcher com regras
        nlp: spacy.language.Language

    Returns
    ----------
        df: pd.DataFrame
    """
    corpus_list = list()
    bow_list = list()
    analyze_corpus_list = list()
    for doc in (nlp(text) for text in text_list):
        matches = matcher(doc)
        current_doc_bow = ""
        current_doc_corpus = ""
        analyze_current_doc_corpus = []
        for match_id, start, end in matches:
            string_id = nlp.vocab.strings[match_id]
            match_span = doc[start:end]
            if string_id[:3] == "id-":
                match_span_text = list(str(match_span.text).split())
                matcher_ngram = ""
                if analyze_mode:
                    analyze_current_doc_corpus.append((match_span_text, string_id))
                for word in match_span_text:
                    if 'corpus' in corpus_and_bow:
                        if word not in current_doc_corpus:
                            if current_doc_corpus != "":
                                current_doc_corpus+=" "
                            current_doc_corpus += word
                    if 'bow' in corpus_and_bow:
                        if matcher_ngram != "":
                            matcher_ngram+="_"
                        matcher_ngram += word
                if 'bow' in corpus_and_bow:
                    if current_doc_bow != "":
                        current_doc_bow+=" "
                    current_doc_bow += matcher_ngram
        if 'corpus' in corpus_and_bow:
            corpus_list.append(current_doc_corpus)
        if 'bow' in corpus_and_bow:
            bow_list.append(current_doc_bow)
        if analyze_mode:
            analyze_corpus_list.append(analyze_current_doc_corpus)
    if 'corpus' in corpus_and_bow:
        df = pd.DataFrame({'features_corpus': corpus_list})
    if 'bow' in corpus_and_bow:
        df1 = pd.DataFrame({'features_bow': bow_list})
        if 'corpus' not in corpus_and_bow:
            df = df1
        else:
            df = pd.concat([df, df1],axis=1)
    if analyze_mode:
        df1 = pd.DataFrame({'analyze_corpus': analyze_corpus_list})
        if ('corpus' not in corpus_and_bow) and ('bow' not in corpus_and_bow):
            df = df1
        else:
            df = pd.concat([df, df1],axis=1)
    return df


def do_preprocess(
    corpus_and_bow: Literal['corpus','bow'],
    df: pd.Series,
    analyze_mode: bool = False,
) -> pd.DataFrame(columns=['features_corpus','features_bow']):
    """Executa pré-processamento em dados

    Função que realiza os passos de pré-processamento da
    extração de sintagmas nominais.

    Parameters
    ----------
        corpus_and_bow: Literal['corpus','bow']
            contains one of list or both
        df: pd.Series

    Returns
    ----------
        X: pd.DataFrame
    """
    X = pd.DataFrame()
    X['features'] = remove_pos_tags_list(
        df.tolist(),
        list_remove_tags
    )
    X['features'] = X['features'].replace(
        hour_patterns, " ", regex=True
    )
    X['features'] = X.features.apply(
        lambda x: apply_parser_clean(x, forward_from='clean')
    )
    X = pd.concat(
        [
            X,
            extract_patterns(
                X['features'].tolist(),
                corpus_and_bow=corpus_and_bow,
                analyze_mode=analyze_mode,
            )
        ],
        axis=1
    )
    if 'corpus' in corpus_and_bow:
        X['features_corpus'] = X.features_corpus.apply(
            lambda x: apply_parser_clean(x, forward_from='strip')
        )
    if 'bow' in corpus_and_bow:
        X['features_bow'] = X.features_bow.apply(
            lambda x: apply_parser_clean(x, forward_from='strip')
        )
    return X


def return_multiprocessing_jobs_and_works(n_jobs: int, n_workers: int, len_df: int) -> Tuple[int, int]:
    """Retorna número de jobs e works para multiprocessing

    Parameters
    ----------
        n_jobs: int
        n_works: int

    Returns
    ----------
        n_jobs: int
        n_works: int
    """
    if (n_workers <= 0) or (n_workers > qtd_cores):
        n_workers = qtd_cores
    if (type(n_jobs) == float):
        n_jobs = int(n_jobs * n_workers)
    if (n_jobs <= 0) or (n_jobs > len_df):
            n_jobs = len_df
    if n_workers > n_jobs:
        n_workers = n_jobs
    return (n_jobs, n_workers)


def parallelize_do_preprocess(
    df: pd.Series,
    corpus_and_bow: Literal['corpus','bow'],
    n_jobs: Union[int,float] = 2.0,
    n_workers: int = -1
) -> pd.Series:
    """Paraleliza o pré-processamento em dados

    Função que otimiza o uso dos núcleos para o multiprocessamento
    e otimiza o tempo.

    Em testes feitos o desempenho foi melhor do que o paralelismo
    padrão fornecido pelo nlp.pipe(n_threads=-1)

    Parameters
    ----------
        df: pd.Series
        n_jobs: int or float
            caso int é o número absoluto de jobs, -1 torna len(df) como n_jobs
            caso float é um multiplicador baseado no número de n_workers, caso exceda len(df) é ajustado para len(df)
        n_workers: int
            Define a quantidade de threads, -1 para usar todos os núcleos

    Returns
    ----------
        pd.Series

    See also:
    ----------
    ref #1: https://docs.python.org/pt-br/3/library/functools.html
    ref #2: https://www.learnpython.org/en/Partial_functions
    """
    literal_list = ['corpus','bow']
    for i in corpus_and_bow:
        if i not in ['corpus','bow']:
            raise ValueError(f'corpus_and_bow must be one of Literal{literal_list} or both')
    n_jobs, n_workers = return_multiprocessing_jobs_and_works(n_jobs, n_workers, len(df))
    with Pool(n_workers) as p:
        logging.info(f"[logging] PreProcess | {n_workers} workers | {n_jobs} jobs |")
        df_split = np.array_split(df, n_jobs)
        func = partial(do_preprocess, corpus_and_bow)
        return pd.concat(tqdm.tqdm(p.imap(func, df_split), total=n_jobs)).reset_index(drop=True)


def create_tf_idf_features(x: pd.Series, features_filter: list) -> str:
    """Seleciona palavras que estão no features filter

    Parameters
    ----------
        x: pd.Series
        features_filter: list

    Returns
    ----------
        tf_idf_features: str
    """
    tf_idf_features = ""
    for word in x['features']:
        if word in ' '.join(features_filter):
            tf_idf_features += word
    return tf_idf_features


def do_inverse_ratio_tfidf_by_categories(
    texto: list
    , col: str
    , clean_df_path: str
    , corpus: np.ndarray
    , ngram_range: tuple
    , max_features: Union[int, None] = None
    , max_df: Union[int, float, None] = 1.0
    , min_df: Union[int, float, None] = 1
                        
) -> pd.DataFrame:
    """Aplica tf-idf por categoria

    Por questões de paralelismo, o dataset não é passado como parâmetro
    mas sim lido durante esta função. Excederia o tamanho máximo de
    memória de váriavel que pode ser repassada a diversos subprocessos.

    Parameters
    ----------
        texto: list of Strings
            Lista de categorias, feita com df["coluna_desejada"].unique()
            e divida por meio de paralelismo
        col: String informando a coluna desejada
        corpus: numpy.ndarray
        ngram_range: tuple
        max_features: int or None, default None
        max_df: int or float, default 1.0
        min_df: int or float, default 1

    Returns
    ----------
        df_filter_all: pd.DataFrame
    """
    df_filter_all = pd.DataFrame()
    df = pd.read_parquet(clean_df_path)
    for txt in texto:
        df_filter = df[df[col]==txt].copy()
        features_filter, bow_filter = bow_inverse_ratio_tfidf_word_agg(
            df_filter['features_bow'].values.astype('U'),
            # corpus=corpus,
            ngram_range=ngram_range,
            max_features=max_features, # delimita a frequência total extraídas
            max_df=max_df, # delimita a frequência máxima de features por texto
            min_df=min_df, # delimita a frequência mínima de features por texto
        )
        df_filter[col] = df_filter.apply(
            lambda x: create_tf_idf_features(x, features_filter)
            , axis=1
        )
        if df_filter_all.empty:
            df_filter_all = df_filter.copy()
        else:
            df_filter_all = df_filter_all.append(df_filter)
    return df_filter_all


def parallelize_anything(
    func: object,
    data: Union[pd.DataFrame, pd.Series, list],
    n_jobs: Union[int,float] = 2.0,
    n_workers: int = -1,
    data_already_split: bool = False,
    merge_function: object = pd.concat,
    **kwargs
) -> Union[pd.DataFrame, pd.Series]:
    """Paralelização coringa

    Esta paralelização é chamada de coringa, pois permite a
    escolha de uma função a ser paralelizada, um objeto data
    que será dado split para dividir porções de si para cada
    subprocesso, e *args para receber argumentos extras que
    serão passados dentro de uma lista para cada subprocesso.

    O retorno de func é limitado por objetos que possam ser
    aplicados pd.Concat(*), como pd.Series ou pd.DataFrame

    Parameters
    ----------
        func: function
            Qualquer método definido no python
        data: any iterable
        *args: any numbers of arguments
            Poderá ser passado qualquer quantidade de argumentos.
            Os mesmos serão passado dentro de uma lista para
            a função dentro do subprocesso filho.
    Returns
    ----------
        pd.concat: pd.DataFrame or pd.Series
    """
    n_jobs, n_workers = return_multiprocessing_jobs_and_works(n_jobs, n_workers, len(data))
    with Pool(n_workers) as p:
        if(data_already_split):
            data_split = data
        else:
            data_split = np.array_split(data, n_jobs)
        # arg_split = np.array_split(args[1], n_jobs)
        if kwargs.items != ():
            func = partial(func, **kwargs)
        return merge_function(
            tqdm.tqdm(
                p.imap(
                    func,
                    data_split,
                    # zip(data_split, arg_split)
                ),
                total=n_jobs
            )
        )


def do_corpus_embeddings_gpu(
    corpus: pd.Series, 
    used_embedder: Literal['multi-qa-mpnet-base-cos-v1','multi-qa-mpnet-base-dot-v1' , 'all-mpnet-base-v2', None]='multi-qa-mpnet-base-cos-v1',
    target_devices: Literal['cuda', 'cpu', None]='cuda',
    with_gpu_mode_process: bool=False,
    with_config_gpu_params: bool=False,
    n_chunk_size: Union[int, None]=None
) -> Tuple[np.ndarray, SentenceTransformer]:
    """Executa encoding e embedding dos sintagmas nominais

    Parameters
    ----------
        corpus: pd.Series
        col: str
    Returns
    ----------
        corpus_embeddings: np.ndarray
        embedder: SentenceTransformer
            Modelo de linguaguem multi-language a ser reutilizado 
            em outras partes do notebook

    """
    print(__name__)

    if __name__ in ['__main__','pocketknife_nlp','pocketKnife.pocketKnife']:
        
        import torch
        import py3nvml
        import multiprocessing
        import torch.multiprocessing as mp
        import os

        QTD_CORES=multiprocessing.cpu_count() - 1

        def gpu_mode_process(
            max_split_size_mb: Union[int, None]=128,
            set_start_method: Union['spawn', 'fork', None]='spawn'
        ):
            ## https://discuss.pytorch.org/t/keep-getting-cuda-oom-error-with-pytorch-failing-to-allocate-all-free-memory/133896/6
            os.environ["PYTORCH_CUDA_ALLOC_CONF"]=f'max_split_size_mb:{max_split_size_mb}'
            multiprocessing.set_start_method(f'{set_start_method}', force=True)

        def config_gpu_params(
            num_gpus: Union[int, list] = 1,
            gpu_select: Union[int, list] = 0,
            gpu_fraction: Union[float, None] = .95,
            max_procs: Union[int, None] = -1,
            gpu_min_memory: Union[int, None] = 256,
            # max_split_size_mb: Union[int, None] = None,
        ):
            py3nvml.utils.grab_gpus(
                num_gpus=num_gpus, # How many gpus your job needs (optional). Can set to -1 to take all
                            # remaining available GPUs.
                gpu_select=gpu_select, # A single int or an iterable of ints indicating gpu numbers to
                            # search through. If None, will search through all gpus.
                gpu_fraction=gpu_fraction, # The fractional of a gpu memory that must be free for the script to see
                                # the gpu as free. Defaults to 1. Useful if someone has grabbed a tiny
                                # amount of memory on a gpu but isn't using it.
                max_procs=max_procs, # Maximum number of processes allowed on a GPU (as well as memory restriction).
                env_set_ok=True, # If false, will complain if CUDA_VISIBLE_DEVICES is already set.
                gpu_min_memory=gpu_min_memory # The minimal allowed graphics card memory amount in MiB.
            )
            # os.environ["PYTORCH_CUDA_ALLOC_CONF"]=f'max_split_size_mb:{max_split_size_mb}'

        start = time.time()

        embedder = SentenceTransformer(used_embedder)

        if target_devices is None:
            if torch.cuda.is_available():
                # target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
                if with_config_gpu_params is True:
                    config_gpu_params()
                if with_gpu_mode_process is True:
                    gpu_mode_process()
                target_devices = ['cuda:{}'.format(i) for i in range(torch.cuda.device_count())]
            else:
                logging.info(f"CUDA is not available. Start {QTD_CORES} CPU worker")
                target_devices = ['cpu']*QTD_CORES
        
        pool = embedder.start_multi_process_pool()

        print(f'get_context: {mp.get_context()}')
        print(f'get_all_sharing_strategies: {mp.get_all_sharing_strategies()}')
        print(f'get_start_method: {mp.get_start_method()}')

        corpus_embeddings = embedder.encode_multi_process(
            sentences=corpus, 
            pool=pool, 
            batch_size=32, 
            chunk_size=n_chunk_size
        )
        
        embedder.stop_multi_process_pool(pool)
        end = time.time()
        
        logging.info(f"{end - start:.3f}s")
        return corpus_embeddings, embedder


def default_replace_missing_url(row):
    default_strings = [
        '[{ "contentType" : "application/rtf" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
        '[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Tahoma;}{\\f1\\fnil Tahoma;}}\r\n{\\colortbl ;\\red0\\green0\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
        '[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Tahoma;}{\\f1\\fnil Tahoma;}}\r\n{\\colortbl ;\\red0\\green0\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
        '[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl{\\f0\\froman\\fcharset0\\fprq2 Times New Roman;}{\\f1\\fswiss\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
        '[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl{\\f0\\froman\\fcharset0\\fprq2 Times New Roman;}{\\f1\\fswiss\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff1{\\fonttbl\\n{\\f0\\fswiss\\fcharset0\\fprq2 Arial;}\\n{\\f1\\froman\\fcharset0" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9GaWx0ZXIvRmxhdGVEZWNvZGUvQWx0ZXJuYXRlL0RldmljZVJHQi9MZW5ndGggMzc1" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl{\\f0\\fswiss\\fcharset0\\fprq2 Arial;}}{\\colortbl;\\red0\\green" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil Tahoma;}}\\r\\n{\\colortbl ;\\red0\\green0\\blue0;}\\r\\n\\viewkind4\\uc1\\pard" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil Tahoma;}{\\f1\\fnil\\fcharset0 Tahoma;}}\\r\\n{\\colortbl ;\\red0\\green0\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl{\\f0\\fswiss\\fcharset0\\fprq2 Arial;}{\\f1\\fswiss\\fcharset0\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl\\n{\\f0\\fswiss\\fcharset0\\fprq2 Arial;}\\n{\\f1\\froman\\fcharset2" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Tahoma;}{\\f1\\fnil Tahoma;}}\\r\\n{\\colortbl ;\\red0\\green0\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc1\\deff0\\adeff0\\deflang0\\deflangfe0\\adeflang0{\\fonttbl\\n{\\f0\\fswiss\\fcharset" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil Tahoma;}{\\f1\\fnil\\fcharset0 Tahoma;}{\\f2\\fnil MS Sans Serif;}}\\r\\n" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Tahoma;}{\\f1\\fnil Tahoma;}}\\r\\n{\\colortbl ;\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fnil Tahoma;}{\\f1\\fnil\\fcharset0 Tahoma;}}\\r\\n{\\colortbl ;\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl{\\f0\\fswiss\\fcharset0\\fprq2 Arial;}{\\f1\\froman\\fcharset0\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 MS Sans Serif;}{\\f1\\fnil MS Sans Serif;}}\\r\\n{\\colortbl " }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fnil Tahoma;}{\\f1\\fnil\\fcharset0 Tahoma;}{\\f2\\fnil MS San" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\fbidis\\ansi\\deff0{\\fonttbl{\\f0\\fswiss\\fprq2\\fcharset0 Arial;}{\\f1\\fnil Tahoma;}}\\r\\n{\\colortbl " }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : " " }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl{\\f0\\froman\\fcharset0\\fprq2 Times New Roman;}}{\\colortbl;\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1046{\\fonttbl{\\f0\\fnil\\fcharset0 Arial;}{\\f1\\fscript\\fprq2\\fcha" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Times New Roman;}{\\f1\\fnil Tahoma;}}\\r\\n{\\colortbl ;\\red" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Times New Roman;}{\\f1\\fnil Tahoma;}}\\r\\n{\\co" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\froman\\fprq2\\fcharset0 Times New Roman;}{\\f1\\fnil Tahoma;}}\\r\\n{\\colort" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fswiss\\fprq2\\fcharset0 Arial;}{\\f1\\fnil\\fcharset0 Arial;}{\\f2\\fnil\\fc" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1046{\\fonttbl{\\f0\\fnil\\fprq2\\fcharset0 Arial;}{\\f1\\fnil\\fcharse" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Arial;}}\\r\\n{\\colortbl ;\\red0\\green0\\blue0;}\\r\\n\\viewkind4" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Arial;}{\\f1\\fnil\\fcharset0 Tahoma;}{\\f2\\fnil Tahoma;}}" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1046{\\fonttbl{\\f0\\fnil\\fcharset0 Arial;}{\\f1\\fnil Tahoma;}}\\r\\n{\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fswiss\\fprq2\\fcharset0 Arial;}{\\f1\\fnil\\fcharset0 Arial;}{\\f2\\froman\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil\\fcharset0 Times New Roman;}{\\f1\\fnil MS Sans Serif;}{\\f2\\fnil Ta" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1046{\\fonttbl{\\f0\\fnil\\fcharset0 AppleSystemUIFontBold;}{\\f1\\fn" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil Tahoma;}{\\f1\\fnil\\fcharset0 Tahoma;}}\\r\\n{\\colortbl ;\\red0\\green0\\" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]; Tórax -  Procedimento: 6122 - Tc Tórax',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1046{\\fonttbl{\\f0\\fnil Tahoma;}{\\f1\\fnil\\fcharset0 Tahoma;}{\\f2" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1046{\\fonttbl{\\f0\\fnil\\fprq2\\fcharset0 Arial;}{\\f1\\fnil\\fprq2\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fswiss\\fcharset0 Arial;}{\\f1\\fnil\\fcharset0 Arial;}{\\f2\\fnil Tahoma;}" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fswiss\\fcharset0 Arial;}{\\f1\\fswiss\\fprq2\\fcharset0 Arial;}{\\f2\\froma" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fswiss Arial;}{\\f1\\fswiss\\fcharset0 Arial;}{\\f2\\fnil Tahoma;}}\\r\\n{\\col" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1046{\\fonttbl{\\f0\\fnil\\fcharset0 Arial;}{\\f1\\fnil Arial;}{\\f2\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0\\fnil MS Sans Serif;}}\\r\\n{\\colortbl ;\\red0\\green0\\blue128;}\\r\\n\\viewkind4" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\deff0\\deflang1033\\fs24{\\fonttbl{\\f0\\fnil\\fcharset0 Times New Roman;}{\\f1\\fni" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
'[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc1\\deff1\\adeff1\\deflang0\\deflangfe0\\adeflang0{\\fonttbl\\n{\\f0\\fswiss\\fcharset" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
        '[{ "contentType" : "application/rtf", "data" : "{\\rtf1\\ansi\\ansicpg1252\\uc0\\deff0{\\fonttbl{\\f0\\froman\\fcharset0\\fprq2 Times New Roman;}{\\f1\\fswiss\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]',
        '[{ "contentType" : "application/rtf", "data" : "{\\\\rtf1\\\\ansi\\\\ansicpg1252\\\\uc0\\\\deff0{\\\\fonttbl{\\\\f0\\\\froman\\\\fcharset0\\\\fprq2 Times New Roman;}{\\\\f1\\\\fswiss\\\\f" }, { "contentType" : "application/pdf" }, { "contentType" : "application/html" }]'
    ]
    for dft_str in default_strings:
        if(dft_str == row):
            return 'etapa_out_1_string_bin'
        else:
            return row


def apply_parser_clean(row, forward_from: Literal['','clean','strip'] = '') -> str:
    import json
    from bs4 import BeautifulSoup

    if row == 'etapa_out_1_string_bin':
        return row
    else:
        try:
            html_clean = row
            forward_from_history = ['']
            if(forward_from in forward_from_history):
                html = json.loads(row)[2]['url']
                example = ['<','>']
                example_replace = [' <','> ']
                html_clean = html
                for i, x in enumerate(example):
                    html_clean = html_clean.replace(x,example_replace[i])
                soup = BeautifulSoup(html_clean, 'html.parser')
                html_clean = soup.get_text()

                sub_txts = ['5 8 8 17','5.8.8.17']
                for sub in sub_txts:
                    x = html_clean.find(sub)
                    if(x!=-1):
                        html_clean = html_clean[x+len(sub):]
            forward_from_history.append('clean')
            if(forward_from in forward_from_history):
                html_clean = cleanString(html_clean)
            forward_from_history.append('strip')
            if(forward_from in forward_from_history):                
                html_clean = " ".join([
                    bigger_word
                    for bigger_word in html_clean.strip().split()
                    if (len(bigger_word) > 1)
                    if bigger_word.count(bigger_word[0]) != len(bigger_word)
                ])
                return(html_clean)
        except KeyError as e:
            # print(e.with_traceback, e, row)
            return 'apply_parser_clean_KeyError'
        except NameError as e:
            print(e.with_traceback, e, row)
            return 'apply_parser_clean_NameError'
        except Exception as e:
            print(e.with_traceback, e, row)
            return 'apply_parser_clean_Exception'


def do_corpus_embeddings(
    # corpus_cdados: pd.DataFrame, col: str, embedder: SentenceTransformer
    corpus: pd.Series, col: str
) -> np.ndarray:
    """Executa encoding e embedding dos sintagmas nominais

    Parameters
    ----------
        corpus: pd.DataFrame
        col: str
        embedder: SentenceTransformer
    Returns
    ----------
        corpus_embeddings: np.ndarray
    """

    embedder = SentenceTransformer("multi-qa-mpnet-base-cos-v1")
    return embedder.encode(corpus[col].values, convert_to_tensor=True)


def do_semantic_information_retrieval_gpu(
    col: str,
    sentences1: pd.DataFrame,
    sentences2: pd.DataFrame,
    corpus_embeddings: np.ndarray,
    top_k: int,
    embedder: SentenceTransformer,
    mvp: bool = False,
    print_results: bool = True,
) -> None:
    """Consulta informação

    Parameters
    ----------
        sentences1: pd.DataFrame
            Query
        sentences2: pd.DataFrame
            Vai aparecer no resultado, mesmo length do corpus
        corpus_embeddings: np.ndarray
        top_k: int
        embedder: SentenceTransformer
    Returns
    ----------
        None
            Imprime os top 'k' resultados
    """
    if type(sentences1) == str:
        sentences1 = pd.DataFrame(
            {
                'keywords':[sentences1]
            }
        )
    if type(sentences1) == pd.DataFrame:
        sentences1['keywords'] = sentences1['keywords'].apply(lambda x: apply_parser_clean(x, forward_from='clean'))
    else:
        raise TypeError('sentences1 must be a pd.DataFrame')
    for i, _ in enumerate(sentences1):
        query_embedding = embedder.encode(
            sentences1.iloc[i, 0], convert_to_tensor=True
        )
        # usar semantic_search para encontrar os k melhores scores
        top_results = util.semantic_search(
            query_embedding, corpus_embeddings, top_k=top_k
        )[0]

        if print_results:
            print("====================== \n")
            print(
                "Consulta: \n\n",
                sentences1.iloc[i, 0],
            )
            print("\n====================== \n")

            print(
                "Top "
                + str(top_k)
                + " descrições mais semelhantes no corpus:\n"
            )
            print("======================")

        # return top_results
        idx = 0
        df_semantic_return = pd.merge(
            sentences2,
            pd.DataFrame(top_results),
            # how='right',
            # left_on='index',
            left_index=True,
            right_on='corpus_id'
        ).sort_values(by='score', ascending=False).reset_index(drop=True)
        # return df_semantic_return
        for resp_semantic in top_results:
            ## semantic search
            row = df_semantic_return[
                df_semantic_return['corpus_id'] == resp_semantic["corpus_id"]
            ]
            if print_results:
                print(
                    'Texto:\n',
                    str(row[col]),
                    '\n\nCorpus:\n',
                    str(row['features_corpus']),
                    "\n\n=== SEMANTIC SEARCH ===\n",
                    "(Score: %.4f)" % (resp_semantic["score"]),
                )
                print("=======================\n")
        if mvp:
            return df_semantic_return, sentences1['keywords']
        else:
            return df_semantic_return
        # break  #


def do_counter(df: pd.Series) -> Counter:
    df_list = df.to_list()
    tokens = [
        token for text in df_list
              for token in get_tokens_clean(text) 
                  if token and token != np.nan and token != 'nan' and text.strip()
             ]
    counter = Counter(tokens)
    # counter = pd.DataFrame(counter)
    return counter


def show_duplicity(df):
    tot = df.count()
    df_dedup = df.parallel_apply(lambda x: x.drop_duplicates())
    qtd_dedupl = df_dedup.parallel_apply(lambda x: x.count())
    if qtd_dedupl.any() == tot.any():
        print('Sem duplicidade!')
        print('População: %d - Variáveis: %d' % df.shape)
    else:
        dupl = tot - qtd_dedupl
        tot  = tot - dupl
        df.drop_duplicates(inplace=True)
        print('Duplicado: %i' % dupl)
        print('Deduplicado: %i' % tot)
        print('População: %d - Variáveis: %d' % df.shape)
        return


def do_drop_fields_nan(df, df_nan):
    rows_drop = df_nan.index[:]
    df_clean = df.drop(rows_drop).reset_index(drop=True)
    return df_clean.reset_index(drop=True)


def do_identify_nan_by_cols(df):
    df_nan = df.T.parallel_apply(lambda x: x.nunique())
    max_ = df_nan.max()
    df_nan = df[df_nan.values < max_]
    return df_nan

def generate_X_train_X_test(df: pd.DataFrame, col: str):
    return df[df[col].str.len() != 0].reset_index(drop=True), df[df[col].str.len() == 0].reset_index(drop=True)

def tf_idf_iterating_all_words(
    cv_fit_dense: str,
    **kwargs,
):
    from math import log
    feature_names = kwargs['feature_names']
    idf=kwargs['idf']
    df=pd.DataFrame(columns=feature_names)
    for row in range(len(cv_fit_dense)):
        tf_idf_list_for_row=[]
        for word in range(len(feature_names)):
            tf = cv_fit_dense[row,word]
            if(tf == 0):
                corpus = 0
            else:
                corpus = (1 / tf) / (idf[word] / log(10))
            tf_idf_list_for_row.append(corpus)
        df.loc[row] = tf_idf_list_for_row
    return df

def do_bow_inverse_ratio_tfidf(
    dataset: pd.DataFrame,
    feature_column: str,
    ngram_range: tuple,
    max_features: Union[int, None] = None,
    max_df: Union[int, float, None] = 1.0,
    min_df: Union[int, float, None] = 1,
    n_jobs: Union[int, float] = 2.0,
    n_workers: Union[int, float] = -1,
):
    feature_values = dataset[feature_column].values
    from sklearn.feature_extraction.text import (
        CountVectorizer,
        TfidfVectorizer,
    )

    cv = CountVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
    )
    cv_fit = cv.fit_transform(feature_values)
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        max_df=max_df,
        min_df=min_df,
    )
    vectorizer.fit(feature_values)
    idf = vectorizer.idf_
    data = cv_fit.todense()
    n_jobs, n_workers = return_multiprocessing_jobs_and_works(n_jobs, n_workers, len(data))
    if n_jobs == 1:
        tf_idf = tf_idf_iterating_all_words(data, feature_names=cv.get_feature_names(), idf=idf)
    else:
        tf_idf = parallelize_anything(
            func=tf_idf_iterating_all_words,
            data=data,
            n_jobs=n_jobs,
            n_workers=n_workers,
            feature_names=vectorizer.get_feature_names(),
            idf=idf
        ).reset_index(drop=True)
    return pd.concat([dataset, tf_idf],axis=1)

def merge_counters(data: list):
    from collections import Counter
    c = Counter()
    for data_iter in data:
        c.update(data_iter)
    return c


def generate_y_feats(df: pd.DataFrame, col: str):
    a = list()
    for row in df[col]:
        for item in row:
            a.append(item)
    a_unique = list(set(a))
    a_unique.sort()
    labels_dict = {
        label: []
        for label in a_unique
    }
    new_rows_list = list()
    for idx, row in enumerate(df[col]):
        items_to_skip = []
        for item in row:
            copied_row = df.iloc[idx].copy()
            items_to_skip.append(item)
            labels_dict[item].append(1)
            copied_row['label'] = item
            new_rows_list.append(copied_row)
        for item in a_unique:
            if item not in items_to_skip:
                labels_dict[item].append(0)
    df_y = pd.DataFrame(labels_dict)
    new_df = pd.concat(new_rows_list,axis=1).T.reset_index(drop=True)
    print(f'onehot.shape: {df_y.shape}')
    print(f'new_df.shape: {new_df.shape}')
    return df_y, new_df, a

def filter_onehot_by_frequency(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    min_freq: int
):
    df_y_filter = pd.DataFrame()
    for col_i in range(len(df_y.columns)):
        col = df_y.iloc[:,col_i]
        if col.sum() > min_freq:
            df_y_filter = pd.concat([df_y_filter, df_y.iloc[:,col_i]],axis=1)
    df_x_filter = df_x[df_y_filter.sum(axis=1)>0]
    df_y_filter = df_y_filter[df_y_filter.sum(axis=1)>0]
    print(f'df_x_filter.shape: {df_x_filter.shape}')
    print(f'df_y_filter.shape: {df_y_filter.shape}')
    print(f'min_freq: {min_freq}')
    return df_x_filter, df_y_filter


def print_all_important_metrics(model, X_test, y_test, classifier_description: str):
    import yaml
    import json
    from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, jaccard_score, balanced_accuracy_score, roc_auc_score, cohen_kappa_score, mutual_info_score

    prediction = model.predict(X_test)
    # bal_acc = balanced_accuracy_score(y_test, prediction)
    f1score = f1_score(y_test, prediction, average='micro')
    jscore  = jaccard_score(y_test, prediction, average='micro')
    # roc_auc = roc_auc_score(y_test, prediction)
    # kappa   = cohen_kappa_score(y_test, prediction)
    # mut_inf = mutual_info_score(y_test, prediction)
    average_score = (f1score + jscore) / 2
    print(yaml.dump(
        {
            classifier_description:{
                'Accuracy': f'{accuracy_score(y_test, prediction)*100:.2f}%',
                'Precision': f'{precision_score(y_test, prediction, average="micro")*100:.2f}%',
                'Recall': f'{recall_score(y_test, prediction, average="micro")*100:.2f}%',
                'F1': f'{f1score*100:.2f}%',
                'Jaccard': f'{jscore*100:.2f}%',
                'Average F1-Jaccard': f'{average_score*100:.2f}%',
            }
        },
        default_flow_style=False
    ))
    
    return prediction, y_test, model


def rename_combination_y(y: np.ndarray, col_names: list):
    from skmultilearn.model_selection.measures import get_combination_wise_output_matrix
    return ((col_names[combination[0]], col_names[combination[1]]) for row in get_combination_wise_output_matrix(y, order=2) for combination in row)


def print_combinations_train_test(y_train: np.ndarray, y_test: np.ndarray, col_names: list):
    df = pd.DataFrame({
        'train': Counter(rename_combination_y(y_train, col_names=col_names)),
        'test': Counter(rename_combination_y(y_test, col_names=col_names))
        # 'train': Counter(str(combination) for row in get_combination_wise_output_matrix(y_train, order=2) for combination in row),
        # 'test' : Counter(str(combination) for row in get_combination_wise_output_matrix(y_test, order=2) for combination in row)
    }).T.fillna(0.0)
    df = df.reindex(sorted(df.columns), axis=1)
    return df


def run_model_OvR_onehot(
    df_x: pd.DataFrame,
    df_y: pd.DataFrame,
    min_freq: int = 0,
    test_size: float = 0.2,
    stratify: bool = False,
    sparse: bool = False,
    model_function = SGDClassifier(),
    classifier_chain_ascending: bool = False,
    ):
    from sklearn.preprocessing import StandardScaler, PowerTransformer, MinMaxScaler
    from sklearn.svm import LinearSVC
    import xgboost as xgb
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
    from sklearn.pipeline import Pipeline, make_pipeline
    from skmultilearn.model_selection import iterative_train_test_split

    df_x_new, df_y_new = filter_onehot_by_frequency(df_x, df_y, min_freq=min_freq)
    # print(df_y_new.sum().sort_values(ascending=False))

    X=df_x_new.iloc[:,7:]
    X=X.astype(np.float)
    y=df_y_new

    if sparse:
        X = sp.csr_matrix(X)    

    if stratify:
        X_train, y_train, X_test, y_test = iterative_train_test_split(X, y.values, test_size = test_size)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=123)
    
    clf = make_pipeline(
        ### normalization/standardize ###
        # StandardScaler(with_mean=False),
        [StandardScaler(),StandardScaler(with_mean=False)][sparse],
        # MinMaxScaler(),
        # PowerTransformer(method='yeo-johnson', standardize=True),
        
        ## algorithms
        model_function
    )

    ovr_model = OneVsRestClassifier(clf).fit(X_train, y_train)
    print_all_important_metrics(ovr_model, X_test, y_test, f'OneVsRestClassifier{[" - Sem Stratify"," - Stratify"][stratify]}')
    
    multi_model = MultiOutputClassifier(clf).fit(X_train, y_train)
    print_all_important_metrics(multi_model, X_test, y_test, f'MultiOutputClassifier{[" - Sem Stratify"," - Stratify"][stratify]}')
    
    chain_model = ClassifierChain(
        clf,
        order=df_y_new.sum().reset_index(drop=True).sort_values(ascending=classifier_chain_ascending).index
    ).fit(X_train, y_train)
    print_all_important_metrics(chain_model, X_test, y_test, f'Classifier Chain{[" - Sem Stratify"," - Stratify"][stratify]}')
    return y_train, y_test, y.columns, ovr_model, multi_model, chain_model