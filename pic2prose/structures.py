import easyocr
from collections import Counter, defaultdict
from math import log
import numpy as np
import re

class Corp:
    
    def __init__(self, image_path:str=None, corpus:list[str]=None) -> None:
        if image_path is None and corpus is None:
            raise Exception('Must provide either an image or text corpus')
        elif image_path is not None and corpus is not None:
            raise Exception('Must provide either an image or text corpus, not both')
        elif image_path is not None and corpus is None:
            self.corpus = self._image_to_text(image_path) 
        else:
            self.corpus = corpus
            
        self.sentences = self._sent_tokenize(self.corpus)
        self.tokens = self._word_tokenize(self.corpus.lower())
        self.characters = set(list(self.corpus))
        self.token_freq = None
        self.co_occurrence_matrix = None
        self.tfidf_matrix = None
        self.bigrams = None
        self.trigrams = None
    
    def _image_to_text(self, image_path:str, contrast_ths:float=1.0, detail:float=1, paragraph:bool=False) -> str:
        reader = easyocr.Reader(['en'])
        img_out  = reader.readtext(image_path, contrast_ths=contrast_ths, detail=detail, paragraph=paragraph)
        corpus = str(" ".join([i[1] for i in img_out]))
        
        del img_out
        
        return corpus
    
    def _word_tokenize(self, text:str) -> list[str]:
        return re.findall(r'\b\w+\b', text)
    
    def _sent_tokenize(self, text:str) -> list[str]:
        return re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        
    def get_token_frequency(self) -> Counter:
        if self.token_freq is None:
            self.token_freq = Counter(self.tokens)
            
        return self.token_freq
    
    def get_co_occurrence_matrix(self) -> np.ndarray:
        if self.co_occurrence_matrix is None:
            vocab = set(self.tokens)
            vocab = list(vocab)
            vocab_index = {word: i for i, word in enumerate(vocab)}
            self.co_occurrence_matrix = np.zeros((len(vocab), len(vocab)))
            
            for sentence in self.sentences:
                sent_tokens = self._word_tokenize(sentence.lower())
                for i, token in enumerate(sent_tokens):
                    for j in range(max(i-1, 0), min(i+2, len(sent_tokens))):
                        if i != j:
                            self.co_occurrence_matrix[vocab_index[token]][vocab_index[sent_tokens[j]]] += 1
                            
        return self.co_occurrence_matrix
    
    def get_tfidf_matrix(self) -> np.ndarray:
        if self.tfidf_matrix is None:
            
            term_freqs = []
            vocab = set(self.tokens)
            vocab = list(vocab)
            vocab_index = {word: i for i, word in enumerate(vocab)}
            
            for sentence in self.sentences:
                sent_tokens = self._word_tokenize(sentence.lower())
                term_freq = [0] * len(vocab)
                
                for token in sent_tokens:
                    if token in vocab_index:
                        term_freq[vocab_index[token]] += 1
                
                term_freqs.append(term_freq)
            
            
            term_freqs = np.array(term_freqs)
            
            
            doc_count = len(self.sentences)
            idf = [log(doc_count / (1 + np.count_nonzero(term_freqs[:, i]))) for i in range(len(vocab))]
            
            
            self.tfidf_matrix = term_freqs * idf
            
        return self.tfidf_matrix
    
    def get_bigrams(self) -> list[tuple[str, str]]:
        if self.bigrams is None:
            self.bigrams = [(self.tokens[i], self.tokens[i + 1]) for i in range(len(self.tokens) - 1)]
            
        return self.bigrams
    
    def get_trigrams(self) -> list[tuple[str, str, str]]:
        if self.trigrams is None:
            self.trigrams = [(self.tokens[i], self.tokens[i + 1], self.tokens[i + 2]) for i in range(len(self.tokens) - 2)]
            
        return self.trigrams
    
    
    def char2idx(self) -> dict[str, int]:
        char_set = set(self.characters)
        
        return {char: idx for idx, char in enumerate(sorted(char_set))}
    
    def idx2char(self) -> dict[int, str]:
        char_set = set(self.characters)
        
        return {idx: char for idx, char in enumerate(sorted(char_set))}
    
    
    def one_hot_encode(self) -> np.ndarray:
        char_idx_map = self.char2idx()
        one_hot_matrix = np.zeros((len(self.characters), len(char_idx_map)))
        
        for i, char in enumerate(self.characters):
            one_hot_matrix[i][char_idx_map[char]] = 1
        
        return one_hot_matrix
    
    
    def build_ngrams(self, n) -> list[tuple[str, ...]]:
        ngrams = []
        
        for i in range(len(self.tokens) - n + 1):
            ngrams.append(tuple(self.tokens[i:i+n]))
            
        return ngrams
    
    
    def lemmatize(self) -> list[str]:
        lemmatized_tokens = []
        for token in self.tokens:
            if token.endswith('ing'):
                lemmatized_tokens.append(token[:-3])
            elif token.endswith('ed'):
                lemmatized_tokens.append(token[:-2])
            elif token.endswith('ly'):
                lemmatized_tokens.append(token[:-2])
            else:
                lemmatized_tokens.append(token)
                
        return lemmatized_tokens