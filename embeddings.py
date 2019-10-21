import numpy as np
np.random.seed(123)

from multiview import GCCA
import time
import unittest
import os
here = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))


def probas_from_counts(counts):
    if isinstance(counts, str):
        with open(counts) as handle:
            counts = {}
            for line in handle:
                line = line.strip().split()
                if len(line) == 2:
                    counts[line[0]] = int(line[1])

    total = sum(counts.values())
    return {word: counts[word] / total for word in counts}

class Embeddings:
    def __init__(self, verbose = 0, do_cache = "none"):
        self.built = False
        self.verbose = verbose
        self.do_cache = do_cache
        assert self.do_cache in ("none", "shelve", "memory")
        self.set_cache()

    def clear_cache(self):
        if self.do_cache == "shelve":
            try:
                self.cache.close()
            except:
                print("Could not close shelf", self.cache)
            try:
                os.remove(self.cache_location)
            except:
                print("Could not remove shelf location", self.cache_location)
            del self.cache

        elif self.do_cache == "memory":
            del self.cache

    def set_cache(self):
        if self.do_cache == "shelve":
            import shelve, tempfile
            self.cache_location = tempfile.mkstemp(suffix = ".shelve")[1]
            os.remove(self.cache_location)
            print("Creating cache in", self.cache_location)
            self.cache = shelve.open(self.cache_location, writeback = False)

        else:
            self.cache = {}
        
    def reset_cache(self):
        self.clear_cache()
        self.set_cache()

    def get_vector(self, x):
        return self.get_vectors([x]).squeeze(0)

    def build(self):
        raise NotImplementedError
    
    def get_vocab(self):
        raise NotImplementedError
   
    def _get_vectors(self, X):
        raise NotImplementedError

    def _verbose_print(self, *args, level = 1, **kwargs):
        if self.verbose >= level:
            kwargs["flush"] = True
            print(*args, **kwargs)

    def _check_cache(self):
        X = np.random.choice(list(self.cache.keys()), size = (10,))
        A = self._get_vectors(X)
        B = np.array([self.cache[x] for x in X])
        
        if not np.allclose(A,B, atol = 1e-5):
            print("Warning: cache corrupted")
            print(abs(A-B))

    def get_vectors(self, X):
        if not self.built:
            self.build()
  
        self._verbose_print("Requested {} vector(s)".format(len(X)))
        before = time.time()

        if self.do_cache == "none":
            vectors = self._get_vectors(X)

        else:
            vectors = [None for _ in X]
            still_missing = []
            for i, x in enumerate(X):
                if x in self.cache:
                    vectors[i] = self.cache[x]
                else:
                    still_missing.append(i)

            if len(still_missing):
                still_missing_X = [X[i] for i in still_missing]
                missing_vectors = self._get_vectors(still_missing_X)
                for i, vec in enumerate(missing_vectors):
                    vectors[still_missing[i]] = vec
                    self.cache[still_missing_X[i]] = vec

            assert not any([x is None for x in vectors])
            vectors = np.array(vectors)

            #self._check_cache()

        self._verbose_print("This took {} seconds".format(round(time.time() - before, 1)))
        self._verbose_print("Shape: {}".format(vectors.shape))
        return vectors


class WietingEmbeddings(Embeddings):
    def __init__(self, pickle, mode = "concat", **kwargs):
        super(WietingEmbeddings, self).__init__(**kwargs)
        self.pickle = pickle
        self.mode = mode
        assert self.mode in ("concat", "add", "words")

    def build(self):
        if not self.built:
            import _pickle

            tmp = _pickle.load(open(self.pickle, "rb"), encoding = "latin1")
            self.word2vec = {word: tmp[1][tmp[2][1][word]] for word in tmp[2][1]}
            
            if self.mode != "words":
                self.ngram2vec = {ngram: tmp[0][tmp[2][0][ngram]] for ngram in tmp[2][0]}

    def _get_vectors(self, X):
        from paranmt.main.example import example
        vectors = []
        shape = self.word2vec["the"].shape
        for x in X:
            x = example(x)
            x.populate_embeddings(self.word2vec, unk = True)
            wordvector = np.mean(x.embeddings, axis = 0)
            if len(wordvector.shape) == 0:
                wordvector = np.zeros(shape)
            x.unpopulate_embeddings()
            if self.mode == "words":
                vector = wordvector
            else:
                x.populate_embeddings_ngrams(self.ngram2vec, size = 3, unk = True)
                ngramvector = np.mean(x.embeddings, axis = 0)
                if len(ngramvector.shape) == 0:
                    ngramvector = np.zeros(shape)
                x.unpopulate_embeddings()
                if self.mode == "add":
                    vector = wordvector + ngramvector
                elif self.mode == "concat":
                    vector = np.concatenate([wordvector, ngramvector])
            vectors.append(vector)
        return np.array(vectors)

class MCQTEmbeddings(Embeddings):
    def __init__(self, batchsize = 16, device = 7,
            model_config = os.path.join(here, "S2V", "model_configs", "MC-UMBC", "eval.json"), **kwargs):
        super(MCQTEmbeddings, self).__init__(**kwargs)
        self.model_config = model_config
        self.batchsize = batchsize
        self.device = device
    
    def build(self):
        if not self.built:
            import sys
            sys.path.append(os.path.join(here, "S2V", "src"))

            import configuration
            import encoder_manager
            import json
            import tensorflow as tf

            with tf.device("/gpu:{}".format(self.device)):
                tf.app.flags.DEFINE_string("results_path", os.path.join(here, "S2V", "s2v_models"), help = "")
                tf.app.flags.DEFINE_string("Glove_path", os.path.join(here, "S2V", "dictionaries", "GloVe"), help = "")
                tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")

                self.model = encoder_manager.EncoderManager()
                with open(self.model_config) as json_config_file:
                    model_config = json.load(json_config_file)

                if type(model_config) is dict:
                    model_config = [model_config]

                for conf in model_config:
                    for vocab_config in conf['vocab_configs']:
                        if not vocab_config['mode'] == 'fixed':   
                            vocab_config['vocab_file'] = os.path.join(here, "S2V", "s2v_models", vocab_config['vocab_file'])
                            vocab_config['embs_file'] = os.path.join(here, "S2V", "s2v_models", vocab_config['embs_file'])
                    conf['checkpoint_path'] = os.path.join(here, "S2V", "s2v_models", conf['checkpoint_path'])
                    model_config = configuration.model_config(conf, mode = "encode")
                    self.model.load_model(model_config)

            sys.path.pop()

        self.built = True

    def _get_vectors(self, X):
        return self.model.encode(X, batch_size = self.batchsize, verbose = self.verbose)


class RandomProjectionEmbeddings(Embeddings):
    def __init__(self, embeddings, size, seed = 0, **kwargs):
        super(RandomProjectionEmbeddings, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.seed = seed
        self.size = int(size)

    def build(self):
        if not self.built:
            d = self.embeddings.get_vectors(["test"] * 100).shape[-1]
            state = np.random.RandomState(self.seed)
            self.proj = state.uniform(low = -1.0/np.sqrt(d), high = 1.0/np.sqrt(d), size = (d,self.size))
        self.built = True

    def _get_vectors(self, X):
        return self.embeddings.get_vectors(X).dot(self.proj)

    def get_vocab(self):
        return self.embeddings.get_vocab()

class MultiViewEmbeddings(Embeddings):
    def __init__(self, views, vocab = None, **kwargs):
        super(MultiViewEmbeddings, self).__init__(**kwargs)
        self.views = views
        self.vocab = vocab
    
    def build(self):
        if not self.built:
            for view in self.views:
                view.build()
        
        self.built = True
    
    def get_vocab(self):
        if self.vocab is None:
            vocab = set(self.views[0].get_vocab())
            for view in self.views[1:]:
                vocab = set(view.get_vocab()).intersection(vocab)

            return [word for word in self.views[0].get_vocab() if word in vocab]
        
        elif isinstance(self.vocab, str):
            return [line.strip() for line in open(self.vocab) if len(line.strip())]

        return self.vocab


class AvgEmbeddings(MultiViewEmbeddings):
    def _get_vectors(self, X):
        vectors = [view.get_vectors(X) for view in self.views]
        max_dim = max([v.shape[-1] for v in vectors])
        vectors = [np.concatenate([v, \
                np.zeros(v.shape[:-1] + (max_dim - v.shape[-1],))], axis = -1) for v in vectors]
        return sum(vectors, 0) / len(vectors)

class ConcEmbeddings(MultiViewEmbeddings):
    def _get_vectors(self, X):
        vectors = [view.get_vectors(X) for view in self.views]
        return np.concatenate(vectors, axis = 1)


class MappedEmbeddings(MultiViewEmbeddings):
    def __init__(self, views, mapper = GCCA(), vocab = None, **kwargs):
        super(MappedEmbeddings, self).__init__(views = views, vocab = vocab, **kwargs)
        self.mapper = mapper

    def fit(self, vocab):
        views = [view.get_vectors(vocab) for view in self.views]
        before = time.time()
        self._verbose_print("Fitting mapper...", end = " ")
        self.mapper.fit(views)
        self._verbose_print("... took {} seconds".format(round(time.time() - before,1)))

    def build(self):
        if not self.built:
            for view in self.views:
                view.build()

            vocab = self.get_vocab()
            if not vocab is None:
                self.fit(vocab)

        self.built = True
    
    def _get_vectors(self, X):
        vectors = [view.get_vectors(X) for view in self.views]
        return self.mapper.transform(vectors)
    

class GensimEmbeddings(Embeddings):
    def __init__(self, path, lowercase = False, allow_unknown = True, **kwargs):
        super(GensimEmbeddings, self).__init__(**kwargs)
        self.path = path
        self.lowercase = lowercase
        self.allow_unknown = allow_unknown

    def _get_vectors(self, X):
        vectors = []
        if self.lowercase:
            X = [x.lower() for x in X]

        for x in X:
            try:
                vectors.append(self.model.wv[x])
            except KeyError as e:
                if self.allow_unknown:
                    vectors.append(np.zeros((self.model.vector_size,)))
                else:
                    raise e

        return np.array(vectors) 
    
    def get_vocab(self):
        return list(self.model.wv.vocab.keys())

class GensimFastTextEmbeddings(GensimEmbeddings):
    def build(self):
        if not self.built:
            from gensim.models import FastText
            if self.path.endswith("gensim"):
                self.model = FastText.load(self.path)
            else:
                self.model = FastText.load_fasttext_format(self.path)
        self.built = True

class Sent2VecEmbeddings(Embeddings):
    def __init__(self, modelpath, **kwargs):
        super(Sent2VecEmbeddings, self).__init__(**kwargs)
        self.modelpath = modelpath

    def build(self):
        if not self.built:
            import sent2vec
            self.model = sent2vec.Sent2vecModel()
            self.model.load_model(self.modelpath)
        self.built = True

    def _get_vectors(self, X):
        return self.model.embed_sentences(X)

class GensimKeyedVectorsEmbeddings(GensimEmbeddings):
    def build(self):
        if not self.built:
            from gensim.models import KeyedVectors
            if self.path.endswith("gensim"):
                self.model = KeyedVectors.load(self.path)
            else:
                self.model = KeyedVectors.load_word2vec_format(self.path)

        self.built = True


class SIFWeightedEmbeddings(Embeddings):
    def __init__(self, embeddings, counts, a = .001, **kwargs):
        super(SIFWeightedEmbeddings, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.a = a
        self.counts = counts
    
    def build(self):
        if not self.built:
            self.embeddings.build()
            probas = probas_from_counts(self.counts)
            self.weights = {word: self.a / (self.a + probas[word]) for word in probas}

        self.built = True
    
    def _get_vectors(self, X):
        vectors = self.embeddings.get_vectors(X)
        weights = np.array([self.weights.get(x, 1) for x in X])
        weighted = vectors * np.expand_dims(weights, -1)
        return weighted

    def get_vocab(self):
        return self.embeddings.get_vocab()

class TopPCRemovedEmbeddings(Embeddings):
    def __init__(self, embeddings, vocab = None, K = 3, N = 100000, **kwargs):
        super(TopPCRemovedEmbeddings, self).__init__(**kwargs)
        self.embeddings = embeddings
        self.K = K
        self.N = N
        self.vocab = vocab

    def build(self):
        if not self.built:
            self.embeddings.build()
            
            vocab = self.get_vocab()[:self.N]
            vectors = self.embeddings.get_vectors(vocab)
            cov = np.cov(vectors.T)
            
            u, sigma, ut = np.linalg.svd(cov)
            selection = [0] * self.K + [1] * (sigma.shape[0] - self.K)
            self.G = u.dot(np.diag(selection)).dot(ut)

        self.built = True

    def _get_vectors(self, X):
        return self.embeddings.get_vectors(X).dot(self.G)

    def get_vocab(self):
        if not self.vocab is None:
            return self.vocab
        return self.embeddings.get_vocab()

    

class AveragedEmbeddings(Embeddings):
    def __init__(self, embeddings, **kwargs):
        super(AveragedEmbeddings, self).__init__(**kwargs)
        self.embeddings = embeddings

    def build(self):
        if not self.built:
            self.embeddings.build()
        self.built = True

    def _get_vectors(self, X):
        if not self.batchsize:
            self.batchsize = len(X)

    def _get_vectors(self, X):
        X = [x.split() for x in X]
        vectors = []
        unique_words = set()
        
        for x in X:
            unique_words.update(x)
        
        unique_words = list(unique_words)
        unique_vectors = self.embeddings.get_vectors(unique_words)
        word2vec = {word: vec for word, vec in zip(unique_words, unique_vectors)}

        for i, x in enumerate(X):
            if len(x) == 0:
                vectors.append(np.zeros_like(unique_vectors[0]))
            else:
                sentence = np.array([word2vec[word] for word in x])
                vectors.append(sentence.mean(axis = 0))

        return np.array(vectors)

class SIFPCAveragedEmbeddings(Embeddings):
    def __init__(self, embeddings, counts = {}, K = 3, a = .001, N = 100000, **kwargs):
        super(SIFPCAveragedEmbeddings, self).__init__(**kwargs)
        sifweighted = SIFWeightedEmbeddings(embeddings, counts = counts, a = a, **kwargs)
        pcremoved = TopPCRemovedEmbeddings(sifweighted, K = K, N = N, **kwargs)
        self.embeddings = AveragedEmbeddings(pcremoved)

    def build(self):
        if not self.built:
            self.embeddings.build()
        self.built = True

    def _get_vectors(self, X):
        return self.embeddings.get_vectors(X)

class Doc2VecEmbeddings(Embeddings):
    def __init__(self, modelpath, **kwargs):
        super(Doc2VecEmbeddings, self).__init__(**kwargs)
        self.modelpath = modelpath
    
    def build(self):
        if not self.built:
            from gensim.models import Doc2Vec
            self.model = Doc2Vec.load(self.modelpath)
        self.built = True

    def _get_vectors(self, X):
        return np.array([self.model.infer_vector(x.split()) for x in X])


class SBERTEmbeddings(Embeddings):
    def __init__(self, name="bert-large-nli-mean-tokens", device=0, batchsize=32, **kwargs):
        super(SBERTEmbeddings, self).__init__(**kwargs)
        self.name = name
        self.device = device
        self.batchsize = batchsize

    def build(self):
        from sentence_transformers import SentenceTransformer
        if not self.built:
            self.model = SentenceTransformer(self.name, device = "cuda:{}".format(self.device))
            self.model.eval()
        self.built = True

    def _get_vectors(self, X):
        vectors = self.model.encode(X, batch_size = self.batchsize, show_progress_bar = bool(self.verbose))
        return np.array(vectors)

class USIFEmbeddings(Embeddings):
    def __init__(self, path, counts, **kwargs):
        super(USIFEmbeddings, self).__init__(**kwargs)
        self.path = path
        self.counts = counts

    def build(self):
        from usif.usif import word2vec, uSIF, word2prob
        if not self.built:
            self.model = uSIF(word2vec(self.path), word2prob(self.counts))
        self.built = True

    def _get_vectors(self, X):
        return np.array(self.model.embed(X))


class InferSentEmbeddings(Embeddings):
    def __init__(self, batchsize = 64, device = 0,
            embpath = os.path.join("InferSent", "dataset", "fastText", "crawl-300d-2M.vec"), 
            modelpath = os.path.join("InferSent", "encoder", "infersent2.pickle"),
            #embpath = os.path.join("InferSent", "dataset", "GloVe", "glove.840B.300d.txt"), 
            #modelpath = os.path.join("InferSent", "encoder", "infersent1.pickle"),
            **kwargs):

        import re
        super(InferSentEmbeddings, self).__init__(**kwargs)
        self.batchsize = batchsize
        self.embpath = embpath
        self.modelpath = modelpath
        self.device = device
        self.version = int(re.findall("infersent([12]).pickle", modelpath)[-1])
        self.emb_dim = int(re.findall("([0-9]+)d", self.embpath)[-1])

    def build(self):
        if not self.built:
            from InferSent.models import InferSent
            import torch

            params_model = {"bsize": self.batchsize, "word_emb_dim": self.emb_dim, 
                    "enc_lstm_dim": 2048, "pool_type": "max", 
                    "dpout_model": 0.0, "version": self.version, "device": self.device}

            self.model = InferSent(params_model)
            self.model.load_state_dict(torch.load(self.modelpath))
            self.model.set_w2v_path(self.embpath)
            self.model.eval()

            if torch.cuda.is_available():
                self.model = self.model.cuda(self.device)

        self.built = True

    def _get_vectors(self, X):
        self.model.build_vocab(X, tokenize = False)
        vectors = []
        for start in range(0, len(X), self.batchsize):
            batch = X[start:min(start+self.batchsize, len(X))]
            vectors.append(self.model.encode(batch, tokenize = False, verbose = False, bsize = self.batchsize))
            self._verbose_print(start + len(batch), "/", len(X), 
                    "({},{})".format(len(batch), max([len(x) for x in batch])), end = "     \r")
        return np.concatenate(vectors, axis = 0)


class USEEmbeddings(Embeddings):
    def __init__(self, batchsize = 128, name = "USE", device = 0,
            url = "https://tfhub.dev/google/universal-sentence-encoder-large/3", **kwargs):
        super(USEEmbeddings, self).__init__(**kwargs)
        self.url = url
        self.batchsize = batchsize
        self.name = name
        self.device = device

    def build(self):
        if not self.built:
            import tensorflow as tf
            import tensorflow_hub as tfhub

            with tf.device("/gpu:{}".format(self.device)):
                self.module = tfhub.Module(self.url, name = self.name, trainable = False)
                self.input = tf.placeholder(dtype = tf.string)
                self.embedded = self.module(self.input)
                config = tf.ConfigProto(allow_soft_placement = True)
                config.gpu_options.allow_growth = True
            
                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())
                self.session.run(tf.tables_initializer())

        self.built = True

    def _get_vectors(self, X):
        vectors = []
        for start in range(0, len(X), self.batchsize):
            batch = np.array(X[start:min(start+self.batchsize, len(X))], dtype = object) 
            vectors.append(self.session.run(self.embedded, feed_dict = {self.input: batch}))
            
            self._verbose_print(start + len(batch), "/", len(X), 
                    "({},{})".format(len(batch), max([len(x) for x in batch])), end = "     \r")
        
        self._verbose_print("")
        return np.concatenate(vectors, axis = 0)



class SIFWeightedContextEmbeddings(Embeddings):
    def __init__(self, model, counts, a = .001, **kwargs):
        super(SIFWeightedContextEmbeddings, self).__init__(**kwargs)
        self.counts = counts
        self.model = model
        self.a = a

    def _get_vectors(self, X):
        vector_list = self.model._get_vector_list(X)
        assert len(X) == len(vector_list)
        
        X_tokenized = [self.model._tokenize(x) for x in X]
        vectors = []


        for x, vec in zip(X_tokenized, vector_list):
            assert len(x) == vec.shape[0]
            if len(x) == 0:
                vectors.append(np.zeros((vector_list[-1].shape[-1],)))
            else:
                weights = np.array([self.weights.get(word, 1) for word in x])
                for _ in range(len(vec.shape) - 1):
                    weights = np.expand_dims(weights, -1)
                vectors.append((vec * weights).mean(axis = 0))

        return np.array(vectors)
         

    def build(self):
        if not self.built:
            self.model.build()

            probas = probas_from_counts(self.counts)
            
            new_probas = {}
            for word in probas:
                for token in self.model._tokenize(word):
                    new_probas[token] = new_probas.get(token, 0) + probas[word]
            total = sum(new_probas.values())
            
            new_probas = {token: new_probas[token] / total for token in new_probas}
            self.weights = {token: self.a / (self.a + new_probas[token]) for token in new_probas}
        
        self.built = True

class SubEmbeddings(Embeddings):
    def __init__(self, model, d, **kwargs):
        super(SubEmbeddings, self).__init__(**kwargs)
        self.model = model
        self.d = d

    def build(self):
        if not self.built:
            self.model.build()
        self.built = True

    def get_vectors(self, X):
        vectors = self.model.get_vectors(X)
        if isinstance(self.d, int):
            return vectors[:,:self.d]
        return vectors[:,self.d]

class ListEmbeddings(Embeddings):
    def __init__(self, **kwargs):
        super(ListEmbeddings, self).__init__(**kwargs)
    
    def get_vector_list(self, X):
        if not self.built:
            self.build()
        return self._get_vector_list(X)

    def _get_vector_list(self, X):
        raise NotImplementedError

    def _tokenize(self, x):
        raise NotImplementedError

class ELMo(ListEmbeddings):
    def __init__(self, layers = "avg", name = "ELMo", device = 0,
            url = "https://tfhub.dev/google/elmo/2", batchsize = 16, **kwargs):
        super(ELMo, self).__init__(**kwargs)
        self.url = url
        self.batchsize = batchsize
        self.name = name
        self.device = device

    def build(self):
        if not self.built:
            import tensorflow_hub as tfhub
            import tensorflow as tf

            with tf.device("/gpu:{}".format(self.device)):
            
                tags = set()
                self.module = tfhub.Module(self.url, tags = tags, name = self.name, trainable = False)
                config = tf.ConfigProto(allow_soft_placement = True)
                config.gpu_options.allow_growth = True
            
                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())
            
                self.input_strings = tf.placeholder(dtype = tf.string, shape = (None,))

                dictionary = self.module(self.input_strings, signature = "default", as_dict = True)
                layer_outputs = [tf.concat([dictionary["word_emb"], dictionary["word_emb"]], axis = -1)]
                layer_outputs.append(dictionary["lstm_outputs1"])
                layer_outputs.append(dictionary["lstm_outputs2"])
            
                self.outputs = {"stack": tf.stack(layer_outputs, -1), "conc": tf.concat(layer_outputs, -1)}
                self.outputs["avg"] = tf.reduce_mean(self.outputs["stack"], axis = -1)
            
                for i in range(len(layer_outputs)):
                    self.outputs[i] = layer_outputs[i]

        self.built = True

    def _tokenize(self, x):
        return x.split(" ")

    def _get_vector_list(self, X, which = "stack"):
        X = [self._tokenize(x) for x in X]
        
        order = np.argsort([len(x) for x in X])
        re_order = np.argsort(order)
        X = [X[i] for i in order]
        
        vectors = []

        for start in range(0, len(X), self.batchsize):
            batch = X[start:min(len(X), start + self.batchsize)]
            feed_dict = {self.input_strings: np.array([" ".join(x) for x in batch], dtype = object)}
            
            if which == "stack":
                output = [self.session.run(self.outputs[i], feed_dict = feed_dict) for i in range(len(self.outputs)) if i in self.outputs]
                output = np.stack(output, axis = -1)
            else:
                output = self.session.run(self.outputs[which], feed_dict = feed_dict)

            for x, vec in zip(batch, output):
                vectors.append(vec[:len(x)])
            
            self._verbose_print(start + len(batch), "/", len(X), 
                    "({},{})".format(len(batch), max([len(x) for x in batch])), end = "     \r")
        
        self._verbose_print("")
        vectors = [vectors[i] for i in re_order]
        return vectors


class MultiViewLayerEmbeddings(ListEmbeddings):
    def __init__(self, model, **kwargs):
        super(MultiViewLayerEmbeddings, self).__init__(**kwargs)
        self.model = model
    
    def build(self):
        if not self.built:
            self.model.build()
        self.built = True

    def _get_vector_list(self, X):
        vectorlist = self.model._get_vector_list(X, which = self.key)
        return vectorlist

    def _tokenize(self, x):
        return self.model._tokenize(x)


class ConcLayerEmbeddings(MultiViewLayerEmbeddings):
    def __init__(self, **kwargs):
        super(ConcLayerEmbeddings, self).__init__(**kwargs)
        self.key = "conc"

class AvgLayerEmbeddings(MultiViewLayerEmbeddings):
    def __init__(self, **kwargs):
        super(AvgLayerEmbeddings, self).__init__(**kwargs)
        self.key = "avg"

class SelectLayerEmbeddings(MultiViewLayerEmbeddings):
    def __init__(self, layer = -1, **kwargs):
        super(SelectLayerEmbeddings, self).__init__(**kwargs)
        self.layer = layer
        
    def _get_vector_list(self, X):
        return self.model._get_vector_list(X, which = self.layer)
        
class MappedLayersEmbeddings(MultiViewLayerEmbeddings):
    def __init__(self, vocab = None, mapper = GCCA(), maxnum = 1000, maxlen = 50, **kwargs):
        super(MappedLayersEmbeddings, self).__init__(**kwargs)
        self.vocab = vocab
        self.mapper = mapper
        self.maxlen = maxlen
        
        if maxnum:
            self.vocab = self.vocab[:maxnum]


    def build(self):
        if not self.built:
            self.model.build()
            vocab = self.vocab
            if not vocab is None:
                if isinstance(vocab, str):
                    vocab = [line.strip() for line in open(self.vocab) if len(line.strip())]
                self.fit(vocab)

        self.built = True

    def fit(self, vocab):
        vectorlist = self.model._get_vector_list(vocab, which = "stack")
        if self.maxlen:
            vectorlist = [vec[:self.maxlen] for vec in vectorlist]

        vectors = np.concatenate(vectorlist, axis = 0)
        assert len(vectors.shape) == 3
        vectors = list(np.transpose(vectors, [2,0,1]))
        before = time.time()
        self._verbose_print("Fitting mapper...", end = " ")
        self.mapper.fit(vectors)
        self._verbose_print("... took {} seconds".format(round(time.time() - before,1)))

    def _get_vector_list(self, X):
        vectorlist = self.model._get_vector_list(X)
        
        lengths = [vec.shape[0] for vec in vectorlist]
        vectors = np.concatenate(vectorlist, axis = 0) # (length*batch, dim, layers)
        vectors = list(np.transpose(vectors, [2,0,1])) # (layers, length*batch, dim)
        mapped = self.mapper.transform(vectors) # (length*batch, dim)
        output = []
        slc = slice(0,0)
        for l in lengths:
            slc = slice(slc.stop, slc.stop + l)
            output.append(mapped[slc])
        return output

class BERT(ListEmbeddings):
    def __init__(self, name = "BERT", batchsize = 128, device = 0, checkpoint = None,
            url = "https://tfhub.dev/google/bert_uncased_L-24_H-1024_A-16/1", **kwargs):

        super(BERT, self).__init__(**kwargs)
        self.url = url
        self.batchsize = batchsize
        self.name = name
        self.device = device
        self.checkpoint = checkpoint

    def build(self):
        if not self.built:
            import tensorflow_hub as tfhub
            import tensorflow as tf
            
            from bert.tokenization import FullTokenizer
            from bert.modeling import reshape_from_matrix, get_shape_list
            import re
            output_rgx = re.compile("{}/bert/encoder/layer_([0-9]+)/output/LayerNorm/batchnorm/add_1".format(self.name))

            with tf.device("/gpu:{}".format(self.device)):
                tags = set()
                self.module = tfhub.Module(self.url, tags = tags, name = self.name, trainable = False)
                
                if self.checkpoint:
                    variables = tf.global_variables(scope = self.name)
                    init_vars = tf.train.list_variables(self.checkpoint)
                    init_vars_names = set([name for name, var in init_vars])
            
                    assignment_map = {}
                    for var in variables:
                        basename = re.findall("^{}/(.*):\\d+$".format(self.name), var.name)[0]
                        assert basename in init_vars_names
                        assignment_map[basename] = var

                    print("Initializing", len(assignment_map), "BERT weights")
                    tf.train.init_from_checkpoint(self.checkpoint, assignment_map)
           
                tensors = {key: self.module._graph._nodes_by_name[key] for key in self.module._graph._nodes_by_name}

                input_name = "{}/bert/embeddings/LayerNorm/batchnorm/add_1".format(self.name)
                input_shape = get_shape_list(tensors[input_name]._outputs[0], expected_rank = 3)
            

                layer_output_names = [name for name in tensors if output_rgx.match(name)]
                layer_output_names.sort(key = lambda x:int(output_rgx.findall(x)[0]))
                layer_outputs = [reshape_from_matrix(tensors[name]._outputs[0], input_shape) for name in layer_output_names]
                config = tf.ConfigProto(allow_soft_placement = True)
                config.gpu_options.allow_growth = True
        
        
            
                self.session = tf.Session(config=config)
                self.session.run(tf.global_variables_initializer())
            
                tokenization_info = self.session.run(self.module(signature="tokenization_info", as_dict=True))
                self.tokenizer = FullTokenizer(**tokenization_info)

                self.outputs = {"stack": tf.stack(layer_outputs, axis = -1), "conc": tf.concat(layer_outputs, axis = -1)}
                self.outputs["avg"] = tf.reduce_mean(self.outputs["stack"], axis = -1)
            
                for i in range(len(layer_outputs)):
                    self.outputs[i] = layer_outputs[i]

        self.built = True

    def _tokenize(self, x):
        return self.tokenizer.tokenize(x)[:510]

    def _get_vector_list(self, X, which = "stack"):

        X = [["[CLS]"] + self._tokenize(x) + ["[SEP]"] for x in X]
        X = [self.tokenizer.convert_tokens_to_ids(x) for x in X]

        order = np.argsort([len(x) for x in X])
        re_order = np.argsort(order)
        X = [X[i] for i in order]

        vectors = []
        for start in range(0, len(X), self.batchsize):
            batch = X[start:min(len(X), start + self.batchsize)]
            maxlen = max([len(x) for x in batch])
            input_ids = np.array([x + [0] * (maxlen - len(x)) for x in batch])
            
            self._verbose_print(start + len(batch), "/", len(X), 
                    "({},{})".format(len(batch), max([len(x) for x in batch])), end = "     \r")

            feed_dict = {"{}/input_ids:0".format(self.name): input_ids, 
                    "{}/input_mask:0".format(self.name): input_ids != 0,
                    "{}/segment_ids:0".format(self.name): input_ids * 0}
            
            if which == "stack":
                output = [self.session.run(self.outputs[i], feed_dict = feed_dict) for i in range(len(self.outputs)) if i in self.outputs]
                output = np.stack(output, axis = -1)
            else:
                output = self.session.run(self.outputs[which], feed_dict = feed_dict)

            for x, vec in zip(batch, output):
                vectors.append(vec[1:len(x)-1])
            
        
        self._verbose_print("")
        vectors = [vectors[i] for i in re_order]
        return vectors


class EmbeddingsTests(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        from nltk.corpus import brown
        from nltk import FreqDist
        import tempfile

        cls.sentences = [" ".join(sent).lower() for sent in brown.sents()]
        cls.embeddingfile = tempfile.mkstemp()[1]
        cls.countfile = tempfile.mkstemp()[1]
    
        counts = FreqDist(w.lower() for w in brown.words())
        words = sorted(list(counts.keys()))[:-10]

        state = np.random.RandomState(0)
        with open(cls.embeddingfile, "w") as embeddinghandle:
            with open(cls.countfile, "w") as counthandle:
                embeddinghandle.write("{} {}\n".format(len(words), 50))
                for word in words:
                    counthandle.write("{} {}\n".format(word, counts[word]))
                    embeddinghandle.write("{} {}\n".format(word, " ".join(map(str, state.uniform(size = (50,))))))
        
        cls.use_embeddings = USEEmbeddings()
        cls.word_embeddings = GensimKeyedVectorsEmbeddings(cls.embeddingfile)
        cls.sif_embeddings = SIFPCAveragedEmbeddings(cls.word_embeddings, counts = cls.countfile)
        cls.views = [cls.use_embeddings, cls.sif_embeddings]

    @classmethod
    def tearDownClass(cls):
        import os
        os.remove(cls.countfile)
        os.remove(cls.embeddingfile)

    def test_use(self):
        vectors = self.use_embeddings.get_vectors(self.sentences[:10])
        self.assertEqual(vectors.shape, (10, 512))
    
    def test_sif(self):
        vectors = self.sif_embeddings.get_vectors(self.sentences[:10])
        self.assertEqual(vectors.shape, (10, 50))

    def test_conc(self):
        conc_embeddings = ConcEmbeddings(self.views)
        vectors = conc_embeddings.get_vectors(self.sentences[:10])
        self.assertEqual(vectors.shape, (10, 1024 + 50 + 512))
    
    def test_map(self):
        map_embeddings = MappedEmbeddings(self.views, vocab = self.sentences[:1000])
        vectors = map_embeddings.get_vectors(self.sentences[:10])
        self.assertEqual(vectors.shape[0], 10)


if __name__ == "__main__":
    unittest.main()
