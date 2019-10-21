import numpy as np
import scipy.linalg as linalg

class SKLearnWrapper:
    def __init__(self, algo = "pls"):
        self.algo = algo

    def fit(self, views):
        from sklearn.cross_decomposition import PLSCanonical
        if self.algo == "pls":
            self.mapper = PLSCanonical(min(views[0].shape[-1], views[1].shape[-1]))
        self.mapper.fit(*views)

    def transform(self, views):
        return sum(self.mapper.transform(*views), 0)

def batch_cov(X, batchsize = 1000): # (dims, N)
    X -= X.mean(axis = 1, keepdims = True)
    cov = np.zeros((X.shape[0], X.shape[0]))
    for i in range(0, X.shape[0], batchsize):
        for j in range(i, X.shape[0], batchsize):
            xi = X[i:i+batchsize]
            xj = X[j:j+batchsize]
            c = xi.dot(xj.T) / (X.shape[1] - 1)
            cov[i:i+batchsize, j:j+batchsize] = c
            cov[j:j+batchsize, i:i+batchsize] = c.T

    return cov

class ConcSVD:
    def __init__(self, d = "sum", seed = 0):
        self.d = d
        self.seed = seed

    def fit(self, views):
        from sklearn.decomposition import PCA as TruncatedSVD
        concat = np.concatenate(views, axis = 1)
        n_components = self.d
        if self.d == "sum":
            n_components = concat.shape[1]-1
        elif self.d == "min":
            n_components = min([view.shape[1] for view in views])
        elif self.d == "max":
            n_components = max([view.shape[1] for view in views])

        n_components = int(n_components)
        self.svd = TruncatedSVD(n_components = n_components, random_state = self.seed)
        self.svd.fit(concat)

    def transform(self, views):
        concat = np.concatenate(views, axis = 1)
        return self.svd.transform(concat)
    
    def save(self, path):
        import _pickle
        obj = {"svd": self.svd}
        with open(path, "wb") as handle:
            _pickle.dump(obj, handle)

    def load(self, path):
        import _pickle
        with open(path, "rb") as handle:
            obj = _pickle.load(handle)
        self.svd = obj["svd"]


class GCCA:
    def __init__(self, tau = 0.1, algo = "eig", d = "sum"):
        self.tau = tau
        self.algo = algo
        self.d = d
    
    def save(self, path):
        import _pickle
        obj = {"mean": self.mean, "theta": self.theta}
        with open(path, "wb") as handle:
            _pickle.dump(obj, handle)

    def load(self, path):
        import _pickle
        with open(path, "rb") as handle:
            obj = _pickle.load(handle)
        self.mean, self.theta = obj["mean"], obj["theta"]

    def fit(self, views):
        dims = [view.shape[1] for view in views]
        concat = np.concatenate(views, axis = 1)
        self.mean = concat.mean(axis = 0)
        cov = np.cov(concat.T)
        
        algo = self.algo

        if concat.shape[-1] > 2000:
            algo = "eigh"


        slc = slice(0,0)
        for dim in dims:
            slc = slice(slc.stop, slc.stop + dim)
            cov[slc, slc] += self.tau * np.diag(cov)[slc].mean() * np.eye(dim)
        

        mask = linalg.block_diag(*[np.ones((dim, dim), bool) for dim in dims])

        if algo == "eig":
            eigvals, eigvecs = linalg.eig(cov * np.invert(mask), cov * mask)
            tol = 0
        elif algo == "eigh":
            eigvals, eigvecs = linalg.eigh(cov * np.invert(mask), cov * mask)
            tol = abs(eigvals.real).max() * len(eigvals) * np.finfo(float).eps

        eigvals, eigvecs = eigvals.real, eigvecs.real
        sorting = np.argsort(eigvals)[::-1]
        eigvals, eigvecs = eigvals[sorting], eigvecs[:,sorting]
        
        #self.theta = eigvecs[:, np.greater(abs(eigvals), tol)]

        d = self.d
        if d == "sum":
            d = concat.shape[-1]
        elif d == "max":
            d = max([view.shape[-1] for view in views])
        elif d == "min":
            d = min([view.shape[-1] for view in views])
        
        d = int(d)
        self.theta = eigvecs[:,:d]

    def transform(self, views, d = None):
        concat = np.concatenate(views, axis = 1)
        return (concat - self.mean).dot(self.theta)


class ReconstructionMV:
    def __init__(self, norm = False, seed = 0, repsize = "sum", loss = "mae", verbose = 1,
            mode = "coupled", device = 6, epochs = 500, activation = "relu", nhid = 0):
        self.repsize = repsize
        self.loss = loss
        self.device = device
        self.seed = seed
        self.epochs = epochs
        self.mode = mode
        self.norm = norm
        self.nhid = nhid
        self.activation = activation
        self.verbose = verbose

        assert self.mode in ("AAEME", "coupled")
        assert self.loss in ("cos", "mse", "si_mse", "kld", "mae", "si_mae")

    def save(self, path):
        self.apply_model.save(path)
        
    def load(self, path):
        from keras.models import load_model
        self.apply_model = load_model(path)

    def build(self, dims):
        from keras.layers import Input, Dense, Lambda, add, concatenate
        from keras.models import Model, Sequential
        from keras.losses import kld, cosine_proximity
        from keras.initializers import glorot_uniform
        import keras.backend as K
        
        repsize = self.repsize
        if repsize in ("sum", "max", "min"):
            repsize = eval(repsize)(dims)

        repsize = int(repsize)
        
        def mse_loss(l):
            [y_true, y_pred] = l
            dim = K.cast(K.shape(y_pred)[-1], K.dtype(y_pred))
            ms = K.sum(K.square(y_true - y_pred), axis = -1) / dim
            if self.loss == "si_mse":
                ms -= K.square(K.sum(y_true - y_pred, axis = -1)) / K.square(dim)
            return ms

        def mae_loss(l):
            [y_true, y_pred] = l
            dim = K.cast(K.shape(y_pred)[-1], K.dtype(y_pred))
            ms = K.sum(K.abs(y_true - y_pred), axis = -1) / dim
            if self.loss == "si_mae":
                ms -= K.abs(K.sum(y_true - y_pred, axis = -1)) / dim
            return ms

        def cos_loss(l):
            [y_true, y_pred] = l
            a = K.sum(y_true * y_pred, axis = -1)
            b = K.sqrt(K.sum(K.square(y_true), axis = -1))
            c = K.sqrt(K.sum(K.square(y_pred), axis = -1))
            return K.square(1- a/(b*c))

        def kld_loss(l):
            [y_true, y_pred] = l
            return kld(K.softmax(y_true), K.softmax(y_pred))
            
        if self.loss == "kld":
            loss_func = kld_loss
        elif self.loss.endswith("mae"):
            loss_func = mae_loss
        elif self.loss.endswith("mse"):
            loss_func = mse_loss
        elif self.loss == "cos":
            loss_func = cos_loss
        
            
        def make_encoder(dim, initializer):
                layers = []
                args = {"units": repsize, "input_shape": (dim,), 
                        "kernel_initializer": initializer, "activation": self.activation}

                for _ in range(self.nhid):
                    layers.append(Dense(**args))
                    if "input_shape" in args:
                        del args["input_shape"]

                args["activation"] = "linear"
                layers.append(Dense(**args, use_bias = False))
                return Sequential(layers)
            
        def make_decoder(dim, initializer):
                layers = []
                args = {"units": repsize, "input_shape": (repsize,), 
                        "kernel_initializer": initializer, "activation": self.activation}

                for _ in range(self.nhid):
                    layers.append(Dense(**args))
                    if "input_shape" in args:
                        del args["input_shape"]
                
                args["units"] = dim
                args["activation"] = "linear"
                
                layers.append(Dense(**args, use_bias = False))
                return Sequential(layers)
        

        views_train = [Input((dim,)) for dim in dims]

        encoders = []
        decoders = []
        initializer = glorot_uniform(seed = self.seed)
        for dim in dims:
                encoders.append(make_encoder(dim, initializer))
                decoders.append(make_decoder(dim, initializer))
                #encoders.append(Dense(repsize, input_shape = (dim,), use_bias = False, kernel_initializer = initializer))
                #decoders.append(Dense(dim, input_shape = (repsize,), use_bias = False, kernel_initializer = initializer))

        outputs_train = []
        loss_layer = Lambda(loss_func, output_shape = lambda shapes: (shapes[0][0],))
        norm_layer = Lambda(lambda x:K.l2_normalize(x, axis = -1))
        reps_train = [encoders[i](views_train[i]) for i in range(len(dims))]

        if self.mode == "coupled":
                if self.norm:
                    reps_train = [norm_layer(x) for x in reps_train]
                for i in range(len(dims)):
                    for j in range(len(dims)):
                        decoded = decoders[j](reps_train[i])
                        outputs_train.append(loss_layer([views_train[j], decoded]))

        elif self.mode == "AAEME":
                rep_train = add(reps_train)
                if self.norm:
                    rep_train = norm_layer(rep_train)
                for j in range(len(dims)):
                    decoded = decoders[j](rep_train)
                    outputs_train.append(loss_layer([views_train[j], decoded]))

        self.train_model = Model(views_train, add(outputs_train))
        self.train_model.compile(loss = lambda t,p:p, optimizer = "adam", target_tensors = K.constant((1,)))
            
        views_apply = [Input((dim,)) for dim in dims]
        outputs_apply = [encoders[i](views_apply[i]) for i in range(len(encoders))]
            
        if self.mode == "AAEME":
                outputs_apply = [add(outputs_apply)]
        if self.norm:
                outputs_apply = [norm_layer(x) for x in outputs_apply]
            
        self.apply_model = Model(views_apply, outputs_apply)


    def fit(self, views, vocab = None):
        import tensorflow as tf
        
        batch_size = 10000 // len(views)
        
        if vocab is None:
            dims = [view.shape[1] for view in views]
        else:
            dims = [view.get_vectors(vocab[:batch_size]).shape[1] for view in views]

        def generator():
            state = np.random.RandomState(self.seed)
            if vocab is None:
                samples = views[0].shape[0]
                X = views
            else:
                samples = len(vocab)
                X = vocab

            idx = list(range(samples))
            while True:
                state.shuffle(idx)
                if vocab is None:
                    X = [x[idx] for x in X]
                else:
                    X = [X[i] for i in idx]
                for start in range(0, samples, batch_size):
                    if start + batch_size > samples: 
                        continue

                    if vocab is None:
                        batch = [x[start:start+batch_size] for x in X]
                    else:
                        batch = [view.get_vectors(X[start:start+batch_size]) for view in views]

                    yield(batch, [])

        def get_spe():
            if vocab is None:
                return views[0].shape[0] // batch_size
            return len(vocab) // batch_size

        with tf.device("/gpu:{}".format(self.device)):

            self.build(dims)
            #train_model.fit_generator(generator(views), steps_per_epoch = get_spe(views), epochs = 300, verbose = 1)
            self.train_model.fit_generator(generator(), steps_per_epoch = get_spe()*self.epochs, epochs = 1, verbose = self.verbose)
            #train_model.fit(views, [], batch_size = 10000 // len(views) + 1, epochs = 300, verbose = 1)
            
            #del train_model
            #del decoders


    def transform(self, views):
        reps = self.apply_model.predict(views)
        if hasattr(reps, "__len__"):
            reps = sum(reps, 0)
        return reps

    def transform_as_list(self, views):
        return self.apply_model.predict(views)
