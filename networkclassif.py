import itertools
from sklearn.utils import check_random_state #, check_X_y, check_array, column_or_1d
import numbers
from sklearn.ensemble.base import BaseEnsemble, _partition_estimators
from sklearn.ensemble.bagging import _parallel_build_estimators
from sklearn.externals.joblib import Parallel, delayed
import pandas as pd
import numpy as np
import pickle
import scipy.sparse as sp
from sklearn.ensemble import BaggingClassifier
#from sklearn.utils.validation import has_fit_parameter
from sklearn.utils.random import sample_without_replacement
import bisect as bis

import copy

MAX_INT = np.iinfo(np.int32).max
def random_feature_sets(n_feature_sets,n_features,causgenes):
    feature_sets = np.zeros((n_feature_sets, n_features))
    for i in range(n_feature_sets):
        #print(np.array(sample_without_replacement(len(causgenes),n_features)))
        npcausgenes = np.array(causgenes) #cuz ordinary array has problems with array as an index
        feature_sets[i, :] = npcausgenes[sample_without_replacement(len(causgenes),n_features)]
    return feature_sets

def _parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                               seeds, verbose, choosing_function, choosing_function_kwargs):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_samples = ensemble.max_samples
    max_features = ensemble.max_features

    if (not isinstance(max_samples, (numbers.Integral, np.integer)) and
            (0.0 < max_samples <= 1.0)):
        max_samples = int(max_samples * n_samples)

    if (not isinstance(max_features, (numbers.Integral, np.integer)) and
            (0.0 < max_features <= 1.0)):
        max_features = int(max_features * n_features)

    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features

    #following lines were commented due to older sklearn which does not have function has_fit_parameter
    #support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
    #                                          "sample_weight")
    support_sample_weight = False #hardcoded limit for testing

    # Build estimators
    estimators = []
    estimators_samples = []
    estimators_features = []
    # Draw features
    feature_sets = np.array(choosing_function(*choosing_function_kwargs)) #mozna jenom args, kwargs je pro dikt, ale to bude opraveno :)
    for i in range(n_estimators):
        if verbose > 1:
            print("building estimator %d of %d" % (i + 1, n_estimators))

        random_state = check_random_state(seeds[i])
        seed = check_random_state(random_state.randint(MAX_INT))
        estimator = ensemble._make_estimator(append=False)

        try:  # Not all estimator accept a random_state
            estimator.set_params(random_state=seed)
        except ValueError:
            pass
        if bootstrap_features:
            raise NotImplementedError("TO DO: Implement this")
        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                indices = random_state.randint(0, n_samples, max_samples)
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts

            else:
                not_indices = sample_without_replacement(
                    n_samples,
                    n_samples - max_samples,
                    random_state=random_state)

                curr_sample_weight[not_indices] = 0

            estimator.fit(X[:, feature_sets[i]], y, sample_weight=curr_sample_weight)
            samples = curr_sample_weight > 0.

        # Draw samples, using a mask, and then fit
        else:
            if bootstrap:
                indices = random_state.randint(0, n_samples, max_samples)
            else:
                indices = sample_without_replacement(n_samples,
                                                     max_samples,
                                                     random_state=random_state)

            sample_counts = np.bincount(indices, minlength=n_samples)

            # print(X.shape)
            # print(feature_sets.shape)
            # print(feature_sets[i])
            estimator.fit((X[indices])[:, feature_sets[i]], y[indices])
            samples = sample_counts > 0.

        estimators.append(estimator)
        estimators_samples.append(samples)
    estimators_features = feature_sets

    return estimators, estimators_samples, estimators_features

def get_network_features():
    raise NotImplementedError('To do')
class GeneNetwork():
    """ Class for creating the network that will be used for classification.
    The goal of this class is to put the network manipulation out of each classifier
    """
    inf =float("inf")
    #temp function
    #resembles the original script
    def __init__(self, K=1):
        self.K=K
        self.loadData()
        self.mergeNetworks()
        self.generateWalk()


    def loadData(self):
        path='data/MDS/datasets/'
        task='BMBT_DT5q'

        #load data
        classes=pd.read_csv(path+'classes_'+task+'.csv')
        self.classes=classes[['class']].as_matrix().T[0]
        self.data = pd.read_csv('data/MDS/datasets/BMBT_DT5q.csv').as_matrix()

        gene2mir=pd.read_csv('data/MDS/interactions/gene2mir.csv')
        gene2mir=sp.coo_matrix(gene2mir.values)
        self.miRNA2gene = gene2mir.T  #because of changes in xgene.org
        with open('data/MDS/interactions/gene2gene.pickle','rb') as f: gene2gene=pickle.load(f, encoding='latin1')

        self.gene2gene=(gene2gene+gene2gene.T).tocoo()
        self.gene2gene.data/=self.gene2gene.data
        self.causgenes = None

        if self.miRNA2gene is not None:
            self.mergeNetworks()
        else:
            self.nw=gene2gene

    def mergeNetworks(self):
        """ Merges gene-gene interaction network with gene-miRNA interaction
        network to create unified network to sample features.
        """

        # take only the max component of the nw
        self.gene2gene=sp.coo_matrix(self.gene2gene)
        self.miRNA2gene=sp.coo_matrix(self.miRNA2gene)
        nc,complabs=sp.csgraph.connected_components(self.gene2gene, directed=True)
        max_comp=np.where(complabs==2)[0]
        # Project the candidate genes into the max component
        #if self.causgenes:
        if False:  #ASK ABOUT IT!
            self.causgenes=self.selphen(set(max_comp),self.causgenes)
        else:
            self.causgenes=np.random.choice(max_comp,100)
        self.causgenes=list(self.causgenes)
#
        # merge network matrices
        nw=sp.csr_matrix((np.concatenate((self.gene2gene.data,self.miRNA2gene.data)), \
        (np.hstack((self.gene2gene.row,self.miRNA2gene.row)),\
        np.hstack((self.gene2gene.col,self.miRNA2gene.row+self.gene2gene.shape[0])))), shape=(sum(self.miRNA2gene.shape),sum(self.miRNA2gene.shape)))


        # normalize network matrix
        self.nw=nw+sp.diags(np.ones(nw.shape[0]),0)
        self.nw.data/=self.nw.data
    def lazy_trans(self,nw,cc,p_stay=.5,iters=10):
        dd=np.array(nw.sum(axis=1)).T[0]
        d_inv=1/dd
        d_inv[d_inv==self.inf]=0
        D_inv=sp.diags(d_inv.T,0)
        W=D_inv*nw

        for i in range(iters):
            dia=np.ones(W.shape[0])
            dia[cc]=0
            I=sp.diags(dia,0)
            E=p_stay*np.ones(W.shape[0])
            E[cc]=1
            E=sp.diags(E,0)
            W=E*W+(1-p_stay)*I

        return W,dd

    def selphen(self,univ,phngns):
        phnset=set([])
        for it in phngns:
            phnset|=set(it)
        return univ & phnset

    def generateWalk(self,p_stay=1,iters=1):
        """ Generates random walk distribution p_k for pseudorandom
        sampling features for the forest

        p_k+1=p_k*W,
        p_k[i] ... probability distribution of reaching gene i after k steps
        W ... graph transition probability matrix

        cdf[c]=(sorted cumulated p_k, original_indices) ... cummulated
        for a causal gene c
        """

        self.cdf=dict([])
        for cc in self.causgenes:
            W,dd=self.lazy_trans(self.nw,cc,p_stay=p_stay,iters=iters)

            pk=np.zeros(W.shape[1])
            pk[W.getrow(cc).indices]=W.getrow(cc).data
            pks=[pk]
            for k in range(1,self.K):
                pk=pk*W
                pks+=[pk]
            pk=np.zeros(W.shape[1])
            for pK in pks:
                pk+=pK
            pk/=self.K
            isort=np.argsort(pk)
            self.cdf[cc]=(np.cumsum(pk[isort]),isort)

    def get_n_feature_sets(self, n_features_sets=1,n_features = 1):
        """

        :param n_features_sets: The number of generated feature sets
        :param n_features: The number of features in each features
        :return: features_sets list of n_features_sets each containing n_features
        """

        # Sampling of the roots:
        # the probability of choosing a gene as a root is proportional to its
        # degree.
        seed_pots=np.array([self.cdf[seed][0][self.cdf[seed][0]>0].shape[0] for seed in self.causgenes])
        seed_cdf=np.cumsum(seed_pots*1./seed_pots.sum())
        roots=[]
        features_sets = []
        for rs in np.random.random_sample(n_features_sets):
            x=bis.bisect(seed_cdf,rs)
            roots+=[ self.causgenes[x]]

        for root in roots:

            fsubs=set([root])

            weights=copy.deepcopy(self.cdf[root][0])
            iw=copy.deepcopy(self.cdf[root][1])

            suma=weights[-1]
            suma=weights.max()

            seed_pot=self.cdf[root][0][self.cdf[root][0]>0].shape[0]

            #print('seed pot {:d} ',seed_pot)

            # Performs weighted random sampling WITHOUT replacement,
            # i.e. for each number of a random vector all genes long,
            # it chooses an apropriate gene. The choice is proportional
            # the probability of particular feature. Namely, the values of
            # cumulative distribution sread 0-1 interval. The interval length
            # of each feature corresponds to its probability.
            for rs in np.random.random_sample(min(int(seed_pot-1),n_features)):

                rs*=suma
                x=bis.bisect(weights,rs)
                wd=weights[x]-weights[x-1]
                suma-=wd # effective removal of selected item
                weights[x:]-=wd
                fsubs.add(iw[x])

            fsubs=list(fsubs)
            #print('fsubs')
            #print(fsubs)
            features_sets.append(fsubs)

        return features_sets

class NetworkBaggingClassifier(BaggingClassifier):
    """ For each classifier in ensemble, it subsets feature space and train the classifier
     using only the subset of the feature space  based on network of interactions.
    """
    def __init__(self,
                 # gene2gene,
                 gnetwork = None,
                 # causgenes=None,
                 # miRNA2gene=None,
                 K=1,
                 n_estimators=1000,
                 criterion="gini",
                 max_depth=1,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 n_jobs=1,
                 random_state=None,
                 max_features='auto',
                 fsubsets=None,
                 bootstrap=False,
                 bootstrap_features=False,
                 oob_score=False,
                 verbose=0,
                 base_estimator=None,
                 max_samples=1.0,
                 n_features = 100,
                 choosing_function = None,
                 choosing_function_kwargs = None,
                 ):
        super(NetworkBaggingClassifier, self).__init__(
            base_estimator=base_estimator,
             n_estimators=n_estimators,
             max_samples=max_samples,
             max_features=max_features,
             bootstrap=bootstrap,
             bootstrap_features=bootstrap_features,
             oob_score=oob_score,
             n_jobs=n_jobs,
             random_state=random_state,
             verbose=verbose)
        self.choosing_function = choosing_function
        self.choosing_function_kwargs = choosing_function_kwargs
        if not choosing_function_kwargs and choosing_function == sample_without_replacement:
            choosing_function_kwargs = (n_estimators,n_features)

    def fit(self, X, y, sample_weight=None):
        """Build a Bagging ensemble of estimators from the training
           set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrices are accepted only if
            they are supported by the base estimator.

        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like, shape = [n_samples] or None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if the base estimator supports
            sample weighting.

        Returns
        -------
        self : object
            Returns self.
        """
        random_state = check_random_state(self.random_state)

        # Convert data
        #X, y = check_X_y(X, y, ['csr', 'csc', 'coo'])

        # Remap output
        n_samples, self.n_features_ = X.shape
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if isinstance(self.max_samples, (numbers.Integral, np.integer)):
            max_samples = self.max_samples
        else:  # float
            max_samples = int(self.max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        if isinstance(self.max_features, (numbers.Integral, np.integer)):
            max_features = self.max_features
        else:  # float
            max_features = int(self.max_features * self.n_features_)

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        # Free allocated memory, if any
        self.estimators_ = None

        # Parallel loop
        #n_jobs, n_estimators, starts = _partition_estimators(n_estimators=self.n_estimators, n_jobs=self.n_jobs)
        n_jobs, n_estimators, starts = _partition_estimators(self)
        MAX_INT = np.iinfo(np.int32).max
        seeds = random_state.randint(MAX_INT, size=self.n_estimators)

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                verbose=self.verbose,
                choosing_function = self.choosing_function,
                choosing_function_kwargs = self.choosing_function_kwargs)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ = list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_samples_ = list(itertools.chain.from_iterable(
            t[1] for t in all_results))
        self.estimators_features_ = list(itertools.chain.from_iterable(
            t[2] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self
