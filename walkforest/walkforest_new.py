# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 16:12:55 2014

@author: andelmi2
MODIFIED SO IT WORKS WITH SKLEARN 15
IT WONT WORK WITH SKLEARN 16, SEE COMMENT NEAR
_partition_estimators - kuncvlad
"""


from sklearn.tree import (DecisionTreeClassifier)
from sklearn.ensemble.forest import ForestClassifier
from sklearn.ensemble.base import _partition_estimators
import numpy as np
import scipy.sparse as sp
from sklearn.tree._tree import DTYPE
from sklearn.utils import array2d
from sklearn.externals.joblib import Parallel, delayed

import bisect as bis

import copy

inf =float("inf")

def _parallel_predict_proba(ests, X, n_classes, n_outputs):
    """Private function used to compute a batch of predictions within a job."""
    n_samples = X.shape[0]

    if n_outputs == 1:
        proba = np.zeros((n_samples, n_classes))

        for est in ests:
            tree=est[0]
            fsubs=est[1]
            proba_tree = tree.predict_proba(X[:,fsubs])

            if n_classes == tree.n_classes_:
                proba += proba_tree

            else:
                for j, c in enumerate(tree.classes_):
                    proba[:, c] += proba_tree[:, j]
    else:
        proba = []

        for k in xrange(n_outputs):
            proba.append(np.zeros((n_samples, n_classes[k])))

        for est in ests:
            tree=est[0]
            fsubs=est[1]
            proba_tree = tree.predict_proba(X[:,fsubs])

            for k in xrange(n_outputs):
                if n_classes[k] == tree.n_classes_[k]:
                    proba[k] += proba_tree[k]

                else:
                    for j, c in enumerate(tree.classes_[k]):
                        proba[k][:, c] += proba_tree[k][:, j]
    return proba

def lazy_trans(nw,cc,p_stay=.5,iters=10):
    dd=np.array(nw.sum(axis=1)).T[0]    
    d_inv=1/dd
    d_inv[d_inv==inf]=0
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
    
def selphen(univ,phngns):
    phnset=set([])    
    for it in phngns:
        phnset|=set(it)
    return univ & phnset

class WalkForestHyperLearner:
    """ Heuristicky urci delku prochazky po nahodne pasece
    Parameters
        ----------
        gene2gene : DataFrame, or scipy.sparse.matrix-like 
            protein-protein interaction matrix
        miRNA2gene : DataFrame, or scipy.sparse.matrix-like 
            target-miRNA interaction matrix  
        causgenes : gene-sets-like set of candidate genes
            for particular disese terms.      
    """
    
    def __init__(self,
                 gene2gene,
                 causgenes=None,
                 miRNA2gene=None,
                 walk_lengths=range(1,10),
                 n_estimators=1000,
                 eps=.01,
                 criterion="gini",
                 max_depth=1,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 n_jobs=1,
                 random_state=None,
                 max_features='auto',
                 fsubsets=None,
                 bootstrap=False,
                 oob_score=False,
                 verbose=0
                 ):
        self.eps=eps             
        self.n_estimators=n_estimators             
        self.learners=[]
        for k in walk_lengths:             
            self.learners+=[WalkForestClassifier(gene2gene,causgenes,miRNA2gene=miRNA2gene,\
                n_estimators=self.n_estimators, random_state=random_state,K=k, max_features=max_features,\
                max_depth=max_depth,min_samples_leaf=min_samples_leaf,bootstrap=bootstrap)]
                
    def fit(self,X,y):
       #print('Zacinam fitovat Walk Forest Hyper Learner')
       self.heur=[]
       for learner in self.learners:
           learner.fit(X,y)
           # incidence of underfitted trees
           I=[sum(y!=t[0].predict(X[:,t[1]])) for t in learner.estimators_]
           self.heur+=[sum(np.array(I)>0)*1./self.n_estimators]
       self.heur=np.array(self.heur)
       
       converg=np.where(self.heur<self.eps)[0]
       self.opt_length=np.argmin(self.heur[1:]-self.heur[:-1]) if len(converg)==0 else converg[0]
  
       return self
       
    def predict(self,X):
       return self.learners[self.opt_length].predict(X)
            
        
class WalkForestClassifier(ForestClassifier):
    """ Nahodna paseka
    Parameters
        ----------
        gene2gene : DataFrame, or scipy.sparse.matrix-like 
            protein-protein interaction matrix
        miRNA2gene : DataFrame, or scipy.sparse.matrix-like 
            target-miRNA interaction matrix  
        causgenes : gene-sets-like set of candidate genes
            for particular disese terms.      
    """
    
    def __init__(self,
                 gene2gene,
                 causgenes=None,
                 miRNA2gene=None,
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
                 oob_score=False,
                 verbose=0
                 ):
        super(WalkForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "max_features",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        
        #self.nw=nw
        # symetrize protein interactions
        gene2gene=(gene2gene+gene2gene.T).tocoo()
        gene2gene.data/=gene2gene.data
        
        
        if miRNA2gene is not None:
            self.mergeNetworks(gene2gene,miRNA2gene,causgenes)
        else:
            self.nw=gene2gene
        self.K=K
        #self.cdf=cdf
        self.generateWalk()
        self.max_depth=max_depth
        self.min_samples_leaf=min_samples_leaf
        self.max_features = max_features

            
    def fit(self,X,y):
        #print('Zacinam fitovat Walk Forest Classifier')
        
        y = np.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]
        y = self._validate_y(y)
        self.n_classes_ = self.n_classes_[0]
        self.classes_ = self.classes_[0] 
        np.random.seed(self.random_state)
        #print(self.random_state)

        self.estimators_=[]
        # print('causgenes')
        # print(self.causgenes)
        # print(len(self.causgenes))
        # print(self.nw.shape)
        # Sampling of the roots:
        # the probability of choosing a gene as a root is proportional to its
        # degree.
        # print('self.cdf')
        # print(self.cdf)
        seed_pots=np.array([self.cdf[seed][0][self.cdf[seed][0]>0].shape[0] for seed in self.causgenes])
        seed_cdf=np.cumsum(seed_pots*1./seed_pots.sum())
        #print('seed_pots')
        #print(seed_pots)
        #print('seed_cdf')
        #print(seed_cdf)
        roots=[]
        for rs in np.random.random_sample(self.n_estimators):
            x=bis.bisect(seed_cdf,rs)
            roots+=[ self.causgenes[x]]
        #print(roots)

        for root in roots:
            
            self.base_estimator=DecisionTreeClassifier()
            self.base_estimator.max_features=None
            self.base_estimator.max_depth=self.max_depth
            self.base_estimator.min_samples_leaf=self.min_samples_leaf        
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
            for rs in np.random.random_sample(min(int(seed_pot-1),self.max_features)):

                rs*=suma
                x=bis.bisect(weights,rs)
                wd=weights[x]-weights[x-1]                  
                suma-=wd # effective removal of selected item             
                weights[x:]-=wd
                fsubs.add(iw[x])

            fsubs=list(fsubs)
            #print('fsubs')
            #print(fsubs)

            if self.bootstrap:
            
                indices = np.random.randint(0, X.shape[0], X.shape[0])
                sample_counts = np.bincount(indices, minlength=X.shape[0])
                self.estimators_.append((self.base_estimator.fit(X[:,fsubs],y,sample_weight=sample_counts),np.array(fsubs)))
            else:
                self.estimators_.append((self.base_estimator.fit(X[:,fsubs],y),np.array(fsubs)))
                
        return self
        
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
            W,dd=lazy_trans(self.nw,cc,p_stay=p_stay,iters=iters)
                        
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
            
    def mergeNetworks(self,gene2gene,miRNA2gene,causgenes):
        """ Merges gene-gene interaction network with gene-miRNA interaction
        network to create unified network to sample features.
        """

        # take only the max compopnennt of the nw
        gene2gene=sp.coo_matrix(gene2gene)
        miRNA2gene=sp.coo_matrix(miRNA2gene)
        nc,complabs=sp.csgraph.connected_components(gene2gene, directed=True)
        max_comp=np.where(complabs==2)[0]
        # Project the candidate genes into the max compononet
        if causgenes:
            print('zde')
            self.causgenes=selphen(set(max_comp),causgenes) 
        else:
            self.causgenes=np.random.choice(max_comp,100)
        self.causgenes=list(self.causgenes)
#       
        # merge network matrices
        nw=sp.csr_matrix((np.concatenate((gene2gene.data,miRNA2gene.data)), \
        (np.hstack((gene2gene.row,miRNA2gene.row)),\
        np.hstack((gene2gene.col,miRNA2gene.row+gene2gene.shape[0])))), shape=(sum(miRNA2gene.shape),sum(miRNA2gene.shape)))

        
        # normalize network matrix        
        self.nw=nw+sp.diags(np.ones(nw.shape[0]),0)
        self.nw.data/=self.nw.data 
        

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the mean predicted class probabilities of the trees in the forest.
        """
        # Check data
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = array2d(X, dtype=DTYPE)

        # Assign chunk of trees to jobs
        n_jobs, n_trees, starts = _partition_estimators(self)
        # IMPORTANT: _partition_estimators signature will be changed in future release
        # by then use: (it will probably work)
        #n_jobs, n_estimators, starts = _partition_estimators(n_estimators=self.n_estimators, n_jobs=self.n_jobs)

        # Parallel loop
        all_proba = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(_parallel_predict_proba)(
                self.estimators_[starts[i]:starts[i + 1]],
                X,
                self.n_classes_,
                self.n_outputs_)
            for i in range(n_jobs))

        # Reduce
        proba = all_proba[0]

        if self.n_outputs_ == 1:
            for j in range(1, len(all_proba)):
                proba += all_proba[j]

            proba /= self.n_estimators

        else:
            for j in range(1, len(all_proba)):
                for k in range(self.n_outputs_):
                    proba[k] += all_proba[j][k]

            for k in range(self.n_outputs_):
                proba[k] /= self.n_estimators

        return proba
        
    