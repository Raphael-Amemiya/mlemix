import networkx as nx
import numpy as np
import pandas as pd

from copy import deepcopy
from joblib import delayed, Parallel 
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted


class HierarchicalLocalClassifier(ClassifierMixin, BaseEstimator):
    """Classificador com base em Classificação Hierárquica Local.
    
    Este classificador se baseia no "Local Classifier per Level”,
    um tipo de Classificador Hierárquico Local, que leva em conta
    a estrutura hierárquica dos dados para treinar um
    classificador por nível hierárquico. 

    Para evitar inconsistências, após o primeiro nível, cada
    classificador local é treinado apenas com amostras das populações
    dos nós parentais que obtiveram uma proporção de ancestralidade
    acima de um valor de corte. No final os resultados de cada nó são
    multiplicados para obtenção da probabilidade das amostras 
    pertencerem a cada população.
    
    Parameters
    ----------
    estimator : object
        classificador que será utilizado como classificador local em cada nível. 
    proba_threshold : float, default=0.01
        Proporção de ancestralidade de corte para definir quais populações
        serão usadas nos classificadores do nível seguinte.

    Attributes
    ----------
    classes_ : ndarray de formato (n_classes,)
        Nome das classes usadas no classificador hirárquico.
    hierarchy_graph_ : networkx.Digraph
        Gráfo direcionado, representando a estrutura hieráquica das
        classes.
    self.hierarchy_nodes_ : list
        Lista contendo os nós em ordem hierárquica.

    Note
    -----
    Para explicação sobre Classificação Hierárquica, consulte o artigo:
    Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical
    classification across different application domains.
    Data Mining and Knowledge Discovery, 22(1), 31-72.
    """
    
    def __init__(
        self,
        estimator,
        proba_threshold = 0.01
    ):
        self.estimator = estimator
        self.proba_threshold = proba_threshold

        
    def fit(self, X, y):
        """Ajuste dos classificadores.
        
        Parameters
        ----------
        X : array-like de formato (n_sample, n_features)
            Array com os valores de atributo para cada amostra.
        y : array-like de formato (n_samples,)
            Array com as classes esperadas de cada amostra.

        Returns
        -------
        self : object
            Retorna a própria instância.
        """
        # Check do sklearn
        check_classification_targets(y)
        self.X_train = X
        y = np.atleast_2d(y)
        self.hierarchy = self.init_hierarchy(y)

        # Inicialição do grafo
        self.hierarchy_graph_ = nx.from_edgelist(self.hierarchy, create_using=nx.DiGraph())
        self.hierarchy_nodes_ = list(nx.topological_sort(self.hierarchy_graph_))
        
        # Classes
        self.classes_ = [np.unique(y[:,c]) for c in range(y.shape[1])]
        self.n_classes_ = [len(np.unique(y[:, z])) for z in range(y.shape[1])]

        # Fit
        self.first_layer_estimator_ = deepcopy(self.estimator[0])
        self.first_layer_estimator_.fit(X, y[:, 0])
        
        self.layers_freq_ = dict()
        for i in range(1, y.shape[1]):
            next_layer_estimator = deepcopy(self.estimator)
            self.layers_freq_[i]=y[:, i]
            
        return self
    

    def init_hierarchy(self, y):
        """Inicialização do array com a estrutura hierárquica."""
        self.depth_ = y.shape[1]
        tree_labels = np.vstack(list({tuple(row) for row in y}))
        tree_labels = np.hstack(
            (
                np.repeat([["root"]], tree_labels.shape[0], axis=0),
                tree_labels
            )
        )

        hierarchy = [tree_labels[:, [l, l+1]] for l in range(self.depth_)]
        hierarchy = np.vstack(hierarchy)
        
        hierarchy = np.vstack(list({tuple(row) for row in hierarchy}))

        hierarchy = np.array(
            [tuple(row) for row in hierarchy if row[0]!=row[1]],
            dtype=[("node1", "O"), ("node2", "O")]
        )

        return hierarchy


    def select_pops(self, X):
        """Seleção das populações para predição."""

        # Predição no primeiro nível
        X = X.reshape(1, -1)
        proba = list()
        estimator = self.first_layer_estimator_
        proba.append(
            estimator.predict_proba(X)
        )
        proba_s = pd.Series(proba[0].ravel(), index=estimator.classes_, name=0)
        
        for layer, freq in self.layers_freq_.items():
            # Em cada nível, checa-se a probabilidade de corte para
            # escolher as populações do nível anterior que serão usadas
            # no classificador do nível atual.
            keep = proba_s[proba_s>self.proba_threshold].index
            keep_successors = [
                self.hierarchy_graph_.successors(k) for k in keep
            ]
            keep_successors = [e for v in keep_successors for e in v]

            base_class = np.unique(freq)
        
            proba_successors = np.zeros(np.size(base_class))
            proba_successors = pd.Series(
                proba_successors,
                index=base_class,
                name=layer
            )

            pop_filter = np.isin(freq, keep_successors)

            # Nos níveis com mais de uma população, usa-se as
            # amostras das população selecionadas para treinar o
            # classificador e fazer predição de ancestralidade.
            if len(keep_successors)==1:
                proba_successors[keep_successors] = np.array([1.0])

            elif len(keep_successors)>1:
                y_train = freq[pop_filter]
                X_train = self.X_train[pop_filter, :]

                estimator = deepcopy(self.estimator[layer])
                estimator.fit(X_train, y_train)

                proba_successors[estimator.classes_] = estimator.predict_proba(X).ravel()

            # Após todas as predições em cada nívels. As predições são padronizadas
            # para obter a predição final com todas as populações.
            for k in proba_s.index:
                k_successors = list(self.hierarchy_graph_.successors(k))
                proba_successors_node = proba_successors[k_successors]

                if proba_successors_node.sum() == 0.0:
                    proba_successors_node[proba_successors_node==0.0] = 1.0/proba_successors_node.size
                
                proba_successors_node = proba_successors_node/proba_successors_node.sum()
                proba_successors[k_successors] = proba_s[k]*proba_successors_node

            proba.append(proba_successors.values.reshape(1,-1))
            proba_s = proba_successors.copy()
        
        return np.concatenate(proba, axis=1)

    
    def predict_proba(self, X):
        """Retorna a probabilidade de cada classe.

        Parameters
        ----------
        X : array-like de formato (n_Sample, n_features)
            Array com os valores de atributo para cada amostra.

        Returns
        -------
        ndarray de formato (n_samples, n_classes)
            Probabilidads de cada classe. 
        """
        check_is_fitted(self)
        X = np.atleast_2d(X)

        proba = np.apply_along_axis(self.select_pops, 1, X)
        proba = np.concatenate(proba, axis=0)
        
        i = 0
        chunks = []

        for pops in self.classes_:
            chunks.append(proba[:, i:i+len(pops)])
            i += len(pops)
        
        return chunks
        

    def predict(self, X):
        """Retorna a classe mais provável.

        A partir do resultado de probabilidade estimada de cada
        classe, retorna a classe mais provável para cada amostra
        em um array X. 

        Parameters
        ----------
        X : array-like de formato (n_samples, n_features)
            Amostras de input.

        Returns
        -------
        ndarray de formato (n_samples, n_classes)
            Classes preditas para cada amostra de X.
        """
        proba = self.predict_proba(X)
        classes_proba = zip(self.classes_, proba)
        
        return [a[np.argmax(b, axis=1)] for a,b in classes_proba]