"""
    Implementação de algoritmo de classificação hierárquica.
    
    Este algoritmo foi inspirado na abordagem de Silla e Freitas(2011),
    quando as categorias a serem preditas possuem certa estrutura
    em níveis. No contexto de ancestralidade global, pode-se considerar
    diferentes níveis, como continentais, subcontinentais, países,
    subregiões ou subgrupos específicos.
    
    Autor
    -----
    Raphael Amemiya <raphael.amemiya@usp.br>
    
    Licença
    -------
    MIT
    
    Referência
    ----------    
    Amemiya, R. (2024). Análise de ancestralidade genética da população 
    de São Paulo.
"""

import dask.array as da
import dask
import networkx as nx
import numpy as np
import pandas as pd

from copy import deepcopy
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted

class HierarchicalLocalClassifier(ClassifierMixin, BaseEstimator):
    """Algoritmo com base em Classificação Hierárquica Local.
    
    Este algoritmo se baseia no "Local Classifier per Level”,
    um tipo de Classificador Hierárquico Local, que leva em conta
    a estrutura de níveis dos dados para treinar um
    classificador por nível. 

    Para evitar inconsistências, após o primeiro nível, cada
    modelo local é treinado apenas com amostras dos grupos
    populacionais dos nós parentais com maior proporção de 
    ancestralidade e/ou que obtiveram uma proporção de
    ancestralidade acima de um valor de corte. No final os
    resultados de cada nó são multiplicados para obtenção das
    proporções de ancestralidade das amostras pertencerem a cada grupo
    populacional.
    
    Parameters
    ----------
    estimator : object
        Modelo que será utilizado como classificador local em cada nível. 
    threshold : float, default=0.01
        Proporção de ancestralidade de corte para definir quais populações
        serão usadas nos modelos do nível seguinte.
    top : int, default=4
        Seleciona n populações com maior contribuição na composição
        de ancestralidade.
    return_last_layer : boolean, deafault=True
        Se True, o algoritmo retorna apenas os resultados do último nível
        de populações.

    Attributes
    ----------
    classes_ : ndarray de formato (n_classes,)
        Nome dos grupos populacionais usadas no modelo.
    hierarchy_graph_ : networkx.Digraph
        Gráfo direcionado, representando a estrutura em níveis.
    self.hierarchy_nodes_ : list
        Lista contendo os nós na ordem dos níveis.

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
        threshold = 0.01,
        top = 4,
        return_last_layer = True,
        n_jobs = 1
    ):
        self.estimator = estimator
        self.threshold = threshold
        self.top = top        
        self.return_last_layer = return_last_layer
        self.n_jobs = n_jobs

        
    def fit(self, X, y):
        """Ajuste dos modelos.
        
        Parameters
        ----------
        X : ndarray-like de formato (n_sample, n_features)
            Array com a dosagem do alelo fornecido para cada SNP. Cada
            linha uma amostra e cada coluna um SNP.
        y : ndarray-like de formato (n_samples,)
            Array com os grupos populacionais esperadas de cada
            amostra. Deve-se separar os grupos de cada nível por
            um '|'. Exemplo: Ásia|Leste_da_Ásia|Japão.

        Returns
        -------
        self : object
            Retorna a própria instância.
        """
        # Checagem de consitência dos dados
        # ref: sklearn.utils
        check_classification_targets(y)

        if isinstance(X, dask.array.core.Array):
            self.X_train = X.compute()
        else:
            self.X_train = X
        
        # Grupos populacionais do modelo
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        y = np.array([c.split('|') for c in y.ravel()])
        y = np.atleast_2d(y)
        self.hierarchy = self.init_hierarchy(y)

        # Inicialição do grafo
        self.hierarchy_graph_ = nx.from_edgelist(self.hierarchy, create_using=nx.DiGraph())
        self.hierarchy_nodes_ = list(nx.topological_sort(self.hierarchy_graph_))
        self._classes = [np.unique(y[:,c]) for c in range(y.shape[1])]
        self._n_classes = [len(np.unique(y[:, z])) for z in range(y.shape[1])]

        # Fit do modelo do primeiro nível
        self.first_layer_estimator_ = deepcopy(self.estimator[0])
        self.first_layer_estimator_.fit(self.X_train, y[:, 0])
        
        # Definição dos níveis
        self.layers_freq_ = dict()
        for i in range(1, y.shape[1]):
            next_layer_estimator = deepcopy(self.estimator)
            self.layers_freq_[i]=y[:, i]
            
        return self

    
    def _filter_proba(self, proba):
        """Checagem para manter n grupos populacionais com maior
        proporção de ancestralidade e/ou que a proporção de
        ancestralidade é maior que um determinado valor."""
        threshold_keep = proba[proba>self.threshold].index
        top_keep = proba.nlargest(n=self.top, keep="all").index
        
        return threshold_keep.intersection(top_keep)


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
        """Seleção das populações para previsão."""
        # Previsão no primeiro nível
        X = X.reshape(1, -1)
        proba = list()
        estimator = self.first_layer_estimator_
        proba.append(
            estimator.predict_proba(X)
        )
        proba_s = pd.Series(proba[0].ravel(), index=estimator.classes_, name=0)
        
        for layer, freq in self.layers_freq_.items():
            # Em cada nível, checa-se os grupos populacionais dentro
            # do nível anterior que serão usados no modelo do nível
            # atual. O método _filter_proba faz a avaliação usando os
            # parâmetros top e threshold.
            keep = self._filter_proba(proba_s)
            
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

            # Nos níveis com mais de um grupo populacional, usa-se as
            # amostras dos grupos selecionados para treinar o
            # modelo e fazer previsão das proporções de ancestralidade.
            if len(keep_successors)==1:
                proba_successors[keep_successors] = np.array([1.0])
            
            elif len(keep_successors)>1:
                y_train = freq[pop_filter]
                X_train = self.X_train[pop_filter, :]

                estimator = deepcopy(self.estimator[layer])
                estimator.fit(X_train, y_train)
                
                proba_successors[estimator.classes_] = estimator.predict_proba(X).ravel()

            # Após todas as previsões em cada nível. As previsões são padronizadas
            # para obter a previsão final com todos os pgrupos opulacionais.
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
        """Retorna a ancestralidade genética de cada grupo
        populacional.

        Parameters
        ----------
        X : ndarray-like de formato (n_sample, n_features)
            Array com a dosagem do alelo fornecido para cada SNP. Cada
            linha uma amostra e cada coluna um SNP.

        Returns
        -------
        ndarray de formato (n_samples, n_classes)
            Ancestralidade genética dos grupos populacionais. 
        """
        check_is_fitted(self)

        if isinstance(X, dask.array.core.Array):
            X = da.atleast_2d(X)

            X = X.compute()

            proba = np.apply_along_axis(self.select_pops, 1, X)

        else:
            X = np.atleast_2d(X)
        
            proba = np.apply_along_axis(self.select_pops, 1, X)

        proba = np.concatenate(proba, axis=0)
        
        i = 0
        chunks = []

        for pops in self._classes:
            chunks.append(proba[:, i:i+len(pops)])
            i += len(pops)
            
        last_layer_y = np.array([c.split('|') for c in self.classes_])[:, -1]
        idx = np.where(self._classes[-1][None, :] == last_layer_y[:, None])[1]

        return chunks[-1][:, idx]
        

    def predict(self, X):
        """Retorna o grupo populacional com maior proporção de
        ancestralidade genética.

        A partir do resultado de ancestralidade genética estimado
        de cada modelo, retorna o grupo populacional com maior
        proporção de ancestralidade para cada uma das amostras de 
        X.

        Parameters
        ----------
        X : ndarray-like de formato (n_sample, n_features)
            Array com a dosagem do alelo fornecido para cada SNP. Cada
            linha uma amostra e cada coluna um SNP.

        Returns
        -------
        ndarray de formato (n_samples,)
            Grupos populacionais previstos para cada amostra de X.
        """
        
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
