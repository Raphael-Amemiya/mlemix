"""
    Implementação de ancestralidade global com abordagem 'Model-based'.
    
    Abordagens 'Model-based' de ancestralidade global estimam
    ancestralidade de um indivíduo com base em um modelo estatístico.
    Por exemplo, a probabilidade de observar um determinado perfil
    genético é obtida usando proporções de ancestralidade e frequências
    alélicas assumindo equilíbrio de Hardy-Weinberg e equilíbrio de
    ligação entre os loci.
    
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

import numpy as np
import numpy.ma as ma
import scipy.optimize as optimize
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import label_binarize
from sklearn.utils.validation import (
    check_is_fitted,
    check_non_negative,
    check_array,
    check_X_y)


class MLEMix(ClassifierMixin, BaseEstimator):
    """Estimador de ancestralidade global.
    
    Estimador específico para dados compostos por SNPs bialélicos. O 
    modelo utiliza Máxima Verossimilhança (MLE) e permite obter
    frequência alélica de cada SNP para cada grupo populacional.
    Também permite obter as proporções de ancestralidade de dois ou
    mais grupos populacionais para cada amostra.
    
    Parameters
    ----------
    _alpha : float, default=0.000001
        Parâmetro para resolver frequências iguais a 0.0 ou 1.0.
        Similar ao Laplace smoothing (0 para nenhum smoothing).
        
    _tol : float, default=0.000001
        Tolerância para término da minimização. Usado no 
        scipy.optimize.minimize.
        
    no_dataset_nan : bool, default=False
        Selecione 'False' para permitir valores nulos no dataset de
        treino.
    
    no_sample_nan : bool, default=False
        Selecione 'False' para permitir valores nulos nas amostras de
        predição.

    Attributes
    ---------
    classes_ : ndarray de formato (n_classes,)
        Nome dos grupos populacionais usados no estimador.
        
    feature_count_ : ndarray de formato (n_classes, n_features)
        Número de SNPs encontrado no treinamento.
        
    feature_freq_ : ndarray de formato (n_classes, n_features)
        Frequência alélica dos grupos populacionais para os alelos
        fornecidos para cada SNP.

    n_features_in_ : int
        Número de SNPs usados no treinamentos.
 
    feature_names_in_ : list de tamanho n_features_in_
        rsid dos SNPs usados no treinamento.

    alleles2_ : list de tamanho n_features_in_
        Alelos dos SNPs usados no treinamento.

    nonan_snps_ : list de tamanho n_features_in_
        Array de True e False. True para os SNPs que tem frequência
        alélica em todos as populações.

    Notes
    -----
    Modelo baseado nas explicações de Estimativa de Máxima
    Verossimilhança para estimar ancestralidade global contidas nas
    referências abaixo:
    
    Alexander, D. H., Novembre, J., & Lange, K. (2009). Fast
    model-based estimation of ancestry in unrelated individuals. Genome
    research, 19(9), 1655-1664.

    Frudakis, T. (2010). Molecular photofitting: predicting ancestry
    and phenotype using DNA. Elsevier.
    
    Examples
    --------
    >>> from mlemix.mlemix import MLEMix

    >>> model = MLEMix()
    >>> model.fit(X, y)
    MLEMix()
    >>> print(model.predict(X_new))
    [2]
    >>> print(model.predict_proba(X_new))
    [[0.30, 0., 0.70 ]]
    """
    
    def __init__(
            self,
            alpha=0.000001,
            tol=0.000001,
            no_dataset_nan=False,
            no_sample_nan=False):
        
        self._alpha = alpha
        self._tol = tol
        self.no_dataset_nan = no_dataset_nan
        self.no_sample_nan = no_sample_nan


    def _check_X(self, X):
        """Validação de X, usado somente nos métodos predict*.
        
        Verifica se X possui a mesma quantidade de SNPs que foram
        usados no treinamento. Em seguida é feito validação com a
        função check_array do sklearn.
        """
                
        _, n_features = X.shape
        if n_features != self.n_features_in_:
            raise ValueError(
                f"X tem {n_features} features, mas {self.__class__.__name__} "
                f"era esperado {self.n_features_in_} features como input."
            )
            
        array_checked = check_array(
            X,
            accept_sparse="csr",
            force_all_finite=self.no_sample_nan
        )
        
        return array_checked


    def _check_X_y(self, X, y, reset=True):
        """Validação de X e y usado nos métodos fit/partial_fit.
        
        Salva o número de SNPs em X. Em seguida é feito validação com
        a função check_X_y do sklearn.
        """
        
        if reset:
            _, self.n_features_in_ = X.shape
        
        checked_X_y = check_X_y(
            X,
            y,
            accept_sparse="csr",
            force_all_finite=self.no_dataset_nan
        )
        
        return checked_X_y


    def _check_partial_fit_first_call(self, classes, n_features, rsid, allele2):
        """Método para validar o primeiro partial_fit.
        
        Retorna 'True' se for o primeiro partial_fit, ao mesmo tempo
        que salva no self informações sobre todas os grupos
        populacionais, que serão apresentadas ao modelo durante o treinamento.
        As chamadas subsequentes do partial_fit checa se 'classes'
        continua consistente com o primeiro partial_fit.
        """
        
        if getattr(self, "classes_", None) is not None:

            if not np.array_equal(self.classes_, classes):
                raise ValueError(
                    "'classes=%r' não é o mesmo do fornecido no "
                    "partial_fit anterior: %r" % (classes, self.classes_)
                )
            elif self.n_features_in_ != n_features:
                raise ValueError(
                    "Número de features não bate com 'partial_fit' anterior"
                )
                
            elif getattr(self, "feature_count_", None) is None:
                raise ValueError(
                    "feature_count_ está vazio, não pode fazer "
                    "partial fit sem fornecer valores inciais."
                )

            elif getattr(self, "chr_count_", None) is None:
                raise ValueError(
                    "chr_count_ está vazio, não pode fazer "
                    "partial fit sem fornecer valores inciais."
                )
            elif rsid and self.feature_names_in_:
                 if not np.array_equal(self.feature_names_in_, rsid):
                    raise ValueError(
                        "'rsid=%r' não confere com o 'partial_fit' "
                        "anterior: %r" % (rsid, self.feature_names_in_)
                    )

            elif allele2 and self.allele2:
                 if not np.array_equal(self.allele2, allele2):
                    raise ValueError(
                        "'allele2=%r' não confere com o partial_fit "
                        "anterior: %r" % (allele2, self.allele2)
                    )
             
            return False

        else:
            # Primeiro partial_fit
            self.classes_ = classes
            self.n_features_in_ = n_features
            self.feature_names_in_ = rsid
            self.allele2_ = allele2

            return True


    def _init_counters(self, n_classes, n_features):
        self.chr_count_ = np.zeros(
            (n_classes, n_features),
            dtype=np.float64
        )
        
        self.feature_count_ = np.zeros(
            (n_classes, n_features),
            dtype=np.float64
        )


    def _count(self, X, Y):
        """Contabilização do número de cromossomos e alelos em X.
        
        Para cada SNP e população, contabiliza o número de cromossomos
        e alelos. Ignora a ocorrência de valores nulos. 
        """
        # Usa função check_non_negative do sklearn para checar ausência
        # de valores negativos.
        check_non_negative(X, "MLEMix (input X)")
        
        # Contabilização do número de alelos por (população, SNP).
        feature_count = np.ma.dot(Y.T, X)
        self.feature_count_ += feature_count
        
        # Contabilização do número de cromossomos por (população, SNP).
        chr_count = Y.T @ (2*(~X.mask))
        self.chr_count_ += chr_count


    def _update_snp_freq(self):
        """Cálculo da frequência alélica e smoothing"""
        
        feature_count = np.ma.masked_invalid(self.feature_count_)
        chr_count = np.ma.masked_invalid(self.chr_count_)
        
        self.feature_freq_ = feature_count / chr_count
        self.feature_freq_ = np.round(self.feature_freq_, 6)
        self.feature_freq_[self.feature_freq_==0.0] += self._alpha
        self.feature_freq_[self.feature_freq_==1.0] -= self._alpha

        # True para SNPs que contém valor de frequência em todas as
        #populações.
        self.nonan_snps_ = ~np.any(self.feature_freq_.mask, axis=0)

        
    def partial_fit(self, X, y, rsid=None, allele2=None):
        """Fit incremental em um chunk de amostras.
        
        Esse método é usado para treinar um conjunto de dados
        incrementalmente chunk por chunk sem precisar carregar
        todo o dataset na memória.

        Parameters
        ----------
        X : ndarray de formato (n_samples, n_features)
            Em que n_samples é o número de amostras e n_features é o
            número de SNPS. Contém o número de allele2 de cada SNP
            observado nas amostras usadas para o treinamento.
            
        y : ndarray de formato (n_samples,)
            Nome dos grupos populacionais(classes) de cada amostra do
            chunk. Deve ser fornecido na mesma ordem que as amostras
            aparecem no X.

        rsid : ndarray de formato (n_features,), opcional
            rsid de todos os SNPs usados no treinamento. Deve ser
            fornecido na mesma ordem que os SNPs estão no X.

        allele2 : ndarray de formato (n_features,), opcional
            Alelo 2 de todos os SNPs usados no treinamento. Deve ser
            fornecido na mesma ordem que os SNPs estão no X.
            
        Returns
        -------
        self : object
            Retorna a instância.
        """

        first_call = not hasattr(self, "classes_")
        X, y = self._check_X_y(X, y, reset=first_call)
        X = np.ma.masked_invalid(X)
        _, n_features = X.shape

        uniq_classes = np.array(sorted(np.unique(y)))
        
        if self._check_partial_fit_first_call(
            uniq_classes,
            n_features,
            rsid,
            allele2):
            # Primeiro partial_fit:
            # inicialização da contabilização de cromossomos e alelos
            n_classes = len(uniq_classes)
            self._init_counters(n_classes, self.n_features_in_)

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # somente uma classe
                Y = np.ones_like(Y)

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d e y.shape[0]=%d não são compatíveis."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        # label_binarize() retorna arrays com dtype=np.int64.
        # Conversão para np.float64 para calculos subsequentes.
        Y = Y.astype(np.float64, copy=False)
      
        # Contabilização dos cromossomos e alelos.
        self._count(X, Y)

        # Cálculo/Atualização da frequência alélica.
        self._update_snp_freq()

        return self


    def fit(self, X, y, rsid=None, allele2=None):
        """Treinamento do modelo.
        
        Parameters
        ----------
        X : ndarray de formato (n_samples, n_features)
            Em que n_samples é o número de amostras e n_features é o
            número de SNPS. Contém o número do alelo2 de cada SNP
            observado nas amostras usadas no treinamento.
            
        y : ndarray de formato (n_samples,)
            Nome das populações(classes) de cada amostra do chunk. Deve
            ser fornecido na mesma ordem que as amostras aparecem no X.
            
        rsid : ndarray de formato (n_features,), opcional
            rsid de todos os SNPs usados no treinamento. Deve ser
            fornecido na mesma ordem que os SNPs estão no X.

        allele2 : ndarray de formato (n_features,), opcional
            Alelos2 de todos os SNPs usados no treinamento. Deve ser
            fornecido na mesma ordem que os SNPs estão no X.
            
        Returns
        -------
        self : object
            Retorna a instância.
        """

        X, y = self._check_X_y(X, y)
        X = np.ma.masked_invalid(X)
        
        _, self.n_features_in_ = X.shape
        
        self.feature_names_in_ = rsid
        self.allele2_ = allele2

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        n_classes = len(self.classes_)

        if Y.shape[1] == 1:
            if n_classes == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # uma classe
                Y = np.ones_like(Y)

        # label_binarize() retorna arrays com dtype=np.int64.
        # Conversão para np.float64 para calculos subsequentes.
        Y = Y.astype(np.float64, copy=False)
      
        # Contabilização dos cromossomos e alelos.
        self._init_counters(n_classes, self.n_features_in_)
        self._count(X, Y)
        
        # Cálculo da frequência alélica.
        self._update_snp_freq()

        return self
    
    
    def _log_likelihood(self, X, admixture_fraction):
        """Método para a função de verossimilhança.
        
        Cálculo do log likelihood partindo dos valores de contagem de
        alelos de cada SNP, frequência alélica dos respectivo alelos e
        proporção de ancestralidade para cada população.
    
        Como requisito do modelo, assumimos que os SNPs são
        independentes entre si, neste caso, apenas para fins de
        cálculo, será utilizado o log para substituir as mutiplicações
        entre SNPs por soma.
        
        Parameters
        ----------
        X : ndarray de formato (n_features,)
            Contém o número de alelle2 de cada SNP observado na
            amostra. 
            
        admixture_fraction : ndarray de formato (n_classes,)
            É um array com as proporções de ancestralidade para cada
            população. Esses valores atuam como 'peso' na fórmula e são
            esses valores que se deve procurar para maximizar a função
            de verossimilhança (nesta aplicação será de minimização usando
            scipy.optimize.minimize).
        
        Returns
        -------
        float
            Valor de likelihood.
        

        Notes
        -----
        self.feature_freq_.T é um array no formato (n_features,
        n_classes) com a frequência do alelo2 de cada SNP para
        cada uma das populações.
        
        like2 é o log likelihood da observação do alelo2 dado as
        frequência alélica de cada população e supostas proporções de
        ancestralidade.
        
        like1 é o log likelihood da observação do allele1(
        obtido com 2 - número de allelo2) dado as frequência
        alélica de cada população (1-frequência do alelo2) e supostas
        proporções de ancestralidade.
        
        like é a soma de like1 e like2. Retorna-se o valor negativo de
        like para poder minimizar a função.
        """
        
        # Dosagem dos alelos
        a1_dose = 2-X[self.nonan_snps_]
        a2_dose = X[self.nonan_snps_]
        
        # Frequências ponderado pelas proporções de ancestralidade.
        # Cálculo necessário, pois uma pessoa pode ter mais de uma
        # ascendência.
        a1_weight_coef = np.log(
            np.ma.dot(
                1 - self.feature_freq_[:, self.nonan_snps_].T,
                admixture_fraction
            )
        )
        a2_weight_coef = np.log(
            np.ma.dot(
                self.feature_freq_[:, self.nonan_snps_].T,
                admixture_fraction
            )
        ) 

        like1 = np.ma.dot(a1_dose, a1_weight_coef)

        like2 = np.ma.dot(a2_dose, a2_weight_coef)

        like = like1 + like2

        return -like     


    def perform_mle(self, X):
        """Minimização da função de verossimilhança.
        
        Para minimizar a função de verossimilhança é utilizado o
        scipy.optimize.minimize.
        
        Parameters
        ----------
        X : ndarray de formato (n_features,)
            Contém o número de alelle2 de cada SNP observado na
            amostra.
            
        Returns
        -------
        percent : ndarray de formato (n_classes, )
            Resultado de ancestralidade. São as proporções para cada
            uma das populações. Somatório igual a 1.
            
        Notes
        -----
        Como se deja decobrir as proporções de ancestralidade que
        minimizam a função de verossimilhança, além da função de 
        verossilhança que se deseja minimizar devemos fornecer um
        palpite inicial. Este palpite inical é um array (n_classes,)
        preenchido com '1' dividido por n_classes. Também é necessário
        algumas condições/restrições: valores em que cada população
        podem variar (0, 1) e somatório de todas as proporções devem
        ser igual a 1. E um valor de término da minimização 'tol'.
        """      

        # palpite inicial
        n_classes = len(self.classes_)
        initial_guess = np.ones(n_classes) / n_classes

        # restrições das frações de ancestralidade
        bounds = tuple((0, 1) for i in range(n_classes))
        constraints = ({'type': 'eq', 'fun': lambda af: np.sum(af) - 1})
        
        # MLE
        likelihood_func = lambda af: self._log_likelihood(X, af)
        admix_frac = optimize.minimize(
            likelihood_func,
            initial_guess,
            bounds=bounds,
            constraints=constraints,
            tol=self._tol).x
                
        return admix_frac


    def predict_proba(self, X):
        """Método para obter as proporções de ancestralidade.
        
        Para cada amotra fornecida, obtém as proporções de ancestralidade
        das populações presentes no treinamento.
        
        Parameters
        ----------
        X : ndarray de formato (n_samples, n_features)
            Em que n_samples é o número de amostras e n_features é o
            número de SNPS. Contém o número de alelo2 de cada SNP
            observado nas amostras que se deseja saber a ancestralidade.
        
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Proporções de ancestarlidade. A ordem das colunas é a mesma
            do self.classes_.
        """
 
        check_is_fitted(self)
        x_2d = np.atleast_2d(X)
        x_2d = self._check_X(x_2d)
        x_2d = np.ma.masked_invalid(x_2d)
        proba = np.apply_along_axis(self.perform_mle, 1, x_2d)
        
        return proba
    

    def predict(self, X):
        """Predição da população mais provável.
        
        A partir dos resultados de proporção de ancestralidade, retorna
        o nome da população que apresentou maior valor de
        ancestralidade.
        
        Parameters
        ----------
        X : ndarray de formato (n_samples, n_features)
            Em que n_samples é o número de amostras e n_features é o
            número de SNPS. Contém o número de alelo2 de cada SNP
            observado nas amostras que se deseja saber a população
            de origem mais provável.
            
        Returns
        -------
        ndarray de formato (n_samples,)
            Nome da população mais provável.
        """

        proba = self.predict_proba(X)
        
        return self.classes_[np.argmax(proba, axis=1)]

    
    def load_snp_freq(
            self,
            classes,
            freq,
            rsid=None,
            allele2=None,
            feature_count=None,
            chr_count=None):
        """Método para inserir valores de frequência sem fazer treino.
        
        Se a pessoa já tiver valores de frequência, ela pode
        carregá-los e fazer as predições sem necessidade de fit.
        
        Paramters
        ---------
        freq : ndarray de formato (n_classes, n_features)
            Frequência alélica do alelo2 de cada SNP para cada
            população.
            
        classes : ndarray de formato (n_classes,)
            Nome de todas as classes que se pretende carregar as
            frequências. Deve estar na ordem correta em relação ao
            freq.
            
        rsid : ndarray de formato (n_features,)
            rsid de todos os SNPs. Deve estar na ordem correta
            em relaçao ao freq.
            
        allele2 : ndarray de formato (n_features,)
            Alelo2 de todos os SNPs. Deve estar na ordem correta
            em relaçao ao freq.

        feature_count : ndarray (n_classes, n_features) default=None
            Array com as contagens de cada alelo2 de cada SNP para
            cada uma das populações. Não precisa ser fornecido se não
            tiver essa informação, porém não será possível realizar
            partial fit futuramente.
        
        chr_count : ndarray (n_classes, n_features) default=None
            Array com o total de alelos 1 e 2 de cada SNP para
            cada uma das populações. Não precisa ser fornecido se não
            tiver essa informação, porém não será possível realizar
            partial fit futuramente.

        Returns
        -------
        None
        """

        self.classes_ = classes
        self.feature_names_in_ = rsid
        self.allele2_ = allele2
        self.feature_freq_ = ma.masked_invalid(freq)
        self.feature_count_ = feature_count
        self.chr_count_ = chr_count
        self.n_features_in_ = freq.shape[1]

        # True para SNPs que contém valor de frequência em todas as
        #populações.
        self.nonan_snps_ = ~np.any(self.feature_freq_.mask, axis=0)
        
        
    @property
    def alpha(self):
        return self._alpha


    @alpha.setter
    def alpha(self, value):
        self._alpha = value


    @property
    def tol(self):
        return self._tol


    @tol.setter
    def tol(self, value):
        self._tol = value


class RegMLEMix(MLEMix):
    """Estimador de ancestralidade global com regularização.
    
    Extensão do MLEMix para Implementação do termo de regularização 
    conforme descrito em Alexander e Lange (2011).
    
    Parameters
    ----------
    lamb : float, default=0.1
        Hiperparâmetro que controla o quanto os coeficientes serão
        regularizados. Valor igual a 0, anula a regularização. 
        
    eps : float, default=0.1
        Hiperparâmetro que controla o quanto os coeficientes serão
        regularizados. Valores devem ser maior que 0 e menor ou igual
        a 1.

    See Also
    --------
    MLEMix : Estimador de ancestralidade global
    """
    def __init__(
            self,
            alpha=0.000001,
            tol=0.000001,
            no_dataset_nan=False,
            no_sample_nan=False,
            lamb=0.1,
            eps=0.1):
        self.lamb = lamb
        self.eps = eps
        super().__init__(
        alpha = alpha,
        tol = tol,
        no_dataset_nan = no_dataset_nan,
        no_sample_nan = no_sample_nan,
        )

    def _log_likelihood(self, X, admixture_fraction):
        """Método para a função de verossimilhança.
        
        Cálculo do log likelihood partindo dos valores de contagem de
        alelos de cada SNP, frequência alélica dos respectivo alelos e
        proporção de ancestralidade para cada população.
    
        Como requisito do modelo, assumimos que os SNPs são
        independentes entre si, neste caso, apenas para fins de
        cálculo, será utilizado o log para substituir as mutiplicações
        entre SNPs por soma.
        
        Parameters
        ----------
        X : ndarray de formato (n_features,)
            Contém o número do alele2 de cada SNP observado na
            amostra. 
            
        admixture_fraction : ndarray de formato (n_classes,)
            É um array com as proporções de ancestralidade para cada
            população. Esses valores atuam como 'peso' na fórmula e são
            esses valores que se deve procurar para maximizar a função

            de verossimilhança (nesta aplicação será de minimização usando
            scipy.optimize.minimize).
        
        Returns
        -------
        float
            Valor de likelihood.
        

        Notes
        -----
        self.feature_freq_.T é um array no formato (n_features,
        n_classes) com a frequência do alelo2 de cada SNP para
        cada uma das populações.
        
        like2 é o log likelihood da observação do alelo2 dado as
        frequência alélica de cada população e supostas proporções de
        ancestralidade.
        
        like1 é o log likelihood da observação do allele1(
        obtido com 2 - número de allelo2) dado as frequência
        alélica de cada população (1-frequência do alelo2) e supostas
        proporções de ancestralidade.
        
        like é a soma de like1 e like2. Retorna-se o valor negativo de
        like para poder minimizar a função.

        Termo de regularização conforme em:
        Alexander, D. H., & Lange, K. (2011). Enhancements to the
        ADMIXTURE algorithm for individual ancestry estimation. 
        BMC bioinformatics, 12(1), 1-6.
        """
        like = super()._log_likelihood(X, admixture_fraction)
        like = -1*like        

        reg = np.sum(np.log(1+(admixture_fraction/self.eps))/np.log(1+(1/self.eps)))
        reg_like = like - (self.lamb*reg)

        return -reg_like


class IncrementalMLEMix(RegMLEMix):
    """Classe para usar o método partial_input do RegMLEMix.
    
    Parameters
    ----------
    n_split : int, default=5
        Número de chunks de amostras para dividir o conjunto de
        treinamento para o partial_fit.
        
    See Also
    --------
    RegMLEMix : Estimador de ancestralidade global com regularização.
    """
    def __init__(
            self,
            alpha=0.000001,
            tol=0.000001,
            no_dataset_nan=False,
            no_sample_nan=False,
            n_split=5,
            lamb=0,
            eps=0.1):
        
        super().__init__(
            alpha=alpha,
            tol=tol,
            no_dataset_nan=no_dataset_nan,
            no_sample_nan=no_sample_nan,
            lamb=lamb,
            eps=eps   
        )

        self.n_split = n_split
        
        
    def fit(self, X, y):
        """Treinamento do modelo.
        
        Parameters
        ----------
        X : ndarray de formato (n_samples, n_features)
            Em que n_samples é o número de amostras e n_features é o
            número de SNPS. Contém o número de alelo2 de cada SNP
            observado nas amostras usadas no treinamento.
            
        y : ndarray de formato (n_samples,)
            Nome das populações(classes) de cada amostra do chunk. Deve
            ser fornecido na mesma ordem que as amostras aparecem no X.
        
        Returns
        -------
        self : object
            Retorna a instância.
        """
        skf = StratifiedKFold(n_splits=self.n_split)
        for _, fit_index in skf.split(X, y):
            X_fit = X[fit_index, :]
            y_fit = y[fit_index]
            super().partial_fit(X_fit, y_fit)

        return self