# mlemix

Repositório para os modelos MLEMix, RegMLEMix e HierarchicalLocalClassifier desenvolvidos para o projeto de mestrado "Análise de ancestralidade genética da população de São Paulo."

## Requerimentos

Verifique os requerimentos no arquivo environment.yml.

## Installation

Primeiro, clone o repositório do github ``mlemix``. Em seguida, crie um ambiente virtual com conda ou mamba para instalar as dependências necessárias.

```
cd mlemix

conda env create -f environment.yml
```

## Modo de uso

Exemplo:
```
    >>> from mlemix.mlemix import MLEMix

    >>> model = MLEMix()
    >>> model.fit(X, y)
    MLEMix()
    >>> print(model.predict(X_new))
    [2]
    >>> print(model.predict_proba(X_new))
    [[0.30, 0., 0.70 ]]
```

## Licença

MIT

Autor
-------

`mlemix` por `Raphael Amemiya <raphael.amemiya@usp.br>`.