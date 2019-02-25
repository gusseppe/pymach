import pandas as pd
import numpy as np
from sklearn import utils
from sklearn.metrics.scorer import check_scoring
from sklearn.model_selection import KFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.model_selection import train_test_split

def distance_error(estimator, X, y, test_size=0.2, seed=7):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed)
    estimator.fit(X_train, y_train)
    distanciaEntrePuntos = 1.5
    y_pred = estimator.predict(X_test)
    x1 = np.int32((y_pred + 2) % 3)
    y1 = np.int32((y_pred - 1) / 3)
    x2 = np.int32((y_test + 2) % 3)
    y2 = np.int32((y_test - 1) / 3)
    # pasar variacion a distancias metros
    vx = np.abs(x1 - x2) * distanciaEntrePuntos
    vy = np.abs(y1 - y2) * distanciaEntrePuntos
    err_distance = np.mean(np.sqrt(vx*vx + vy*vy))
    return err_distance

def _createDataset(urlDataset, seed = 7):
    dataset = pd.read_csv(urlDataset)
    names_ = dataset.columns.values
    dataset = utils.shuffle(dataset, random_state=seed).reset_index(drop=True)
    dataset = dataset.apply(pd.to_numeric)
    X = dataset[names_[:-1]]
    y = dataset[names_[-1]]
    return X, y


def _individual_to_params(individual, parametros):
    individual = np.int32(individual)
    name_values = list(parametros.items())
    return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))


def getModelAccuracy(parametros, individual, estimator, score_cache, resultados):
    X, y = _createDataset("Tx_0x06") # adaptado de búsqueda de configuración hay que adaptar para sacarlo
    params = _individual_to_params(individual, parametros)
    score = 0
    scoring = "accuracy"
    nombreModelo = str(estimator).split('(')[0]
    paramkey = nombreModelo + str(np.int32(individual))
    if paramkey in score_cache:
        score = score_cache[paramkey]
    else:
        resultIndividuo = []
        cv = KFold(n_splits=10, shuffle=False)
        scorer = check_scoring(estimator, scoring=scoring)
        for train, test in cv.split(X, y):
            resultIndividuo.append(_fit_and_score(estimator=estimator, X=X, y=y, scorer=scorer, parameters=params,
                                            train=train, test=test, verbose=0, fit_params=None, return_times=True))
        accuracy = np.array(resultIndividuo)[:, 0] #accuracy
        runtime = np.array(resultIndividuo)[:, 2] + np.array(resultIndividuo)[:, 1] #runtime train+test
        # error = distance_error(estimator, X, y)
        score = accuracy.mean()
        score_cache[paramkey] = score
        dict_result = {
            'Modelo': nombreModelo,
            'Parametros': params,
            'Accuracy': accuracy.mean(),
            'stdAccuracy': accuracy.std(),
            'Runtime': runtime.mean(),
            'accuracy_values': accuracy,
            'runtime_values': runtime,
        }
        resultados.append(dict_result)
    return score

def prettyPrint(indice, individual, parametros):
    dict_result = _individual_to_params(individual, parametros)
    dict_result['Accuracy'] = individual[-1]
    result = pd.DataFrame([dict_result]).to_string(index=False).split('\n')
    if(indice == 0):
        print('indice\t' + result[0])
    print(str(indice) + '\t' + result[1])
