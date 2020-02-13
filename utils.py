import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

# Features used for the benchmark model
BASIC_FEATURES = ['original_shape_Sphericity',
                  'original_shape_SurfaceVolumeRatio',
                  'original_shape_VoxelVolume',
                  'SourceDataset',
                  'Nstage',
                  'original_glcm_JointEntropy',
                  'original_glcm_Id',
                  'original_glcm_Idm']


def preprocess_radiomics(file_path):
    """ Load radiomics data from file and preprocess them """
    # load radiomics features
    radiomics = pd.read_csv(file_path + '/features/radiomics.csv', index_col=0, header=1)
    radiomics = radiomics[1:]  # skip blank line
    radiomics.index = radiomics.index.astype('int')  # convert indices to int
    radiomics.index.names = ['PatientID']  # rename index column
    return radiomics


def preprocess_clinical(file_path):
    """ Load clinical data from file and preprocess them """
    clinical = pd.read_csv(file_path + '/features/clinical_data.csv', index_col=0)
    clinical['SourceDataset'] = clinical['SourceDataset'].str[-1].astype('int') - 1
    return clinical


def preprocess(file_path, separated_output=False, select_features='basic', normalize=False, scaler=None):
    """ Load data from file and preprocess them

    Parameters
    ----------
    file_path : str ('train' or 'test')
        path where to find the feature datasets

    separated_output : boolean
        if True, return separately features and output (containing survival information), else return everything in the
        same DataFrame

    select_features : str ('basic', 'basic_age', 'all', 'all-[...]')
        indicates which features we want to keep from the available feature dataset

    normalize : boolean
        if True, the data are normalized by columns and the associated scaler is returned

    scaler : StandardScaler
        if it exists, the data are scaled according to the provided Scaler.

    Returns
    -------
    dataset : Pandas Dataframe
        dataset containing the required features and the output ('Event' and 'SurvivalTime')
        depending on separated_output.

    dic_return : dict
        contain the list of selected columns and optionally the output dataset and the Scaler.

    """
    assert not (normalize and (scaler is not None))  # cannot ask to get scaler and apply other scaler

    dic_return = {}

    clinical = preprocess_clinical(file_path)
    radiomics = preprocess_radiomics(file_path)

    # concatenate feature DataFrames
    features = pd.concat([clinical, radiomics], axis=1)

    if select_features == 'basic':
        selected_cols = BASIC_FEATURES.copy()

    elif select_features == 'basic_age':
        selected_cols = BASIC_FEATURES.copy()
        selected_cols.append('age')

    elif select_features.find('all') == 0:
        selected_cols = list(features.columns)
        select_features = select_features[4:]  # remove 'all'
        for feature_to_remove in select_features.split('-'):
            if feature_to_remove not in selected_cols:
                print("{} is not in the list of features".format(feature_to_remove))
                continue
            selected_cols.remove(feature_to_remove)

    else:
        raise ValueError("Invalid value for argument 'select_features'")

    dic_return['select_features'] = selected_cols.copy()
    features = features[selected_cols]

    if scaler is not None:
        print("Normalize feature columns with given scaler")
        features[:] = scaler.transform(features)

    if normalize:
        print("Normalize feature columns")
        scaler = StandardScaler()
        features[:] = scaler.fit_transform(features)
        dic_return['scaler'] = scaler

    if file_path == 'train':
        # load output table
        output = pd.read_csv('output.csv', index_col=0)

        if not separated_output:
            # concatenate features and output
            dataset = pd.concat([features, output.loc[features.index]], axis=1)
            selected_cols.extend(['SurvivalTime', 'Event'])

        else:
            dic_return['output'] = output.loc[features.index]
            dataset = features

    elif file_path == 'test':
        dataset = features

    return dataset[selected_cols], dic_return


def predict(model, dataset, save=False, filename=None, skmodel=False):
    """ Compute survival predictions

    Parameters
    ----------
    cph : model
        trained survival model

    dataset : Pandas DataFrame
        dataset on which the prediction is done.

    save : boolean
        if set to True the prediction table is saved as a csv file.

    filename : str
        name of the file into which the predictions are saved.

    skmodel : boolean
        whether the model is adapted from sklearn model

    Returns
    -------
    Pandas DataFrame containing predictions.
    """

    if skmodel:
        prediction = pd.DataFrame(model.predict(dataset), index=dataset.index)
    else:
        prediction = model.predict_median(dataset)

    prediction.index.name = 'PatientID'
    prediction['Event'] = 'nan'
    prediction.columns = ['SurvivalTime', 'Event']
    max_value = max(prediction['SurvivalTime'][prediction['SurvivalTime'] < np.inf]) + prediction['SurvivalTime'][
        prediction['SurvivalTime'] < np.inf].mean()
    prediction['SurvivalTime'] = prediction['SurvivalTime'].apply(lambda x: min(x, max_value))

    if save:
        if filename is None:
            filename = 'predictions'
        if filename.find('.') >= 0:
            filename = filename[
                filename.rfind('/') + 1, filename.find('.')]  # so we can support filename containing '.csv'
        prediction.to_csv('predictions/' + filename + '.csv')

    return prediction


def KME_score(c1, c2):
    """ Compute KM-score from confidence interval DataFrames (that we typically get calling kmf.confidence_interval_)

    Parameters
    ----------
    c1 : Pandas DataFrame
        two-columns DataFramed containing lower and upper bound of confidence interval of the first survival function

    c2 : Pandas DataFrame
        same for the second survival function

    Returns
    -------
    dif : float
        KM-score describing the discrepancy between two survival function envelop.
    """
    c1.index = c1.index.astype(int)
    c2.index = c2.index.astype(int)
    n = int(max(c1.index.max(), c2.index.max())) + 1

    # build the full dataset
    data = pd.DataFrame(index=np.arange(n), columns=['c1_down', 'c1_up', 'c2_down', 'c2_up'], dtype='float')

    # fill the dataset
    for df, df_lab in zip([c1, c2], [['c1_down', 'c1_up'], ['c2_down', 'c2_up']]):
        data.loc[df.index, df_lab] = np.array(df)

    # complete the dataset by interpolation
    data = data.interpolate(method='linear', limit_direction='forward')

    assert np.all(~data.isna())

    dif = np.linalg.norm(data['c1_up'] - data['c2_up'], ord=1) + np.linalg.norm(data['c1_down'] - data['c2_down'],
                                                                                ord=1)
    # we normalize to ease comparison
    dif = dif / n
    return dif


def tuned_regressors_to_json(tuned_regressors):
    """ Function that store our dictionary of models into .json file"""
    aux_dic = {}

    for regressor_id, v in tuned_regressors.items():
        aux_dic[regressor_id] = v.copy()
        aux_dic[regressor_id][0] = aux_dic[regressor_id][0]._params

    with open('saved_models/regressors.json', 'w') as fp:
        json.dump(aux_dic, fp, indent=4)