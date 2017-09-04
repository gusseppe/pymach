import os
import re
import csv
import errno
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder
from collections import OrderedDict


def sizeof_file(name, suffix='B'):
    num = os.stat(name).st_size
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def mean_error_localization(y_pred, y_real):
    pos = {}
    # pos['1'] = (0,0)
    # pos['2'] = (1.5,0)
    # pos['3'] = (3,0)
    # pos['4'] = (0,1.5)
    # pos['5'] = (1.5,1.5)
    # pos['6'] = (3,1.5)

    y = -1.5
    for e in range(1,16):
        if (e % 3) == 1:
            y += 1.5

        pos[str(e)] = (((e-1)%3)*1.5, y)

    error = []
    for a,b in zip(y_pred, y_real):
        error.append(np.linalg.norm(np.array(pos[str(a)]) - np.array(pos[str(b)])))

    return error, np.mean(np.array(error))

def model_map_name(name):
    models = ['GradientBoostingClassifier', 'ExtraTreesClassifier',
              'RandomForestClassifier', 'DecisionTreeClassifier',
              'LinearDiscriminantAnalysis', 'SVC', 'KNeighborsClassifier',
              'LogisticRegression', 'AdaBoostClassifier', 'VotingClassifier',
              'GaussianNB', 'MLPClassifier']

    names = ['GDM', 'ET', 'RF', 'DT', 'LDA', 'SVM', 'k-NN', 'LoR',
             'AB', 'Voting', 'GNB', 'MLP']

    dict_names = dict(zip(models, names))

    return dict_names[name]

def total_report():
    import define
    # import analyze
    import prepare
    import fselect
    import evaluate
    import improve

    path = './market/LocalizationNew_Tx'

    for i in range(5, 8):

        data_name = "./uploads/LocalizationNew_Tx"+str(i)+".csv"
        # data_name = "iris.csv"
        class_name = "class"
        definer = define.Define(
            data_path=data_name,
            header=None,
            response=class_name).pipeline()

        preparer = prepare.Prepare(definer).pipeline()
        selector = fselect.Select(definer).pipeline()
        evaluator = evaluate.Evaluate(definer, preparer, selector)
        improver = improve.Improve(evaluator).pipeline()

        improver.save_full_report('./market/LocalizationNew_Tx'+str(i))
        improver.save_score_report('./market/LocalizationNew_Tx'+str(i))


def error_loc_plot(errors, path):
    # d = {}
    # d['values'] = errors
    # d['Model'] = [path]*len(errors)
    #
    # df = pd.DataFrame(d)
    sns.set_style("whitegrid")

    list_df = []
    for d in errors:
        list_df.append(pd.DataFrame(d))

    df_errors = pd.concat(list_df)
    df_errors.reset_index(drop=True)

    data_name = path.replace(".csv", "")
    # plt.figure(figsize=(10, 10))
    ax = sns.boxplot(x="model", y="values", data=df_errors)
    ax.set(xlabel='Model evaluated', ylabel='Error (m)')
    # _ = sns.boxplot(x="model", y="values", data=df_errors, showmeans=True)
    # _ = sns.stripplot(x="model", y="values", data=df_errors, jitter=True, edgecolor="gray")

    # tips = sns.load_dataset("tips")
    # ax = sns.boxplot(x="day", y="total_bill", data=tips)

    # medians = df_errors.groupby(['model'])['values'].mean().values
    medians = df_errors.groupby(['model'], sort=False)['values'].mean().values
    median_labels = [str(np.round(s, 2)) for s in medians]

    pos = range(len(medians))
    for tick, label in zip(pos, ax.get_xticklabels()):
        ax.text(pos[tick], medians[tick], median_labels[tick],
                horizontalalignment='center', size='x-small', color='black', weight='semibold')
    print(df_errors.describe())
    plt.savefig(data_name+'_error'+'.eps', format='eps')
    plt.close()


def localization():
    APP_PATH = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(APP_PATH, 'uploads')
    devices_file = 'devices'
    devices_path = os.path.join(UPLOAD_FOLDER, devices_file)
    files = [f for f in os.listdir(UPLOAD_FOLDER) if re.match(r'p[0-9]+', f)]
    len_files = len(files)
    print(devices_path)
    print('len_file', len_files)
    print('files', files)

    if len_files != 0:
        for i in range(1, len_files+1):
            NUMBER = i
            name_file = 'p'+str(NUMBER)
            path_file = os.path.join(UPLOAD_FOLDER, name_file)
            with open(path_file, 'r') as in_file:
                stripped = (line.strip() for line in in_file)
                lines = (line for line in stripped if line)
                grouped = zip(*[lines] * 2)
                with open(path_file+'.csv', 'w') as out_file:
                    writer = csv.writer(out_file)
                    writer.writerow(('Beacon', 'RSSI'))
                    writer.writerows(grouped)

            df = pd.read_csv(path_file+'.csv')
            os.remove(path_file+'.csv')
            condition = ''

            with open(devices_path, 'r') as inFile:
                stripped = (line.strip() for line in inFile)

                for x in stripped:
                    condition += '(Beacon == "' + str(x) + '")' + ' | '
                condition = condition[:-3]
            df = df.query(condition)

            encoder = LabelEncoder()
            df['Beacon'] = encoder.fit_transform(df['Beacon'])

            mapped_df = {}
            for b, data in df.groupby('Beacon'):
                mapped_df[b] = data

            dict_to_df = OrderedDict()
            for index in range(len(mapped_df)):
                dict_to_df[index] = mapped_df[index]['RSSI']

            df_outliers = pd.DataFrame(dict_to_df)

            #Dealing with nan and duplicates
            df_outliers = df_outliers.fillna(method='ffill')
            df_outliers = df_outliers.dropna()
            df_outliers = df_outliers.drop_duplicates()

            #Scaling
            scaler = StandardScaler()
            df_outliers_t = scaler.fit_transform(df_outliers.ix[:, 0:len(df_outliers.columns)-1])
            df_outliers_t = pd.DataFrame(df_outliers_t)

            #Removing outliers aplying the mean
            df_no_outliers = df_outliers_t[df_outliers_t.apply(lambda x: np.abs(x - x.median()) / x.std() < 3).all(axis=1)]
            df_no_outliers = scaler.inverse_transform(df_no_outliers)
            df_no_outliers = pd.DataFrame(df_no_outliers)
            vector_no_out = NUMBER*np.ones((len(df_no_outliers),), dtype=np.int)
            vector_out = NUMBER*np.ones((len(df_outliers),), dtype=np.int)
            df_no_outliers = df_no_outliers.assign(position=pd.Series(vector_no_out).values)
            df_outliers = df_outliers.assign(position=pd.Series(vector_out).values)

            #Saving files with outliers and without it.
            df_outliers.to_csv(os.path.join(UPLOAD_FOLDER, name_file+'_outliers'+'.csv'), index=False)
            df_no_outliers.to_csv(os.path.join(UPLOAD_FOLDER, name_file+'_no_outliers'+'.csv'), index=False)


        for file in ["outliers.csv", "no_outliers.csv"]:
            file_out = os.path.join(UPLOAD_FOLDER, "Localization_"+file)
            with open(file_out,"a") as fout:
                name_first_file = os.path.join(UPLOAD_FOLDER, "p1_"+file)
                with open(name_first_file) as f_in:
                    for line in f_in:
                        fout.write(line)
                os.remove(name_first_file)
                for num in range(2,len_files+1):
                    name = os.path.join(UPLOAD_FOLDER,"p"+str(num)+"_"+file)
                    with open(name) as f:
                        f.readline() # skip the header
                        for line in f:
                             fout.write(line)
                    os.remove(name)
    else:
        print("There are no files for localization")
