import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def convert_dataset_to_binary(df):
    """
    This function takes a Pandas DataFrame as input and converts it into a binary dataset. The function converts the
    categorical classes in the DataFrame into binary 1 or 0 classes using LabelBinarizer and the categorical
    features into binary 1 or 0 features using OneHotEncoder. The 'class' column is dropped from the DataFrame and
    the resulting feature matrix is returned along with the binary classes.

    Args:
        df (DataFrame): Pandas DataFrame that contains the dataset to be converted to a binary dataset.
    """

    # Convert the categorical classes into binary 1 or 0 classes using LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(df['class'])
    classes = lb.transform(df['class'])

    # Convert the categorical features into binary 1 or 0 features using OneHotEncoder
    df.drop('class', axis=1, inplace=True)
    one = OneHotEncoder(dtype=int)
    one.fit(df)
    feat_mat = one.transform(df).toarray()

    # Check if classes only have 2 classes. If not, create one versus all for the first class
    if classes.shape[1] >= 2:
        classes = classes[:, 0]
    return feat_mat, classes


def chose_dataset(name):
    """
    This function is used to select and load a dataset of interest from the UCI Machine Learning Repository.
        The function takes a string parameter, 'name', which is used to specify the name of the dataset to be loaded,
        and returns two variables, X and y, which contain the features and labels, respectively, of the selected dataset

    Args:
        name (string): Contain the name of the desirable dataset.
    """
    if name == 'mushrooms':
        # mushrooms Dataset
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data'
        mush_df = pd.read_csv(url, header=None)
        column_names = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                        'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                        'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                        'spore-print-color', 'population', 'habitat']
        mush_df.columns = column_names
        X, y = convert_dataset_to_binary(mush_df)
        y = y.flatten()

    elif name == 'tic-tac-toe':
        # tic-tac-toe Dataset
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data'
        tic_tac_toe_df = pd.read_csv(url, header=None)
        column_names = ['top-left-square', 'top-middle-square', 'top-right-square',
                        'middle-left-square', 'middle-middle-square', 'middle-right-square',
                        'bottom-left-square', 'bottom-middle-square', 'bottom-right-square', 'class']
        tic_tac_toe_df.columns = column_names
        X, y = convert_dataset_to_binary(tic_tac_toe_df)
        y = y.flatten()

    elif name == 'spect':
        # spect Dataset
        url1 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test'
        url2 = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train'
        spect_test_df = pd.read_csv(url1, header=None)
        spect_train_df = pd.read_csv(url2, header=None)
        column_names = ['class', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12',
                        'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20', 'F21', 'F22']
        spect_test_df.columns = column_names
        spect_train_df.columns = column_names
        spect_df = pd.concat([spect_test_df, spect_train_df], axis=0)
        X = spect_df.drop('class', axis=1).values
        y = spect_df['class'].values

    elif name == 'monks':
        # monks Dataset
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/monks-problems/monks-1.test'
        monks_df = pd.read_csv(url, header=None, sep=r'\s+')
        column_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'ID']
        monks_df.columns = column_names
        monks_df = monks_df.drop('ID', axis=1)
        X, y = convert_dataset_to_binary(monks_df)
        y = y.flatten()

    elif name == 'car':
        # car Dataset
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data'
        car_df = pd.read_csv(url, header=None)
        column_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
        car_df.columns = column_names
        X, y = convert_dataset_to_binary(car_df)

    else:
        print("Error:" + name + "is not define in the provided database.")
        raise ValueError("Invalid value for 'name'.")

    return X, y
