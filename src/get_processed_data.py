import numpy as np
import pandas as pd
import os

def read_data():
    # Set data paths for raw_data
    RAW_DATA_FP = os.path.join(os.path.pardir, 'data', 'raw')
    TRAIN_FP = os.path.join(RAW_DATA_FP, 'train.csv')
    TEST_FP = os.path.join(RAW_DATA_FP, 'test.csv')
    # read data into dataframes
    train_df = pd.read_csv(TRAIN_FP, index_col='PassengerId')
    test_df = pd.read_csv(TEST_FP, index_col='PassengerId')
    # concatenate datasets into one
    df = pd.concat([train_df, test_df])
    return df

def process_data(df):
    # using method chaining concept
    return (df
            # Create title feature
            .assign(Title = lambda row: row['Name'].apply(get_title))
            # work with missing values
            .pipe(fill_missing_values)
            # create fare bin feature
            .assign(Fare_Bin = lambda x: pd.qcut(x['Fare'], 4, labels=['very_low','low','high','very_high']))
            .assign(Age_State = lambda x: np.where(
                x['Age'] <= 12, 'Child', 
                np.where(x['Age'] < 20, 'Teenager', 
                    np.where(x['Age'] < 60, 'Adult', 
                        'Senior')
                )
            ))
            .assign(Family_Size = lambda x: x['Parch'] + x['SibSp'] + 1)
            .assign(Is_Mother = lambda x: np.where(
                (x['Sex'] == 'female') & 
                (x['Parch'] > 0) & 
                (x['Age'] > 20) & 
                (x['Title'] != 'Miss'), 1, 0))
            .assign(Cabin = lambda x: np.where(x['Cabin'] == 'T', np.nan, x['Cabin']))
            .assign(Deck = lambda x: x['Cabin'].apply(lambda row: get_Deck(row)))
            .assign(Is_Male = lambda row: np.where(row['Sex'] == 'male', 1, 0))
            .assign(Is_Female = lambda row: np.where(row['Sex'] == 'female', 1, 0))
            .pipe(pd.get_dummies, columns=['Deck', 'Pclass', 'Title', 'Fare_Bin', 'Embarked', 'Age_State'], dtype=int)
            .drop(['Cabin', 'Name', 'Ticket', 'Parch', 'SibSp', 'Sex'], axis=1)
            .pipe(reorder_columns)
        )

# function to extract title from name
def get_title(name):
    # add title_group to bin titles for age median
    title_group = {
        'mr': 'Mr',
        'miss': 'Miss',
        'mrs': 'Mrs',
        'master': 'Master',
        'rev': 'Sir',
        'dr': 'Officer',
        'col': 'Officer',
        'mlle': 'Miss',
        'major': 'Officer',
        'ms': 'Mrs',
        'lady': 'Lady',
        'sir': 'Sir',
        'mme': 'Mrs',
        'don': 'Sir',
        'capt': 'Officer',
        'the countess': 'Lady',
        'jonkheer': 'Sir',
        'dona': 'Lady'
    }
    first_name_with_title = name.split(',')[1]  # extract text after comma
    title = first_name_with_title.split('.')[0]  # extract text before period
    title = title.strip().lower()  # strip whitespace and change to lower case
    return title_group[title]

def get_Deck(cabin):
    # assign NaN Cabin Values to Deck placeholder 'Z'
    return np.where(pd.notnull(cabin), str(cabin)[0].upper(), 'Z')

def fill_missing_values(df):
    # Embarked
    df['Embarked'] = df['Embarked'].fillna('C')
    # Fare
    median_fare = df[(df['Pclass'] == 3) & (df['Embarked'] == 'S')]['Fare'].median()
    df['Fare'] = df['Fare'].fillna(median_fare)
    # Age
    title_age_median = df.groupby('Title')['Age'].transform('median')
    df['Age'] = df['Age'].fillna(title_age_median)
    return df

def reorder_columns(df):
    # ensure Survived column is in Col_Index 0
    df.insert(0, 'Survived', df.pop('Survived'))
    return df

def write_data(df):
    # save dataframe
    PROCESSED_DATA_FP = os.path.join(os.path.pardir, 'data', 'processed')
    TRAIN_FN = os.path.join(PROCESSED_DATA_FP, 'train.csv')
    TEST_FN = os.path.join(PROCESSED_DATA_FP, 'test.csv')
    proc_train = df.loc[~df['Survived'].isna()]
    proc_test = df.loc[df['Survived'].isna()]
    # drop 'Survived' column
    proc_test = proc_test.drop(columns='Survived')
    # output data to processed folder
    proc_train.to_csv(TRAIN_FN)
    proc_test.to_csv(TEST_FN)

if __name__ == '__main__':
    df = read_data()
    df = process_data(df)
    write_data(df)
