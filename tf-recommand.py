import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse, code, utils
import models.Recommendation as model_recomm


parser = argparse.ArgumentParser()
parser.add_argument('--num_steps', default=1000, type=int, help='number of trannig steps')

def get_ratings(file='data/ratings_4055.csv'):
    df = pd.read_csv(file, header=None, names=['item', 'user', 'rating', 'date'])
    return df

def evaluate():
    pass

def train_item_based():
    pass

def train_mf(df, num_steps=10000):
    if 'real_rating' in df.columns:
        train_df = df[df['rating'] > 0]
    else:
        train_df = df

    def _set_index(df, keys):
        """
        Set index of given key (after drop duplicates)
        :param df:
        :param keys:
        :return:
        """
        for key in keys:
            unique_df = df.drop_duplicates([key]).reset_index()

            df.insert(len(df.columns), key + '_index', np.zeros(len(df), dtype=np.int32))

            for i, row in unique_df.iterrows():
                origin_index = list(df[df[key] == row[key]].index)
                df.loc[origin_index, key + '_index'] = i

    _set_index(train_df, ['item', 'user'])

    # code.interact(local=locals())

    total_item, total_user = len(train_df.drop_duplicates(['item'])), len(train_df.drop_duplicates(['user']))

    estimator = model_recomm.MatrixFactorizationEstimator(
        R_shape=(total_item, total_user),
        num_features=10,
        model_dir="models/mf/"
    )

    input_fn = tf.estimator.inputs.pandas_input_fn(
        x=train_df, y=train_df['rating'], batch_size=2, num_epochs=None, shuffle=True
    )
    estimator.train(input_fn=input_fn, steps=num_steps)

    return estimator

def hide_rating(data, frac=0.3):
    """
    Hide some ratings for evaluate

    :param data:
    :param frac: how many ratings should be hidden
    :return:
        DataFrame with additional column `real_rating` for origin ratings,
        and some `rating` assigned by 0
    """
    df = data
    if 'rating' not in df:
        raise model_recomm.ColumnNotFoundError('rating')

    if 'real_rating' in df:
        df['real_rating'] = np.array(df['rating'])
    else:
        df.insert(len(df.columns), 'real_rating', np.array(df['rating']))

    if frac > 0:
        # hide top `frac` ratings for every user
        def _hide(df, frac):
            for i, user in df['user'].drop_duplicates().items():
                user_ratings = df[df['user'] == user]
                hide_len = int(len(user_ratings) * 0.3)
                if hide_len == 0:
                    continue
                index = user_ratings.index[:hide_len]
                df.loc[index, 'rating'] = 0
            return df

        df = _hide(df, frac)

    df = df.sample(frac=1).reset_index(drop=True)

    return df

def main(argv):
    args = parser.parse_args(argv[1:])

    # train_set, test_set = utils.split_test_set(get_ratings(), frac=0.3, random=True)
    # data = get_ratings('data/ratings_test.csv')
    data = get_ratings()

    # data = hide_rating(data, frac=0.3)
    k, pred_user, pred_item = 10, '140756691', ''

    # item_based = model_recomm.ItemBased(data)
    # recommend_list = item_based.top_k(user=pred_user, k=k, sim_fn=item_based.sim_cos)

    model = train_mf(data, num_steps=args.num_steps)
    recommend_list = model.top_k(user=pred_user, k=k)

    # 140756691

    print(recommend_list)

    # print(item_based.predict('用户1', '图书2', item_based.sim_cos))
    # code.interact(local=locals())

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
