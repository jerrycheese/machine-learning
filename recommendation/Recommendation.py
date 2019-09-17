from tensorflow.estimator import Estimator
import tensorflow as tf
import numpy as np
import pandas as pd
import code

class ColumnNotFoundError(Exception):
    """Raised when adding API names to symbol that already has API names."""
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr('Can not found column `{}`'.format(self.value))


class ItemBased(Estimator):


    def __init__(self, data):
        """

        :param data: DataFrame of data set
        """
        columns = ['rating', 'user', 'item']
        for col in columns:
            if col not in data:
                raise ColumnNotFoundError(col)

        self.data = data
        self.item_cols = {}

    def predict(self, user, item, sim_fn):
        """
        :param user:
        :param item:
        :param sim_fn: the similarity function to use
        :return: prediction of ratings, `None` for unknown rating
        """
        df = self.data

        user_ratings = df[df['user'] == user]
        sum_sim, sum_sim_rating = 0.0, 0.0

        for i, row in user_ratings.iterrows():
            # skip invisible rating
            if 'real_rating' in row and row['rating'] == 0:
                continue

            exist_item = row['item']

            similarity = sim_fn(exist_item, item)

            # print('the {} and {} similarity is {}'.format(predict_book, book_id, similarity))

            sum_sim += similarity

            sum_sim_rating += similarity * row['rating']

        if sum_sim == 0:
            # mean of rating of this item
            item_ratings = df[df['item'] == item]
            if len(item_ratings) > 0:
                return np.mean(item_ratings['rating'])
            # unknown
            return None
        else:
            return sum_sim_rating / sum_sim

    def top_k(self, user, k, sim_fn):
        """
        Get the top K items, sort by predict rating
        :param user:
        :param k: the K
        :param sim_fn: the similarity function to use
        :return:
        """
        df = self.data

        if 'real_rating' in df:
            user_rated = df[(df['user'] == user) & (df['rating'] != 0)]
        else:
            user_rated = df[df['user'] == user]


        unrated_item = set(df['item']) - set(user_rated['item'])

        score_df = pd.DataFrame(columns=['item', 'real_rating' , 'predict_rating'])

        items, real_ratings, predict_ratings = [], [], []

        # make prediction
        for item in unrated_item:
            rating = self.predict(user, item, sim_fn)
            real_rating = None
            if 'real_rating' in df:
                user_item_df = df[(df['user'] == user) & (df['item'] == item)]
                if len(user_item_df) > 0:
                    real_rating = type(rating)(user_item_df['real_rating'])

            items.append(item)
            real_ratings.append(real_rating)
            predict_ratings.append(rating)
            print('Predict item {}, rating = {}'.format(item, rating))

        score_df['item'] = items
        score_df['real_rating'] = real_ratings
        score_df['predict_rating'] = predict_ratings
        score_df = score_df.sort_values(by='predict_rating', ascending=False)
        if k > len(score_df):
            return score_df
        else:
            return score_df[:k]

    def _get_two_item_col(self, item1, item2):
        """
        Get the vector of to items
        :param item1:
        :param item2:
        :return: two vectors(np array, not matrix)
        """
        df = self.data
        # if length of cache greater then 20, del two caches
        # if len(self.item_cols) > 100:
        #     keys = list(self.item_cols.keys())
        #     del self.item_cols[keys[0]]
        #     del self.item_cols[keys[1]]


        items = [item1, item2]
        items_df = []

        for item in items:
            # if item not in self.item_cols:
            if 'real_rating' in df.columns:
                item_df = df[(df['item'] == item) & (df['rating'] != 0)]
            else:
                item_df = df[df['item'] == item]

            items_df.append(item_df)

        item1_df = items_df[0]
        item2_df = items_df[1]

        merge_by_user = pd.merge(item1_df, item2_df, on='user', suffixes=[1, 2])

        return merge_by_user['rating1'], merge_by_user['rating2']

    def sim_eulid(self, item1, item2):
        item_col1, item_col2 = self._get_two_item_col(item1, item2)

        return 1.0 / (1.0 + np.linalg.norm(item_col1 - item_col2))

    def sim_pears(self, item1, item2):
        item_col1, item_col2 = self._get_two_item_col(item1, item2)

        if len(item_col1) < 3:
            return 1.0
        return (1 + np.corrcoef(item_col1, item_col2, rowvar=0)[0][1]) / 2

    def sim_cos(self, item1, item2):
        item_col1, item_col2 = self._get_two_item_col(item1, item2)

        if len(item_col1) == 0 or len(item_col2) == 0:
            return 0

        num = np.matmul(item_col1, item_col2)
        denom = np.linalg.norm(item_col1) * np.linalg.norm(item_col2)

        return 0.5 + num / denom / 2

    def clear_cache(self):
        self.item_cols = {}
        self.predictions = {}


class MatrixFactorizationEstimator(Estimator):

    def __init__(self, R_shape, num_features, model_dir=None, config=None, warm_start_from=None):

        X_init = np.ones((R_shape[0], num_features), np.float64)
        Theta_init = np.ones((R_shape[1], num_features), np.float64)

        if config and 'lmd' in config:
            lmd = config['lmd']
        else:
            lmd = 1

        def _model_fn(features, labels, mode, params):
            Theta = tf.get_variable('Theta', Theta_init.shape, dtype=tf.float64)
            X = tf.get_variable('X', X_init.shape, dtype=tf.float64)

            # bX = tf.map_fn(lambda e: X[e, :], features['item_index'], dtype=tf.float64)
            #
            # bTheta = tf.map_fn(lambda e: Theta[e, :], features['user_index'], dtype=tf.float64)


            pred = tf.map_fn(lambda e:
                             tf.reshape(
                                 tf.matmul(
                                     tf.reshape(X[e['item_index'], :], (1, num_features)),
                                     tf.transpose(tf.reshape(Theta[e['user_index'], :], (1, num_features)))
                                 ),
                                 (1,)
                             ),
                             features, dtype=tf.float64)

            pred = tf.reshape(pred, (tf.shape(pred)[0],))

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = {
                    'rating': pred
                }
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            # Compute loss.
            # loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

            loss = tf.reduce_sum(tf.square(pred - tf.cast(features['rating'], tf.float64))) / 2

            penalty = tf.map_fn(lambda e:
                                    tf.reduce_sum(X[e['item_index'], :]) +
                                    tf.reduce_sum(Theta[e['user_index'], :]),
                                    features, dtype=tf.float64)
            penalty = lmd * tf.reduce_sum(penalty) / 2

            loss += penalty

            # print(loss)

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode, loss=loss, eval_metric_ops=None)

            # Create training op.
            assert mode == tf.estimator.ModeKeys.TRAIN

            optimizer = tf.train.GradientDescentOptimizer(0.001)
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

            # train_op = tf.group(tf.assign_add(tf.train.get_global_step(), 1))
            return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        super(MatrixFactorizationEstimator, self).__init__(
            model_fn=_model_fn,
            model_dir=model_dir,
            config=config,
            warm_start_from=warm_start_from)

    def top_k(self, user, k):
        """ It is hard to make prediction for all book 
        So we predict the rating of first 3 user to first 5 book
        
        """
        data = {'user_index': [], 'item_index': []}
        for ui in range(3):
            for ti in range(5):
                data['user_index'].append(ui)
                data['item_index'].append(ti)

        data = pd.DataFrame(data)

        input_fn = tf.estimator.inputs.pandas_input_fn(
            x=data, shuffle=False, num_epochs=1, batch_size=2
        )

        res = []
        for e in self.predict(input_fn):
            res.append(e)
            # print(e)
        return res

