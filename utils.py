

def split_test_set(df, frac=0.3, random=True):
    """
    Split DataFrame to train set and test set
    :param df:
    :param frac:
    :param random:
    :return:
    """
    test_size = int(len(df) * min(frac, 1))
    if random:
        df = df.sample(frac=1).reset_index(drop=True)
    return df[test_size:].reset_index(drop=True), df[:test_size].reset_index(drop=True)
