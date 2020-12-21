from sklearn.model_selection import KFold, ShuffleSplit


def get_indices_split_validation(args, event_log):
    """ Produces indices for training and test set of a split-validation """

    if args.shuffle:

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.val_split, random_state=0)

        train_index_per_fold = []
        test_index_per_fold = []

        for train_indices, test_indices in shuffle_split.split(event_log):
            train_index_per_fold.append(
                train_indices)  # TODO there is actually no fold (we do have split validation), rename this also
            test_index_per_fold.append(test_indices)

        return train_index_per_fold[0], test_index_per_fold[0]

    else:

        indices_ = [index for index in range(0, len(event_log))]
        return indices_[:int(len(indices_) * args.split_rate_test)], \
               indices_[int(len(indices_) * args.split_rate_test):]


def get_indices_k_fold_validation(args, event_log):
    """
    Produces indices for each fold of a k-fold cross-validation
    :param args:
    :param event_log:
    :return:
    """

    kFold = KFold(n_splits=args.num_folds, random_state=args.seed_val, shuffle=args.shuffle)

    train_index_per_fold = []
    test_index_per_fold = []

    for train_indices, test_indices in kFold.split(event_log):
        train_index_per_fold.append(train_indices)
        test_index_per_fold.append(test_indices)

    return train_index_per_fold, test_index_per_fold


def get_test_set(args, event_log):
    """
    Retrieves indices of test set.

    Parameters
    ----------
    args : Namespace
        Settings of the configuration parameters.
    event_log : list of dicts, where single dict represents a case
        The initial event log.

    Returns
    -------
    test_cases : list of dicts, where single dict represents a case
        List of cases.

    """

    _, test_indices = get_indices_split_validation(args, event_log)
    test_cases = []
    for idx in test_indices:
        test_cases.append(event_log[idx])

    return test_cases
