from __future__ import print_function, division
import tensorflow as tf
from datetime import datetime
import xnap.nap.preprocessing.utilts as utils


def train(args, preprocessor, event_log):

    # todo: rename variables for a generic way independently of split/cross validation (especially cases of fold)
    if args.cross_validation:
        # cases = preprocessor.get_cases_of_fold(event_log, [all_indices])
        raise ValueError('cross_validation not yet implemented in XNAP2.0')
    else:
        # get preprocessed proces instances for split validation
        train_indices, test_indices = utils.get_indices_split_validation(args, event_log)

    train_cases = []
    for idx in train_indices:  # 0 because of no cross validation
        train_cases.append(event_log[idx])

    # similar to nap2.0tf hpo l 76 ff
    train_subseq_cases = preprocessor.get_subsequences_of_cases(train_cases)

    feature_tensor_x_train = preprocessor.get_features_tensor(args, event_log, train_subseq_cases)
    label_tensor_y_train = preprocessor.get_labels_tensor(args, train_cases)

    max_length_process_instance = preprocessor.get_max_case_length(event_log)
    num_features = preprocessor.get_num_features()
    num_event_ids = preprocessor.get_num_activities()

    print('Create machine learning model ... \n')
    if args.dnn_architecture == 0:
        # input layer
        main_input = tf.keras.layers.Input(shape=(max_length_process_instance, num_features), name='main_input')

        # hidden layer
        b1 = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(100,
                                 activation="tanh",
                                 kernel_initializer='glorot_uniform',
                                 return_sequences=False,
                                 dropout=0.2))(
            main_input)

        # output layer
        act_output = tf.keras.layers.Dense(num_event_ids,
                                           activation='softmax',
                                           name='act_output',
                                           kernel_initializer='glorot_uniform')(b1)

    model = tf.keras.models.Model(inputs=[main_input], outputs=[act_output])

    optimizer = tf.keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                          schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('%sca_%s_%s_%s.h5' % (
        args.model_dir,
        args.task,
        args.data_set[0:len(args.data_set) - 4],
        preprocessor.iteration_cross_validation),
                                                          monitor='val_loss',
                                                          verbose=0,
                                                          save_best_only=True,
                                                          save_weights_only=False,
                                                          mode='auto')

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=10,
                                                      verbose=0,
                                                      mode='auto',
                                                      min_delta=0.0001,
                                                      cooldown=0,
                                                      min_lr=0)
    model.summary()
    start_training_time = datetime.now()

    model.fit(feature_tensor_x_train, {'act_output': label_tensor_y_train},
              validation_split=args.val_split,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)

    training_time = datetime.now() - start_training_time

    return training_time.total_seconds()