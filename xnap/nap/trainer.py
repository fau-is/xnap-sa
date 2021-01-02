from __future__ import print_function, division
import tensorflow as tf
from datetime import datetime
import xnap.utils as utils
from sklearn.ensemble import RandomForestClassifier
import joblib


def train(args, preprocessor, event_log, train_indices, output):

    train_cases = preprocessor.get_subset_cases(args, event_log, train_indices)
    train_subseq_cases = preprocessor.get_subsequences_of_cases(train_cases)

    features_tensor = preprocessor.get_features_tensor(args, event_log, train_subseq_cases)
    labels_tensor = preprocessor.get_labels_tensor(args, train_cases)

    print('Create machine learning model ... \n')
    if args.classifier == "DNN":
        # Deep Neural Network
        train_dnn(args, preprocessor, event_log, features_tensor, labels_tensor, output)

    if args.classifier == "RF":
        # Random Forest
        train_random_forest(args, preprocessor, features_tensor, labels_tensor, output)


def train_dnn(args, preprocessor, event_log, features_tensor, labels_tensor, output):

    max_case_len = preprocessor.get_max_case_length(event_log)
    num_features = preprocessor.get_num_features()
    num_activities = preprocessor.get_num_activities()

    # if args.dnn_architecture == 0:
    # Bidirectional LSTM

    # input layer
    main_input = tf.keras.layers.Input(shape=(max_case_len, num_features), name='main_input')

    # hidden layer
    b1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,
                                                            activation="tanh",
                                                            kernel_initializer='glorot_uniform',
                                                            return_sequences=False,
                                                            dropout=0.2))(main_input)

    # output layer
    act_output = tf.keras.layers.Dense(num_activities,
                                       activation='softmax',
                                       name='act_output',
                                       kernel_initializer='glorot_uniform')(b1)

    model = tf.keras.models.Model(inputs=[main_input], outputs=[act_output])

    optimizer = tf.keras.optimizers.Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8,
                                          schedule_decay=0.004, clipvalue=3)

    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=optimizer)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(utils.get_model_dir(args, preprocessor),
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
    model.fit(features_tensor, {'act_output': labels_tensor},
              validation_split=args.val_split,
              verbose=1,
              callbacks=[early_stopping, model_checkpoint, lr_reducer],
              batch_size=args.batch_size_train,
              epochs=args.dnn_num_epochs)
    training_time = datetime.now() - start_training_time
    output["training_time_seconds"].append(training_time.total_seconds())


def train_random_forest(args, preprocessor, features_tensor_flattened, labels_tensor, output):

    model = RandomForestClassifier(n_jobs=-1,                           # use all processors
                                   random_state=0,
                                   n_estimators=100,                    # default value
                                   criterion="gini",                    # default value
                                   max_depth=None,                      # default value
                                   min_samples_split=2,                 # default value
                                   min_samples_leaf=1,                  # default value
                                   min_weight_fraction_leaf=0.0,        # default value
                                   max_features="auto",                 # default value
                                   max_leaf_nodes=None,                 # default value
                                   min_impurity_decrease=0.0,           # default value
                                   bootstrap=True,                      # default value
                                   oob_score=False,                     # default value
                                   warm_start=False,                    # default value
                                   class_weight=None)                   # default value

    start_training_time = datetime.now()
    model.fit(features_tensor_flattened, labels_tensor)
    training_time = datetime.now() - start_training_time
    output["training_time_seconds"].append(training_time.total_seconds())

    joblib.dump(model, utils.get_model_dir(args, preprocessor))
