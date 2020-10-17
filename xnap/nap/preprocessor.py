import numpy
import pandas
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.log import Event
import xnap.utils as utils
from sklearn.model_selection import KFold, ShuffleSplit
import category_encoders


def get_attribute_data_type(attribute_column):
    """ Returns the data type of the passed attribute column 'num' for numerical and 'cat' for categorial """

    column_type = str(attribute_column.dtype)

    if column_type.startswith('float'):
        attribute_type = 'num'
    else:
        attribute_type = 'cat'

    return attribute_type

class Preprocessor(object):

    iteration_cross_validation = 0
    activity = {}
    context = {}
    unique_events_map_to_id = None
    unique_event_ids_map_to_name = None

    def __init__(self, args):
        self.activity = {
            'end_char': '!',
            'chars_to_labels': {},
            'labels_to_chars': {},
            'event_ids_to_one_hot': {},
            'one_hot_to_event_ids':{},
            'label_length': 0,
        }
        self.context = {
            'attributes': [],
            'encoding_lengths': []
        }

    def get_event_log(self, args):
        """ Constructs an event log from a csv file using PM4PY """

        #load data with pandas and encode
        df = pandas.read_csv(args.data_dir + args.data_set, sep=';')
        df_enc = self.encode_data(args, df)

        parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: args.case_id_key}
        event_log = log_converter.apply(df_enc, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
        event_log = self.add_end_event_to_event_log_cases(args, event_log)

        self.get_context_attributes(df)
        return event_log

    def get_context_attributes(self, df=None):
        """ Retrieves names of context attributes """

        if df is not None:
            attributes = []
            column_names = df.columns
            for i in range(len(column_names)):
                if i > 2:
                    attributes.append(column_names[i])
            self.context['attributes'] = attributes
        else:
            return self.context['attributes']

    def encode_context_attribute(self, args, df, column_name):
        """ Encodes values of a context attribute for all events in an event log """

        data_type = get_attribute_data_type(df[column_name])
        encoding_mode = self.get_encoding_mode(args, data_type)

        encoding_columns = self.encode_column(args, df, column_name, encoding_mode)

        if isinstance(encoding_columns, pandas.DataFrame):
            self.set_length_of_context_encoding(len(encoding_columns.columns))
        elif isinstance(encoding_columns, pandas.Series):
            self.set_length_of_context_encoding(1)

        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)

        return df[column_name]

    def add_end_event_to_event_log_cases(self, args, event_log):
        """ Adds an end event at the end of each case in an event log """

        end_label = self.char_to_label(self.get_end_char())
        end_event = Event()
        end_event[args.activity_key] = list(end_label)

        for case in event_log:
            case._list.append(end_event)

        return event_log

    def encode_data(self, args, df):
        """ Encodes an event log represented by a dataframe """

        utils.llprint('Encode data ... \n')

        encoded_df = pandas.DataFrame(df.iloc[:, 0])

        for column_name in df:
            column_index = df.columns.get_loc(column_name)

            if column_index == 0 or column_index == 2:
                # no encoding of case id and timestamp
                encoded_df[column_name] = df[column_name]
            else:
                if column_index == 1:
                    #Created a mapping in order to also use raw avitivity names in csv files in contrast to naptf2.0
                    self.unique_events_map_to_id = self.map_event_name_to_event_id(df[column_name])
                    self.unique_event_ids_map_to_name = self.map_event_id_to_event_name()
                    #transform event log activities to event ids TODO Is this necessary since the log could already be converted?
                    #TODO maybe this step can be skipped but the mapping of raw names and one hot encodings could be saved instead!
                    for index, row in df.iterrows():
                        event_name = df.iloc[index, column_index]
                        event_id = self.unique_events_map_to_id[event_name]
                        df.iloc[index, column_index] = event_id

                    #TODO encode activities really necessarry to convert to chars? why not leave event id as int?
                    encoded_column = self.encode_activities(args, df, column_name)
                    #save Mapping of one hot activities to ids
                    self.save_mapping_one_hot_activities_to_id(df[column_name], encoded_column)
                if column_index > 2:
                    encoded_column = self.encode_context_attribute(args, df, column_name)
                encoded_df = encoded_df.join(encoded_column)

        return encoded_df

    def get_encoding_mode(self, args, data_type):
        """ Returns the encoding method to be used for a given data type """

        if data_type == 'num':
            mode = args.encoding_num
        elif data_type == 'cat':
            mode = args.encoding_cat

        return mode

    def encode_activities(self, args, df, column_name):
        """ Encodes activities for all events in an event log """

        encoding_mode = args.encoding_cat

        if encoding_mode == 'hash':
            encoding_mode = 'onehot'

        df = self.add_end_char_to_activity_column(df, column_name)
        encoding_columns = self.encode_column(args, df, column_name, encoding_mode)
        self.save_mapping_of_activities(df[column_name], encoding_columns)
        if isinstance(encoding_columns, pandas.DataFrame):
            self.set_length_of_activity_encoding(len(encoding_columns.columns))
        elif isinstance(encoding_columns, pandas.Series):
            self.set_length_of_activity_encoding(1)

        df = self.transform_encoded_attribute_columns_to_single_column(encoding_columns, df, column_name)
        df = self.remove_end_char_from_activity_column(df)

        return df[column_name]

    #TODO check if this step is unnecesseary because it only needs to be done since naptf2.0 has a converted event log and xnap2.0 has a raw event log
    def map_event_name_to_event_id(self, df_column):
        unique_events = []
        for event in df_column:
            if event not in unique_events:
                unique_events.append(event)

        unique_events_map_to_id = {}
        for i, c in enumerate(unique_events):
            unique_events_map_to_id[c] = i

        return unique_events_map_to_id

    def map_event_id_to_event_name(self):
        unique_ids_map_to_event_names = {}
        for event_name in self.unique_events_map_to_id:
            unique_ids_map_to_event_names[self.unique_events_map_to_id[event_name]] = event_name

        return unique_ids_map_to_event_names

    def add_end_char_to_activity_column(self, df, column_name):
        """ Adds single row to dataframe containing the end char (activity) representing an artificial end event """

        df_columns = df.columns
        new_row = []
        for column in df_columns:
            if column == column_name:
                new_row.append(self.get_end_char())
            else:
                new_row.append(0)

        row_df = pandas.DataFrame([new_row], columns=df.columns)
        df = df.append(row_df, ignore_index=True)

        return df

    def remove_end_char_from_activity_column(self, df):
        """ Removes last row of dataframe containing the end char (activity) representing an artificial end event """

        orig_rows = df.drop(len(df) - 1)
        return orig_rows

    def encode_column(self, args, df, column_name, mode):
        """ Returns columns containing encoded values for a given attribute column """

        if mode == 'min_max_norm':
            encoding_columns = self.apply_min_max_normalization(df, column_name)

        elif mode == 'onehot':
            encoding_columns = self.apply_one_hot_encoding(df, column_name)

        elif mode == 'hash':
            encoding_columns = self.apply_hash_encoding(args, df, column_name)
        else:
            # no encoding
            encoding_columns = df[column_name]

        return encoding_columns

    def save_mapping_of_activities(self, column, encoded_column):
        """ Creates a mapping for activities (chars + labels (= encoded activity)) """

        activities_chars = self.convert_activities_to_chars(column.values.tolist())

        encoded_column_tuples = []
        for entry in encoded_column.values.tolist():
            if type(entry) != list:
                encoded_column_tuples.append((entry,))
            else:
                encoded_column_tuples.append(tuple(entry))

        tuple_all_rows = list(zip(activities_chars, encoded_column_tuples))

        tuple_unique_rows = []
        for tuple_row in tuple_all_rows:
            if tuple_row not in tuple_unique_rows:
                tuple_unique_rows.append(tuple_row)

        self.activity['chars_to_labels'] = dict(tuple_unique_rows)
        self.activity['labels_to_chars'] = dict([(t[1], t[0]) for t in tuple_unique_rows])

        return

    def save_mapping_one_hot_activities_to_id(self, activity_column, encoded_column):
        """ Creates a mapping for activities (chars + labels (= encoded activity)) """

        activity_ids = activity_column.values.tolist()

        encoded_column_tuples = []
        for entry in encoded_column.values.tolist():
            if type(entry) != list:
                encoded_column_tuples.append((entry,))
            else:
                encoded_column_tuples.append(tuple(entry))

        tuple_all_rows = list(zip(activity_ids, encoded_column_tuples))

        tuple_unique_rows = []
        for tuple_row in tuple_all_rows:
            if tuple_row not in tuple_unique_rows:
                tuple_unique_rows.append(tuple_row)

        self.activity['event_ids_to_one_hot'] = dict(tuple_unique_rows)
        self.activity['one_hot_to_event_ids'] = dict([(t[1], t[0]) for t in tuple_unique_rows])

        return

    def transform_encoded_attribute_columns_to_single_column(self, encoded_columns, df, column_name):
        """ Transforms multiple columns (repr. encoded attribute) to a single column in a dataframe """

        encoded_values_list = encoded_columns.values.tolist()
        df[column_name] = encoded_values_list

        return df

    def convert_activity_to_char(self, activity):
        """ Convert initial activities' representations to chars """

        if activity != self.get_end_char():
            # 161 below is ascii offset
            activity = chr(int(activity) + 161) #TODO

        return activity

    def convert_activities_to_chars(self, activities):
        """ Convert initial activity's representation to char """

        chars = []
        for activity in activities:
            chars.append(self.convert_activity_to_char(activity))

        return chars

    def apply_min_max_normalization(self, df, column_name):
        """ Normalizes a dataframe column with min max normalization """

        column = df[column_name].fillna(df[column_name].mean())
        encoded_column = (column - column.min()) / (column.max() - column.min())

        return encoded_column

    def apply_one_hot_encoding(self, df, column_name):
        """ Encodes a dataframe column with one hot encoding """

        onehot_encoder = category_encoders.OneHotEncoder(cols=[column_name])
        encoded_df = onehot_encoder.fit_transform(df)

        encoded_column = encoded_df[encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith("%s_" % column_name)]]

        return encoded_column

    def apply_hash_encoding(self, args, df, column_name):
        """ Encodes a dataframe column with hash encoding """

        hash_encoder = category_encoders.HashingEncoder(cols=[column_name],
                                                        n_components=args.num_hash_output,
                                                        hash_method='md5')
        encoded_df = hash_encoder.fit_transform(df)
        encoded_column = encoded_df[encoded_df.columns[pandas.Series(encoded_df.columns).str.startswith('col_')]]

        new_column_names = []
        for number in range(len(encoded_column.columns)):
            new_column_names.append(column_name + "_%d" % number)

        encoded_column = encoded_column.rename(columns=dict(zip(encoded_column.columns.tolist(), new_column_names)))

        return encoded_column

    def set_length_of_activity_encoding(self, num_columns):
        """ Save number of columns representing an encoded activity """
        self.activity['label_length'] = num_columns

    def set_length_of_context_encoding(self, num_columns):
        """ Save number of columns representing an encoded context attribute """
        self.context['encoding_lengths'].append(num_columns)

    def get_length_of_activity_label(self):
        """ Returns number of columns representing an encoded activity """
        return self.activity['label_length']

    def get_lengths_of_context_encoding(self):
        """ Returns number of columns representing an encoded context attribute """
        return self.context['encoding_lengths']

    def get_activity_labels(self):
        """ Returns labels representing encoded activities of an event log """
        return list(self.activity['labels_to_chars'].keys())

    def get_activity_chars(self):
        """ Returns chars representing activities of an event log """
        return list(self.activity['chars_to_labels'].keys())

    def char_to_label(self, char):
        """ Maps a char to a label (= encoded activity) """
        return self.activity['chars_to_labels'][char]

    def label_to_char(self, label):
        """ Maps a label (= encoded activity) to a char """
        return self.activity['labels_to_chars'][label]

    def get_event_id_from_one_hot(self, one_hot_encoding):
        """
        :param one_hot_encoding:
        :return: event id
        """
        return self.activity['one_hot_to_event_ids'][tuple(one_hot_encoding)]

    def get_end_char(self):
        """ Returns a char symbolizing the end (activity) of a case """
        return self.activity['end_char']

    def get_num_activities(self):
        """ Returns the number of activities (incl. artificial end activity) occurring in the event log """

        return len(self.get_activity_labels())

    def context_exists(self):
        """ Checks whether context attributes exist """

        return len(self.get_context_attributes(df=None)) > 0

    def get_num_features(self):
        """ Returns the number of features used to train and test the model """

        num_features = 0
        num_features += self.get_length_of_activity_label()
        for len in self.get_lengths_of_context_encoding():
            num_features += len

        return num_features

    def get_max_case_length(self, event_log):
        """ Returns the length of the longest case in an event log """

        max_case_length = 0
        for case in event_log:
            if case.__len__() > max_case_length:
                max_case_length = case.__len__()

        return max_case_length

    def get_indices_k_fold_validation(self, args, event_log):
        """ Produces indices for each fold of a k-fold cross-validation """
        kFold = KFold(n_splits=args.num_folds, random_state=0, shuffle=False)

        train_index_per_fold = []
        test_index_per_fold = []

        for train_indices, test_indices in kFold.split(event_log):
            train_index_per_fold.append(train_indices)
            test_index_per_fold.append(test_indices)

        return train_index_per_fold, test_index_per_fold

    def get_indices_split_validation(self, args, event_log):
        """ Produces indices for training and test set of a split-validation """

        shuffle_split = ShuffleSplit(n_splits=1, test_size=args.val_split, random_state=0)

        train_index_per_fold = []
        test_index_per_fold = []

        for train_indices, test_indices in shuffle_split.split(event_log):
            train_index_per_fold.append(train_indices) ##TODO there is actually no fold (we do have split validation), rename this also
            test_index_per_fold.append(test_indices)

        return train_index_per_fold[0], test_index_per_fold[0]

    def get_cases_of_fold(self, event_log, index_per_fold):
        """ Retrieves cases of a fold """

        cases_of_fold = []

        for index in index_per_fold[self.iteration_cross_validation]:
            cases_of_fold.append(event_log[index])

        return cases_of_fold

    def get_subsequences_of_cases(self, cases):
        """ Creates subsequences of cases representing increasing prefix sizes """

        subseq = []

        for case in cases:
            for idx_event in range(0, len(case._list)):
                if idx_event == 0:
                    continue
                else:
                    subseq.append(case._list[0:idx_event])

        return subseq

    def get_next_events_of_subsequences_of_cases(self, cases):
        """ Retrieves next events (= suffix) following a subsequence of a case (= prefix) """

        next_events = []

        for case in cases:
            for idx_event in range(0, len(case._list)):
                if idx_event == 0:
                    continue
                else:
                    next_events.append(case._list[idx_event])

        return next_events

    def get_features_tensor(self, args, mode, event_log, subseq_cases):
        """ Produces a vector-oriented representation of feature data as a 3-dimensional tensor """

        num_features = self.get_num_features()
        max_case_length = self.get_max_case_length(event_log)

        if mode == 'train':
            features_tensor = numpy.zeros((len(subseq_cases),
                                    max_case_length,
                                    num_features), dtype=numpy.float64)
        else:
            features_tensor = numpy.zeros((1,
                                    max_case_length,
                                    num_features), dtype=numpy.float32)

        for idx_subseq, subseq in enumerate(subseq_cases):
            left_pad = max_case_length - len(subseq)

            for timestep, event in enumerate(subseq):

                # activity
                activity_values = event.get(args.activity_key)
                for idx, val in enumerate(activity_values):
                    features_tensor[idx_subseq, timestep + left_pad, idx] = val

                # context
                if self.context_exists():
                    start_idx = 0

                    for attribute_idx, attribute_key in enumerate(self.context['attributes']):
                        attribute_values = event.get(attribute_key)

                        if not isinstance(attribute_values, list):
                            features_tensor[idx_subseq, timestep + left_pad, start_idx + self.get_length_of_activity_label()] = attribute_values
                            start_idx += 1
                        else:
                            for idx, val in enumerate(attribute_values, start=start_idx):
                                features_tensor[idx_subseq, timestep + left_pad, idx + self.get_length_of_activity_label()] = val
                            start_idx += len(attribute_values)

        return features_tensor


    def get_labels_tensor(self, args, cases_of_fold):
        """ Produces a vector-oriented representation of labels as a 2-dimensional tensor """

        subseq_cases = self.get_subsequences_of_cases(cases_of_fold)
        next_events = self.get_next_events_of_subsequences_of_cases(cases_of_fold)
        num_event_labels = self.get_num_activities()
        activity_labels = self.get_activity_labels()

        labels_tensor = numpy.zeros((len(subseq_cases), num_event_labels), dtype=numpy.float64)

        for idx_subseq, subseq in enumerate(subseq_cases):
            for label_tuple in activity_labels:
                if list(label_tuple) == next_events[idx_subseq][args.activity_key]:
                    labels_tensor[idx_subseq, activity_labels.index(label_tuple)] = 1.0
                else:
                    labels_tensor[idx_subseq, activity_labels.index(label_tuple)] = 0.0

        return labels_tensor

    def get_predicted_label(self, predictions):
        """ Returns label of a predicted activity """

        labels = self.get_activity_labels()
        max_prediction = 0
        event_index = 0

        for prediction in predictions:
            if prediction >= max_prediction:
                max_prediction = prediction
                label = labels[event_index]
            event_index += 1

        return label



    def convert_process_instance_list_id_to_name(self, process_instances):
        """
        takes a 2 dimensional input (meaning a list of process instances with their event ids and
        converts it to event names
        """
        process_instances_converted = []
        for process_instance in process_instances:
            process_instance_converted = []
            for event in process_instance:
                process_instance_converted.append(self.unique_event_ids_map_to_name[event])
            process_instances_converted.append(process_instance_converted)

        return process_instances_converted

    def get_random_process_instance(self, event_log, lower_bound, upper_bound):
        """
        Selects a random process instance from the complete event log.
        :param args:
        :param lower_bound:
        :param upper_bound:
        :return: process instance.
        """

        while True:
            x = len(event_log)
            rand = numpy.random.randint(x)
            size = len(event_log[rand])

            if lower_bound <= size <= upper_bound:
                break

        return event_log[rand]

    def get_cropped_instance_label(self, prefix_size, process_instance):
        """
        Crops the next activity label out of a single process instance.
        :param prefix_size:
        :param process_instance:
        :return:
        """

        if prefix_size == len(process_instance) - 1:
            #end marker
            return self.get_end_char()
        else:
            # label of next act
            return process_instance[prefix_size]

    def get_event_type_from_event_id(self, event_id):
        """
        :param event_id:
        :return: event type as a raw name
        """
        if event_id == len(self.unique_event_ids_map_to_name):
            return self.get_end_char()
        else:
            return self.unique_event_ids_map_to_name[event_id]

    def get_event_id_from_event_name(self, event_name):
        """
        :param event_name:
        :return: event id
        """
        if event_name == self.get_end_char():
            return len(self.unique_events_map_to_id)
        else:
            return self.unique_events_map_to_id[event_name]














