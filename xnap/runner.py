import xnap.config as config
import xnap.utils as utils
from xnap.nap.preprocessor import Preprocessor
import xnap.nap.tester as test
import xnap.nap.trainer as train
import xnap.exp.explainer as exp


if __name__ == '__main__':

    args = config.load()
    if args.seed:
        utils.set_seed(args)
    measures = utils.measures
    if args.mode == 0 or args.mode == 2:
        utils.clear_measurement_file(args)
    preprocessor = Preprocessor()
    event_log = preprocessor.get_event_log(args)

    # split validation
    train_indices, test_indices = preprocessor.get_indices_split_validation(args, event_log)

    if args.mode == 0:

        train.train(args, preprocessor, event_log, train_indices, measures)
        test.test(args, preprocessor, event_log, test_indices, measures)

        measures = utils.calculate_measures(args, measures)
        utils.print_measures(args, measures)
        utils.write_measures(args, measures)

    elif args.mode == 1:

        trace = preprocessor.get_random_case(event_log, args.rand_lower_bound, args.rand_upper_bound)
        exp.calc_and_plot_relevance_scores_instance(event_log, trace, args, preprocessor, train_indices)

    elif args.mode == 2:

        output_exp = utils.measures
        manipulated_prefixes = exp.get_manipulated_prefixes_from_relevance(args, preprocessor, event_log,
                                                                           train_indices, test_indices, output_exp)
        test.test_manipulated_prefixes(args, preprocessor, event_log, manipulated_prefixes, test_indices)

        output_exp = utils.calculate_measures(args, output_exp)
        utils.print_measures(args, output_exp)
        utils.write_measures(args, output_exp)
