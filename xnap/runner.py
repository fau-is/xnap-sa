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
    output = utils.load_output()
    utils.clear_measurement_file(args)
    preprocessor = Preprocessor()
    event_log = preprocessor.get_event_log(args)

    if args.cross_validation:
        # train_indices, test_indices = preprocessor.get_indices_k_fold_validation(args, event_log)
        raise ValueError('cross_validation not yet implemented in XNAP2.0')
    if not args.cross_validation:
        train_indices, test_indices = preprocessor.get_indices_split_validation(args, event_log)

    if args.mode == 0:

        train.train(args, preprocessor, event_log, train_indices, output)
        test.test(args, preprocessor, event_log, test_indices, output)

        output = utils.get_output(args, preprocessor, output)
        utils.print_output(args, output, -1)
        utils.write_output(args, output, -1)

    elif args.mode == 1:

        trace = preprocessor.get_random_case(event_log, args.rand_lower_bound, args.rand_upper_bound)
        exp.calc_and_plot_relevance_scores_instance(event_log, trace, args, preprocessor)

    else:
        output_exp = utils.load_output()
        manipulated_prefixes = exp.get_manipulated_prefixes_from_relevance(args, preprocessor, event_log, test_indices,
                                                                           output_exp)
        test.test_manipulated_prefixes(args, preprocessor, event_log, manipulated_prefixes, test_indices)

        output_exp = utils.get_output(args, preprocessor, output_exp)
        utils.print_output(args, output_exp, -1)
        utils.write_output(args, output_exp, -1)
