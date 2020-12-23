import xnap.config as config
import xnap.utils as utils
from xnap.nap.preprocessing.preprocessor import Preprocessor as Preprocessor
import xnap.nap.tester as test
import xnap.nap.trainer as train
import xnap.exp.lrp.lrp as lrp
import xnap.exp.lime.lime as lime
import xnap.exp.exp_evaluator as exp_evaluator

if __name__ == '__main__':

    args = config.load()
    if args.seed:
        utils.set_seed(args)
    output = utils.load_output()
    utils.clear_measurement_file(args)
    preprocessor = Preprocessor(args)
    event_log = preprocessor.get_event_log(args)

    if args.mode == 0:

        if args.cross_validation:
            # todo: implement cross_validation
            """
            for iteration_cross_validation in range(0, args.num_folds):
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)
            """
            raise ValueError('cross_validation not yet implemented in XNAP2.0')
        else:
            train.train(args, preprocessor, event_log, output)
            test.test(args, preprocessor, event_log, output)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)

    elif args.mode == 1:

        trace = preprocessor.get_random_process_instance(event_log, args.rand_lower_bound, args.rand_upper_bound)
        if args.xai == "lrp":
            lrp.calc_and_plot_relevance_scores_instance(event_log, trace, args, preprocessor)
        if args.xai == "lime":
            lime.calc_and_plot_relevance_scores_instance(event_log, trace, args, preprocessor)

    else:
        output_exp = utils.load_output()
        manipulated_prefixes = exp_evaluator.get_manipulated_test_prefixes_from_relevance(args, preprocessor, event_log,
                                                                                          output_exp)
        test.test_manipulated_prefixes(args, preprocessor, event_log, manipulated_prefixes)

        output_exp = utils.get_output(args, preprocessor, output_exp)
        utils.print_output(args, output_exp, -1)
        utils.write_output(args, output_exp, -1)
