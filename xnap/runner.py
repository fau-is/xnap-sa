import xnap.config as config
import xnap.utils as utils
from xnap.nap.preprocessor import Preprocessor as Preprocessor
import xnap.nap.tester as test
import xnap.nap.trainer as train
import xnap.exp.lrp.lrp as lrp


if __name__ == '__main__':

    args = config.load()
    output = utils.load_output()
    utils.clear_measurement_file(args)

    if args.mode == 0:

        preprocessor = Preprocessor()
        event_log = preprocessor.get_event_log(args)

        if args.cross_validation:
            for iteration_cross_validation in range(0, args.num_folds):
                ##the following code is not used so far since cross_validation defaults to false and num_folds defaults to 0
                """
                preprocessor.data_structure['support']['iteration_cross_validation'] = iteration_cross_validation

                output["training_time_seconds"].append(train.train(args, preprocessor))
                test.test(args, preprocessor)

                output = utils.get_output(args, preprocessor, output)
                utils.print_output(args, output, iteration_cross_validation)
                utils.write_output(args, output, iteration_cross_validation)

            utils.print_output(args, output, iteration_cross_validation + 1)
            utils.write_output(args, output, iteration_cross_validation + 1)
            """

        else:
            output["training_time_seconds"].append(train.train(args, preprocessor))
            test.test(args, preprocessor)

            output = utils.get_output(args, preprocessor, output)
            utils.print_output(args, output, -1)
            utils.write_output(args, output, -1)


    elif args.mode == 1:
        preprocessor = Preprocessor(args)
        trace = preprocessor.get_random_process_instance(args.rand_lower_bound, args.rand_upper_bound)
        lrp.calc_and_plot_relevance_scores_instance(trace, args, preprocessor)

    else:
        preprocessor = Preprocessor(args)
        trace = preprocessor.get_random_process_instance(args.rand_lower_bound, args.rand_upper_bound)
        scores = lrp.calc_relevance_scores_instance(trace, args, preprocessor)
        print(scores)