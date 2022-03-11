
import argparse
from computer_vision.cnn import FashionMNISTClassifier

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Use a CNN classifier to classify the "+
        "Fashion MNIST training dataset",
        epilog="Built with <3 by Emmanuel Byrd at 8th Light Ltd."
    )
    parser.add_argument(
        "--model_path", metavar="./model.h5", type=str, default="./model.h5",
        help="The read/write path of the trained model " + 
        "(default: ./model.h5)"
    )
    parser.add_argument(
        "--history_path", metavar="./train_hist", type=str, 
        default="./train_hist",
        help="The read/write path of the training history "+
        "(default: ./hist)"
    )
    parser.add_argument(
        "--epochs", metavar="10", type=int, default=10,
        help="Number of training epochs or complete sweeps over the dataset " +
        "(default: 10)"
    )
    parser.add_argument(
        "--mini_batch_size", metavar="128", type=int, default=128,
        help="Size of the mini-batches (default: 128)"
    )
    parser.add_argument(
        "--validation_split", metavar="0.2", type=float, default=0.2,
        help="Ratio of the data amount to use in validation (default: 0.2)"
    )
    parser.add_argument(
        "--use_stored", action=argparse.BooleanOptionalAction, type=bool,
        help="Use this flag to load a previously model and history"
    )
    parser.add_argument(
        "--train", action=argparse.BooleanOptionalAction, type=bool,
        help="Use this flag to train from scratch"
    )
    parser.add_argument(
        "--plot", action=argparse.BooleanOptionalAction, type=bool,
        help="Plot the training history"
    )
    parser.add_argument(
        "--save_plot", metavar="./plot.png", type=str,
        help="Path to store the generated plot. Leave blank to ignore."
    )
    parser.add_argument(
        "--save_model", action=argparse.BooleanOptionalAction,type=bool,
        help="Save the model"
    )
    return parser

def execute_cnn(args):
    cnn_trainer = FashionMNISTClassifier(args.model_path, args.history_path)

    if args.use_stored:
        cnn_trainer.load_model()
        cnn_trainer.load_train_history()
        cnn_trainer.show_summary()
    elif args.train:
        cnn_trainer.load_dataset()
        cnn_trainer.build_model()
        cnn_trainer.show_summary()
        cnn_trainer.compile()
        cnn_trainer.train(
            args.mini_batch_size, 
            args.epochs, 
            args.validation_split
        )
    else:
        print (
            "Do you want to load a model or train a new one? "+
            "(--use_stored, --train)"
        )
        return
    
    if args.plot or args.save_plot:
        cnn_trainer.plot_hist(args.plot, args.save_plot)

    if args.save_model:
        cnn_trainer.save_model()
        cnn_trainer.save_train_history()

def main():
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args()

    execute_cnn(args)

    print("Finished.")
    
if __name__ == "__main__":
    main()