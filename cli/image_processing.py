"""CLI to use the convolution examples"""
import argparse
import numpy as np
from PIL import Image

from computer_vision.image_processing import kernel, pipelines, reshaping
from plotter import MultiPlot


def buil_arg_parser():
    """Parses the user's arguments"""
    parser = argparse.ArgumentParser(
        description="Explore Image Processing techniques that use convolutions",
        epilog="Built with <3 by Emmanuel Byrd at 8th Light Ltd.")
    parser.add_argument(
        "--source-path", metavar="./image.jpg", type=str,
        required=True,
        help="The read path of the input image (required)"
    )
    parser.add_argument(
        "--destination-path", metavar="./image.jpg", type=str,
        help="The write path of the output image"
    )
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, type=bool,
        help="Whether to show the resulting plot"
    )
    parser.add_argument(
        "--color", action=argparse.BooleanOptionalAction, type=bool,
        help="Whether to use all 3 channels from an image"
    )
    parser.add_argument(
        "--example",
        choices=['kernel', 'gauss', 'blur', 'opening', 'closing',
                 'inner_border', 'outer_border'],
        help="Examples to choice from",
        required=True,
    )
    parser.add_argument(
        "--kernel",
        choices=['top', 'bottom', 'left', 'right',
                 'top_sobel', 'bottom_sobel', 'left_sobel', 'right_sobel',
                 'sharpen', 'outline'],
        help="The write path of the output image"
    )
    parser.add_argument(
        "--gauss-sigma", metavar="1.", type=float, default=1.,
        help="Sigma parameter of the Gaussian filter (default: 1.0)"
    )
    parser.add_argument(
        "--gauss-size", metavar="5", type=int, default=5,
        help="Size of the Gaussian filter (default: 5)"
    )
    return parser


color_agnostic_examples = [
    "kernel", "gauss", "blur"
]

triple_plot_examples = [
    "opening", "closing", "inner_border", "outer_border"
]


def draw_figures(args: argparse.Namespace, plotter: MultiPlot):
    """Show or save the generated figures"""

    suptitle = "Example - " + args.example
    if args.example in ["gauss", "blur"]:
        suptitle += f" size:{args.gauss_size} sigma:{args.gauss_sigma}"
    figure = plotter.draw(suptitle)

    if args.show:
        figure.show()
        input("Press any key to continue...")

    if args.destination_path:
        print("Saving plot in " + args.destination_path)
        figure.savefig(args.destination_path)


def output_color_agnostic(args: argparse.Namespace, input_img: np.ndarray):
    """Create the examples that are suitable for color and grayscale inputs"""
    if args.example == "kernel":
        kernel_choice = kernel.from_name(args.kernel)
        output = pipelines.padded_convolution_same_kernel(
            input_img, kernel_choice)
    elif args.example == "gauss":
        kernel_gauss = kernel.simple_gauss(args.gauss_size, args.gauss_sigma)
        output = pipelines.padded_convolution_same_kernel(
            input_img, kernel_gauss)
    elif args.example == "blur":
        output = pipelines.padded_blur(
            input_img, args.gauss_size, args.gauss_sigma)
    return output


def outputs_triple_plot_examples(
        args: argparse.Namespace, input_img: np.ndarray):
    """Create the examples that produce three figures in the plot"""

    if args.example == "opening":
        return pipelines.opening(input_img)

    if args.example == "closing":
        return pipelines.closing(input_img)

    if args.example == "inner_border":
        return pipelines.inner_border(input_img)

    # args.example == "outer_border"
    return pipelines.outer_border(input_img)


def execute_color(args: argparse.Namespace):
    """Do the example in color"""
    plotter = MultiPlot()
    img = np.asarray(Image.open(args.source_path))
    plotter.add_figure(img, "input")

    img_reshaped = reshaping.channel_as_first_dimension(img)
    output_reshaped = output_color_agnostic(args, img_reshaped)

    output = reshaping.channel_as_last_dimension(output_reshaped)
    plotter.add_figure(output.astype(int), "output")

    draw_figures(args, plotter)


def execute_grayscale(args: argparse.Namespace):
    """Do the example in grayscale"""
    plotter = MultiPlot()
    img = np.asarray(Image.open(args.source_path).convert("L"))
    plotter.add_figure(img, "input", "gray")

    if args.example in color_agnostic_examples:
        output = output_color_agnostic(args, img)
    elif args.example in triple_plot_examples:
        middlestep, output = outputs_triple_plot_examples(args, img)
        plotter.add_figure(middlestep, "middle step", "gray")

    plotter.add_figure(output, "output", "gray")

    draw_figures(args, plotter)


def main():
    """Main function"""
    arg_parser = buil_arg_parser()
    args = arg_parser.parse_args()

    if args.color and args.example not in color_agnostic_examples:
        print("Color examples do not support " + args.example)
        return

    if args.color:
        execute_color(args)
    else:
        execute_grayscale(args)

    print("Finished.")


if __name__ == "__main__":
    main()
