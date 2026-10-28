import argparse
from datetime import datetime, timezone

from daneel.detection import TransitModel
from daneel.parameters import Parameters


def main():
    parser = argparse.ArgumentParser(description="Model and analyse exoplanet transits.")

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("-t", "--transit", action="store_true", help="Plot a transit light curve")

    actions.add_argument("-d", "--detect", action="store_true", help="Run exoplanet detection")
    actions.add_argument("-a", "--atmosphere", action="store_true", help="Run atmospheric characterisation")
    actions.add_argument(
        "-g",
        "--galaxy",
        action="store_true",
        help="Plot a GalaxyMNIST galaxy and its radial brightness profile",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image for --transit (default: lc.png) or --galaxy (default: galaxy.png)",
    )

    parser.add_argument(
        "--index",
        type=int,
        default=7,
        help="GalaxyMNIST image index (default: 7)",
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.now(timezone.utc)

    input_pars = Parameters(args.input_file).params

    if args.transit:
        transit = TransitModel(input_pars['transit'])
        transit.plot_light_curve(args.output or "lc.png")
    elif args.galaxy:
        # Imported here so that the galaxy dependencies (h5py, datasets, ...)
        # are only needed when --galaxy is used.
        from daneel.galaxies import plot_galaxy

        result = plot_galaxy(
            data_root="data",
            split="train",
            index=args.index,
            output_path=args.output or "galaxy.png",
            show=False,
        )
        print(f"Galaxy {result['index']} ({result['split']}): {result['class_name']}")
        print(f"Saved: {result['output_path']}")
    elif args.detect:
        parser.error("detection is not implemented yet; use --transit")
    elif args.atmosphere:
        parser.error("atmospheric characterisation is not implemented yet; use --transit")

    finish = datetime.now(timezone.utc)
    print(f"Daneel completed in {(finish - start).total_seconds():.2f}s")


if __name__ == "__main__":
    main()
