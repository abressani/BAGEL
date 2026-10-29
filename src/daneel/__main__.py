import argparse
from datetime import datetime, timezone
from pathlib import Path

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
    actions.add_argument("-g", "--galaxy", action="store_true", help="Plot a GalaxyMNIST galaxy and its radial brightness profile "
        "(data_root, split and index are read from the 'galaxy' section of the input file)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output image for --transit (default: lc.png) or --galaxy (default: galaxy.png)",
    )
    parser.add_argument("-idx", "--index", type=int, default=None,
        help="GalaxyMNIST image index; only used with --galaxy when the "
        "input file's 'galaxy' section has no 'index'",
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.now(timezone.utc)

    input_pars = Parameters(args.input_file).params

    if args.transit:
        transit = TransitModel(input_pars['transit'])
        transit.plot_light_curve(args.output or "lc.png")

    elif args.galaxy:
        from daneel.galaxies import plot_galaxy

        galaxy_pars = input_pars.get("galaxy")
        if not isinstance(galaxy_pars, dict):
            parser.error("input file must contain a 'galaxy' section")

        if galaxy_pars.get("data_root") is None:
            parser.error("'galaxy' section is missing: data_root")

        # The YAML takes priority; --index is only a fallback.
        index = galaxy_pars.get("index")
        if index is None:
            index = args.index
        elif args.index is not None:
            print(f"Warning: using index {index} from the input file; ignoring --index {args.index}")
        if index is None:
            parser.error("no galaxy index given: set 'index' in the 'galaxy' section or pass --index")

        data_root = Path(galaxy_pars["data_root"]).expanduser()
        if not data_root.is_absolute():
            parser.error("galaxy.data_root must be an absolute path")

        result = plot_galaxy(
            data_root=data_root,
            split=galaxy_pars.get("split", "train"),
            index=int(index),
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
