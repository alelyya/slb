if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(dest='filenames', metavar='filename', nargs='*')
    parser.add_argument('-o', dest='output_folder', action='store', help='./results/ if not specified')
    parser.add_argument('-v', dest='verbose', action='store_true', help='save all intermediate results')
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    from processing import main
    main(args.filenames, args.output_folder, args.verbose)
