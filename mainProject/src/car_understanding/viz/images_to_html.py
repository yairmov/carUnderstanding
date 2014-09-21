from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import sys
from os import listdir
from os.path import isfile, join, basename
from image_grid_html_builder import ImageGridHtmlBuilder
'''
USAGE:
images_to_html 1.jpg 2.jpg 3.jpg --output_path images.html
'''


def GetFileList(arg):
  if isinstance(arg, str):
    # arg is a name of a directory, get all files from it
    return [ f for f in listdir(arg) if isfile(join(arg,f)) ]

  if isinstance(arg, list) and isinstance(arg[0], str):
    return arg

  raise Exception('input must be a single string or a list of strings.')


def main(argv=None):  # IGNORE:C0111
  '''Command line options.'''

  if argv is None:
      argv = sys.argv
  else:
      sys.argv.extend(argv)

  parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
  parser.add_argument(dest="image_files",
                                   help="list of files to save to html. Could also be a folder name",
                                   nargs='+', default=None)
  parser.add_argument("--output_path", dest="output_path",
                                   help="path to html output file.", default=None, type=str)
  parser.add_argument("--title", dest="title",
                                   help="title of html.", default='', type=str)

  # Process arguments
  args = parser.parse_args()
  if args.output_path is None:
    raise Exception('Must provide output path')

  image_files = GetFileList(args.image_files)
  builder = ImageGridHtmlBuilder()
  for img in image_files:
    name = basename(img)
    builder.AddBox(img, name)

  builder.SetMaxWidth(256)
  builder.SaveToFile(args.output_path, args.title)




if __name__ == '__main__':
    main()
