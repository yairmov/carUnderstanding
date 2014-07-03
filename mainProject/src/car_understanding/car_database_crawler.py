'''
Script for crawling http://www.cardatabase.net/ and downloading
all car images and meta data.
'''
import BeautifulSoup as bs
import urllib2
from PIL import Image
import io
import json
import time
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
import sys
import os
import logging


#: Glogbal logger instance
LOG = logging.getLogger(__name__)
#: Debug level names as a string
LOG_HELP = ','.join(["%d=%s" % (4-x, logging.getLevelName((x+1)*10)) for x in xrange(5)])
#: Console LOG format
LOG_FORMAT_CONS = '%(asctime)s %(funcName)-12s %(levelname)8s %(message)s'
#: File LOG format
LOG_FORMAT_FILE = '%(asctime)s %(funcName)s[%(process)d] %(levelname)10s %(message)s'
#: Levels of logging translation (count of -v, log level)
LOGLEVEL_DICT = { 1 : 50, 2:40, 3:20, 4:10, 5:1 }

DEFAULT_VERBOSITY = 0


def GetImageFromUrl(url):
    r = urllib2.urlopen(url)
    image_file = io.BytesIO(r.read())
    img = Image.open(image_file)
    return img

'''
Saves image to disk. The ext is guessed based on the file encoding.
Image should be a PIL image
'''
def SaveImgToDisk(img, filename, ext=None):
    if ext is None:
        if img.format == 'PNG':
            ext = '.png'
        elif img.format == 'JPEG':
            ext = '.jpg'
        else:
            print 'WARNING: Could not guess encoding based on format: '.format(img.format)
            ext = '.jpg'

    img.save(filename + ext)

'''
Takes a string like this: <a href="http://www.cardatabase.net/search/search.php?manufacturer=BMW">More</a>
outputs: {'manufacturer' : 'BMW'}
'''
def ExtractDataFromString(data_str):
    (first, value) = data_str.split('=')
    (dump, key) = first.split('?')
    return {key : value}


'''
Queries for a single web page, and extracts the car image, and
meta data.
img, meta_data = GetImageAndData(1)
'''
def GetImageAndData(img_id):
    base_website = 'http://www.cardatabase.net'
    url = base_website + '/search/photo_search.php?id={:08}&size=large'.format(img_id)
    page = urllib2.urlopen(url)
    soup = bs.BeautifulSoup(page)

    # Get the image
    img=soup.findAll('img',{'class':'imageshadow'})
    img_url = base_website + img[0].get('src')
    img = GetImageFromUrl(img_url)

    # Now Get metadata
    meta_data = {}
    links = soup.findAll('a')
    for l in links:
        if len(l.contents) > 0 and l.contents[0] == 'More':
            meta_data.update(ExtractDataFromString(l.attrs[0][1]))

    return img, meta_data


def RunCrawl(args):

  IMG_NAME_FMT   = os.path.normpath(os.path.join(args.output_path , 'img_{:07}'))
  DATA_NAME_FMT = os.path.normpath(os.path.join(args.output_path , 'meta_{:07}.json'))

  for ii in xrange(1, 10):
    # img, meta_data = GetImageAndData(ii)
    LOG.info('Saving image: '  + IMG_NAME_FMT.format(ii))
    LOG.info('Saving data: '  + DATA_NAME_FMT.format(ii))
    time.sleep(args.delay)



def main(argv=None):  # IGNORE:C0111
  if argv is None:
    argv = sys.argv
  else:
    sys.argv.extend(argv)

  program_shortdesc = __import__('__main__').__doc__.split("\n")[1]
  program_license = '''%s

  Copyright 2014 Yair Movshovitz-Attias. All rights reserved.

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

  USAGE ??
  ''' % (program_shortdesc)
  parser = ArgumentParser(description=program_license,
                          formatter_class=RawDescriptionHelpFormatter)

  parser.add_argument(dest="output_path", help="path to save downloaded data." , default=None)
  parser.add_argument("-l", "--logfile", dest="logfile",    default = None, help = "Log to file instead off console [default: %default]" )
  parser.add_argument("-v", action="count", dest="verbosity", default = DEFAULT_VERBOSITY, help = "Verbosity. Add more -v to be more verbose (%s) [default: %%default]" % LOG_HELP)
  parser.add_argument("-d", "--delay", dest="delay", default = 1, help = "time delay between images in seconds. [default: %default]" )



  # Process arguments
  args = parser.parse_args()

  verbosity = LOGLEVEL_DICT.get(int(args.verbosity), DEFAULT_VERBOSITY)

  # Set up logging
  if args.logfile is None:
      logging.basicConfig(level=verbosity, format=LOG_FORMAT_CONS)
  else:
      logfilename = os.path.normpath(args.logfile)
      logging.basicConfig(level=verbosity, format=LOG_FORMAT_FILE, filename=logfilename, filemode='a')
      print >> sys.stderr, "Logging to %s" % logfilename

  LOG.info("Got arguments: ")
  LOG.info(args)



  RunCrawl(args)

if __name__ == '__main__':
    main()
