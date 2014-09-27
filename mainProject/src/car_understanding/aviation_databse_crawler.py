'''
Script for crawling http://www.myaviation.net/ and downloading
all airplane images and meta data.
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
from path import path


#: Glogbal logger instance
LOG = logging.getLogger(__name__)
#: Debug level names as a string
LOG_HELP = ','.join(["%d=%s" % (4-x, logging.getLevelName((x+1)*10)) for x in xrange(5)])
#: Console LOG format
LOG_FORMAT_CONS = '%(asctime)s %(lineno)s - %(funcName)-12s %(levelname)8s %(message)s'
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
Takes a string like this: <a href="http://www.myaviation.net/search/search.php?manufacturer=BMW">More</a>
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
def GetImageAndData(img_id, base_website='http://www.myaviation.net'):
    url = base_website + '/search/photo_search.php?id={:08}&size=large'.format(img_id)
    try:
      page = urllib2.urlopen(url)
      soup = bs.BeautifulSoup(page)

      # Get the image
      img=soup.findAll('img')
      if (len(img) == 0): # no image found
        return None, None

      img_url = None
      for ii in range(len(img)):
      src = img[ii].get('src')
        if src is not None and src.endswith('big'):
          img_url = base_website + src
          break

      if img_url is None:
        return None, None

      img = GetImageFromUrl(img_url)

      # Now Get metadata
      meta_data = {}
      links = soup.findAll('a')
      for l in links:
          if len(l.contents) > 0 and l.contents[0] == 'More':
              meta_data.update(ExtractDataFromString(l.attrs[0][1]))
    except:
      e = sys.exc_info()[0]
      LOG.warning("Failed to grab image: {}. Error: {}".format(img_id, e))
      return None, None

    return img, meta_data


def MakeDirIfNeeded(dirname):
  p = path(dirname)
  if not p.isdir():
    p.makedirs()

def RunCrawl(args):

  IMG_NAME_FMT   = 'img_{:07}.jpg'
  DATA_NAME_FMT = 'meta_{:07}.json'


  MakeDirIfNeeded(args.output_path)

  for ii in xrange(args.start_id , args.end_id):
    time.sleep(args.delay)

    if ii % 100 == 0:
      # sleep for longer
      time.sleep(60)

    dir_path = os.path.join(args.output_path, '{:04}'.format(ii / 1000))
    MakeDirIfNeeded(dir_path)

    curr_img_fname = os.path.join(dir_path, IMG_NAME_FMT.format(ii))
    curr_data_fname = os.path.join(dir_path, DATA_NAME_FMT.format(ii))

    img, meta_data = GetImageAndData(ii, args.website)
    if (img is None):
      LOG.info("Was not able to get page for image: {}".format(ii))
      continue

    LOG.info('Saving image: '  + curr_img_fname)
    try:
      img.save(curr_img_fname)
    except:
      e = sys.exc_info()[0]
      LOG.warning("Failed to save image: {}. Error: {}".format(ii, e))
      continue

    LOG.info('Saving data: '  + curr_data_fname)
    with open(curr_data_fname, 'w') as f:
      json.dump(meta_data, f)






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
  parser.add_argument("-d", "--delay", dest="delay", default = 2, help = "time delay between images in seconds. [default: %default]" )
  parser.add_argument("-s", "--start", dest="start_id", default = 1, type=int, help = "img id to start from. [default: %default]" )
  parser.add_argument("-e", "--end", dest="end_id", default = 10, type=int, help = "img id to end at. [default: %default]" )
  parser.add_argument("-w", "--website", dest="website", default = "http://www.myaviation.net", type=str, help = "web site to crawl. [default: %default]" )


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
