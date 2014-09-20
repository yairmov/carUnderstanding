import base64
from PIL import Image
import tempfile


def ImageToBase64(img):
    if type(img) == str:
      fname= img
    else:
      f = tempfile.NamedTemporaryFile()
      fname = f.name
      img.save(f.name, ".jpg")

    with open(fname, "rb") as img_file:
      return base64.b64encode(img_file.read())

class ImageGridHtmlBuilder(object):
  """docstring for ImageGridHtmlBuilder"""
  def __init__(self):
    super(ImageGridHtmlBuilder, self).__init__()
    self.html_data_ = ''
    self.box_id_ = 0


  # def AddRuler(self, title=''):
  #   self.html_data_ += '<div class="stamp stamp1"></div>'
    # self.html_data_ += '<hr>' + '<center> <h2> ' + str(title) + '</h2></center> <br> <hr>'

  def AddBox(self, img, title):
      img_base_64 = ImageToBase64(img)
      img_txt = kImageTemplate.format(data=img_base_64)
      box_str = kTableTemplate.format(data=img_txt, title=title, bid=str(self.box_id_))
      self.html_data_ +=  box_str + '\n'
      self.box_id_ += 1

  def SaveToFile(self, filename, title='Untitled'):
    data = kHtmlTemplate.replace('__BODY__', self.html_data_)
    data = data.replace('__TITLE__', title)
    with open(filename, "wb") as out_file:
      out_file.write(data)



kHtmlTemplate = '''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />

  <title>__TITLE__</title>

  <meta http-equiv="X-UA-Compatible" content="IE=edge,chrome=1">

  <style media="screen" type="text/css">
    .box {
    margin: 5px;
    padding: 5px;
    background: #D8D5D2;
    font-size: 11px;
    line-height: 1.4em;
    float: left;
    # max-width: 100px;
    -webkit-border-radius: 5px;
       -moz-border-radius: 5px;
            border-radius: 5px;
    }
    .stamp {
    position: absolute;
    background: orange;
    border: 4px dotted black;
  }

    .masonry,
    .masonry .masonry-brick {
      -webkit-transition-duration: 0.7s;
         -moz-transition-duration: 0.7s;
          -ms-transition-duration: 0.7s;
           -o-transition-duration: 0.7s;
              transition-duration: 0.7s;
    }

    .masonry {
      -webkit-transition-property: height, width;
         -moz-transition-property: height, width;
          -ms-transition-property: height, width;
           -o-transition-property: height, width;
              transition-property: height, width;
    }

    .masonry .masonry-brick {
      -webkit-transition-property: left, right, top;
         -moz-transition-property: left, right, top;
          -ms-transition-property: left, right, top;
           -o-transition-property: left, right, top;
              transition-property: left, right, top;
    }
  </style>


  <!-- scripts at bottom of page -->

</head>
<body >
<div id="container" class="clearfix">

__BODY__

</div>
<script src="http://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
<script src="http://www.cs.cmu.edu/~ymovshov/Files/js/masonry.pkgd.min.js"></script>
<script src="http://www.cs.cmu.edu/~ymovshov/Files/js/isotope.pkgd.min.js"></script>
<script src="http://www.cs.cmu.edu/~ymovshov/Files/js/imagesloaded.pkgd.min.js"></script>
<script src="http://www.cs.cmu.edu/~ymovshov/Files/js/modernizr-transitions.js"></script>
<script>
  var $container = $('#container');
  $container.imagesLoaded( function(){
    $container.masonry({
      itemSelector : '.box',
      columnWidth: 50,
      isAnimated: !Modernizr.csstransitions
    });
  });
</script>
</body>
</html>
'''

kTableTemplate = '''<div class="box  {bid}">
      <table style="display:inline-table;">
      <tr><td>
      {data}
      </td></tr>
      <tr><td>{title}</tr></table>
    </div>
      '''

kImageTemplate = '''<img src="data:image/jpeg;base64,
{data}
"  />'''

if __name__ == '__main__':
  img = '/Users/yair/Downloads/test.jpg'
  builder = ImageGridHtmlBuilder()
  builder.AddBox(img, 'test image 1')
  builder.AddBox(img, 'test image 2')
  # builder.AddRuler('LALA')
  builder.AddBox(img, 'test image 3')
  builder.AddBox(img, 'test image 4')
  builder.AddBox(img, 'test image 4')
  builder.AddBox(img, 'test image 4')
  builder.AddBox(img, 'test image 4')
  builder.SaveToFile('/Users/yair/Downloads/test.html', 'my test html')
