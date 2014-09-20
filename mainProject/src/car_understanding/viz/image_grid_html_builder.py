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
    self.with_sort_data_ = False
    self.sort_def_ = None
    self.sort_by_dict_ = None


  # def AddRuler(self, title=''):
  #   self.html_data_ += '<div class="stamp stamp1"></div>'
    # self.html_data_ += '<hr>' + '<center> <h2> ' + str(title) + '</h2></center> <br> <hr>'

  def AddBox(self, img, title, meta_dict={'confidence': 0}):
      img_base_64 = ImageToBase64(img)
      img_txt = kImageTemplate.format(data=img_base_64)

      meta_str = ''
      for key, value in meta_dict.items():
        meta_str += ' {0}="{1}"'.format(key, value)
      meta_str += ' bid="{}"'.format(self.box_id_)
      box_str = kTableTemplate.format(data=img_txt, title=title,
                                                        meta=meta_str)
      self.html_data_ +=  box_str + '\n'
      self.box_id_ += 1

  def AddSortingFunctionality(self, sort_by_dict):
    self.sort_by_dict_ = sort_by_dict
    get_sort_data = ''
    for key, is_number in sort_by_dict.items():
      number_keyword = ''
      if is_number:
        number_keyword = 'parseFloat'
      get_sort_data += '{key}: "[{key}] {keyword}", \n'.format(key=key, keyword=number_keyword)

    self.sort_def_ = get_sort_data
    self.with_sort_data_ = True


  def SaveToFile(self, filename, title='Untitled'):
    data = kHtmlTemplate.replace('__TITLE__', title)

    if self.with_sort_data_:
      data = data.replace('__GET_SORT_DATA__', self.sort_def_)
      # Add buttons
      button_str = '<div id="sorts" class="button-group">\n'
      button_str +='<button class="button is-checked " data-sort-by="original-order">original order</button>\n'
      for key in self.sort_by_dict_.keys():
        button_str +='<button class="button " data-sort-by="{0}">{0}</button>\n'.format(key)
      button_str += '</div>\n'

    else:
      # data = data.replace('__SORT_BY__', '')
      data = data.replace('__GET_SORT_DATA__', '')

    data = data.replace('__BODY__', self.html_data_ )
    data = data.replace('__BUTTONS__', button_str)

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

    /* ---- button ---- */

.button {
  display: inline-block;
  padding: 0.5em 1.0em;
  background: #EEE;
  border: none;
  border-radius: 7px;
  background-image: linear-gradient( to bottom, hsla(0, 0%, 0%, 0), hsla(0, 0%, 0%, 0.2) );
  color: #222;
  font-family: sans-serif;
  font-size: 16px;
  text-shadow: 0 1px white;
  cursor: pointer;
}

.button:hover {
  background-color: #8CF;
  text-shadow: 0 1px hsla(0, 0%, 100%, 0.5);
  color: #222;
}

.button:active,
.button.is-checked {
  background-color: #28F;
}

.button.is-checked {
  color: white;
  text-shadow: 0 -1px hsla(0, 0%, 0%, 0.8);
}

.button:active {
  box-shadow: inset 0 1px 10px hsla(0, 0%, 0%, 0.8);
}

/* ---- button-group ---- */

.button-group:after {
  content: '';
  display: block;
  clear: both;
}

.button-group .button {
  float: left;
  border-radius: 0;
  margin-left: 0;
  margin-right: 1px;
}

.button-group .button:first-child { border-radius: 0.5em 0 0 0.5em; }
.button-group .button:last-child { border-radius: 0 0.5em 0.5em 0; }

  </style>


  <!-- scripts at bottom of page -->

</head>
<body >
__BUTTONS__
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

  // Allow sorting
  var $container = $('#container').isotope({
    getSortData: {
      __GET_SORT_DATA__
    }
  });
  $container.isotope({
    sortBy : 'original-order',
    sortAscending: true,
  });

  // sort items on button click
  $('#sorts').on( 'click', 'button', function() {
    var sortByValue = $(this).attr('data-sort-by');
    if (sortByValue == 'original-order') {
     var sortAscendingValue = true
    }
    else {
     var sortAscendingValue = false
    }
    $container.isotope({ sortBy: sortByValue, sortAscending: sortAscendingValue});
  });
  // change is-checked class on buttons
    $('.button-group').each( function( i, buttonGroup ) {
      var $buttonGroup = $( buttonGroup );
      $buttonGroup.on( 'click', 'button', function() {
        $buttonGroup.find('.is-checked').removeClass('is-checked');
        $( this ).addClass('is-checked');
      });
    });

</script>
</body>
</html>
'''

kTableTemplate = '''<div class="box"  {meta}>
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
  builder.AddBox(img, 'test image 1', meta_dict={'confidence': 1, 'class_name': 2})
  builder.AddBox(img, 'test image 2', meta_dict={'confidence': 2, 'class_name': 1})
  # builder.AddRuler('LALA')
  builder.AddBox(img, 'test image 3', meta_dict={'confidence': 3, 'class_name': 3})
  builder.AddBox(img, 'test image 4', meta_dict={'confidence': 4, 'class_name': 2})

  builder.AddSortingFunctionality({'confidence' : True, 'class_name': True})

  builder.SaveToFile('/Users/yair/Downloads/test.html', 'my test html')
