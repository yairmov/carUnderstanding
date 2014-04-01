function dense_sift(tmp_dir_name)
  % add vlfeat to path
  run('/usr0/home/ymovshov/Documents/Research/Code/thirdParty/vlfeat/toolbox/vl_setup');

  % Load data from python
  data = load(fullfile(tmp_dir_name, 'data.mat'));
  img_names = data.img_cell;
  out_names = data.data_cell;

  n_imgs = length(img_names)
  for i=1:n_imgs
    im = imread(img_names{i})
    [frames, descrs] = vl_phow(im2single(im), 'step', 4, 'sizes', [8 12 16 24 30]);
    save(out_names{i}, 'frame', 'descrs')
  end


  exit;
end