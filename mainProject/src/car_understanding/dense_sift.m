% function dense_sift(tmp_dir_name)
  fprintf('In MATLAB\n')
  % add vlfeat to path
  run('/usr0/home/ymovshov/Documents/Research/Code/thirdParty/vlfeat/toolbox/vl_setup');
  fprintf('loaded vlfeat\n')


  tmp_dir_name = './tmp';
  % Load data from python
  fprintf('loading data\n')
  data = load(fullfile(tmp_dir_name, 'data.mat'));
  img_names = data.img_cell;
  out_names = data.data_cell;

  n_imgs = length(img_names);
  fprintf('running dense sift on %d images\n', n_imgs)

  for i=1:n_imgs
    im = imread(img_names{i});
    return
    [frames, descrs] = vl_phow(im2single(im), 'step', 4, 'sizes', [8 12 16 24 30]);
    save(out_names{i}, 'frame', 'descrs');
  end
  fprintf('done!\n')


  quit;
% end
