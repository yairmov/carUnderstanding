function dense_sift(tmp_dir_name)
  fprintf('In MATLAB\n')
  % add vlfeat to path
  run('/usr0/home/ymovshov/Documents/Research/Code/thirdParty/vlfeat/toolbox/vl_setup');
  % run('/usr0/home/ymovshov/Documents/pre_2014_research/Code/3rd_Party/vlfeat/toolbox/vl_setup');
  fprintf('loaded vlfeat\n')

  % open matlab worker pool
  % matlabpool open 12


  % tmp_dir_name = './tmp';
  % Load data from python
  fprintf('loading data\n')
  data = load(fullfile(tmp_dir_name, 'data.mat'));
  img_names = data.img_cell;
  out_names = data.data_cell;
  sizes = double(data.sizes);

  n_imgs = length(img_names);
  fprintf('running dense sift on %d images\n', n_imgs)

  % for i=1:n_imgs
  for i=1:1
    fprintf('%s\n', img_names{i})
    im = imread(img_names{i});
    [frames, desc] = vl_phow(im2single(im), 'step', 4, 'sizes', sizes, 'FloatDescriptors', true);
    frames = frames'; % now each row of frames is: [x, y, ???, patch_size]
    desc = desc';
    save(out_names{i}, 'frames', 'desc', '-v7');
  end
  fprintf('done!\n')


  % close pool
  % matlabpool close force
  quit;
% end
