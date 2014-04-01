function dense_sift(tmp_dir_name)
  data = load(fullfile(tmp_dir_name, 'data.mat'));
  img_names = data.img_cell;
  out_names = data.data_cell;
end