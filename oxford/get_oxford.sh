# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
pushd oxford/

# Get data
mkdir data/
curl http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz | tar xz -C data/

# Get labels
mkdir groundtruth/
curl http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz | tar xz -C groundtruth/

popd



