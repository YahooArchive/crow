# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
pushd paris/

# Get data
mkdir data/
curl http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz | tar xz -C data/
curl http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz | tar xz -C data/

# Put everything in same dir for simplicity
mv data/paris/defense/* data/
mv data/paris/eiffel/* data/
mv data/paris/general/* data/
mv data/paris/invalides/* data/
mv data/paris/louvre/* data/
mv data/paris/moulinrouge/* data/
mv data/paris/museedorsay/* data/
mv data/paris/notredame/* data/
mv data/paris/pantheon/* data/
mv data/paris/pompidou/* data/
mv data/paris/sacrecoeur/* data/
mv data/paris/triomphe/* data/
rm -rf data/paris/

# Get labels
mkdir groundtruth/
curl http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_120310.tgz | tar xz -C groundtruth/

popd



