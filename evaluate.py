# Copyright 2015, Yahoo Inc.
# Licensed under the terms of the Apache License, Version 2.0. See the LICENSE file associated with the project for terms.
import sys
import os
import glob
from functools import partial
from tempfile import NamedTemporaryFile
import numpy as np
from crow import run_feature_processing_pipeline, apply_crow_aggregation, apply_ucrow_aggregation, normalize


def get_nn(x, data, k=None):
    """
    Find the k top indices and distances of index data vectors from query vector x.

    :param ndarray x:
        the query vector
    :param ndarray data:
        the index vectors
    :param int k:
        optional k to truncate return

    :returns ndarray idx:
        the indices of index vectors in ascending order of distance
    :returns ndarray dists:
        the squared distances
    """
    if k is None:
        k = len(data)

    dists = ((x - data)**2).sum(axis=1)
    idx = np.argsort(dists) 
    dists = dists[idx]

    return idx[:k], dists[:k]


def simple_query_expansion(Q, data, inds, top_k=10):
    """
    Get the top-k closest vectors, average and re-query

    :param ndarray Q:
        query vector
    :param ndarray data:
        index data vectors
    :param ndarray inds:
        the indices of index vectors in ascending order of distance
    :param int top_k:
        the number of closest vectors to consider

    :returns ndarray idx:
        the indices of index vectors in ascending order of distance
    :returns ndarray dists:
        the squared distances
    """
    Q += data[inds[:top_k],:].sum(axis=0)
    return normalize(Q)


def load_features(feature_dir, verbose=True):
    """
    Iterate and load numpy pickle files in the provided directory along with the filename.

    :param feature_dir: directory to iterate or list of directories
    :type feature_dir: str or list
    :param bool verbose:
        optional flag to disabled progress printout

    :yields ndarray X:
        the ndarray from the pickle file
    :yields str name:
        the name of the file without file extension
    """
    if type(feature_dir) == str:
        feature_dir = [feature_dir]

    for directory in feature_dir:
        for i, f in enumerate(os.listdir(directory)):
            name = os.path.splitext(f)[0]

            # Print progress
            if verbose and not i % 100:
                sys.stdout.write('\rProcessing file %i' % i)
                sys.stdout.flush()

            X = np.load(os.path.join(directory, f))

            yield X, name

    sys.stdout.write('\n')
    sys.stdout.flush()


def load_and_aggregate_features(feature_dir, agg_fn):
    """
    Given a directory of features as numpy pickles, load them, map them
    through the provided aggregation function, and return a list of
    the features and a list of the corresponding file names without the
    file extension.

    :param feature_dir: directory to iterate or list of directories
    :type feature_dir: str or list
    :param callable agg_fn:
        map function for raw features

    :returns list features:
        the list of loaded features
    :returns list names:
        corresponding file names without extension
    """
    print 'Loading features %s ...' % str(feature_dir)
    features = []
    names = []
    for X, name in load_features(feature_dir):
        names.append(name)
        X = agg_fn(X)
        features.append(X)

    return features, names


def get_ap(inds, dists, query_name, index_names, groundtruth_dir, ranked_dir=None):
    """
    Given a query, index data, and path to groundtruth data, perform the query,
    and evaluate average precision for the results by calling to the compute_ap
    script. Optionally save ranked results in a file.

    :param ndarray inds:
        the indices of index vectors in ascending order of distance
    :param ndarray dists:
        the squared distances
    :param str query_name:
        the name of the query
    :param list index_names:
        the name of index items
    :param str groundtruth_dir:
        directory of groundtruth files
    :param str ranked_dir:
        optional path to a directory to save ranked list for query

    :returns float:
        the average precision for this query
    """

    if ranked_dir is not None:
        # Create dir for ranked results if needed
        if not os.path.exists(ranked_dir):
            os.makedirs(ranked_dir)
        rank_file = os.path.join(ranked_dir, '%s.txt' % query_name)
        f = open(rank_file, 'w')
    else:
        f = NamedTemporaryFile(delete=False)
        rank_file = f.name

    f.writelines([index_names[i] + '\n' for i in inds])
    f.close()

    groundtruth_prefix = os.path.join(groundtruth_dir, query_name)
    cmd = './compute_ap %s %s' % (groundtruth_prefix, rank_file)
    ap = os.popen(cmd).read()

    # Delete temp file
    if ranked_dir is None:
        os.remove(rank_file)

    return float(ap.strip())


def fit_whitening(whiten_features, agg_fn, d):
    """
    Calculate whitening parameters

    :param str whiten_features: 
        directory of features to fit whitening
    :param callable agg_fn: 
        aggregation function
    :param int d: 
        final feature dimension

    :returns dict params:
        a dict of transformation parameters
    """

    # Load features for fitting whitening
    data, _ = load_and_aggregate_features(whiten_features, agg_fn)

    # Whiten, and reduce dim of features
    # Whitening is trained on the same images that we query against here for expediency
    print 'Fitting PCA/whitening wth d=%d on %s ...' % (d, whiten_features)
    _, whiten_params = run_feature_processing_pipeline(data, d=d)

    return whiten_params


def run_eval(queries_dir, groundtruth_dir, index_features, whiten_params, out_dir, agg_fn, qe_fn=None):
    """
    Run full evaluation pipeline on specified data.

    :param str queries_dir: directory of query features
    :param str groundtruth_dir: directory of groundtruth info
    :param index_features: directory or list of directories of index features
    :type index_features: str or list
    :param str whiten_features: directory of features to fit whitening
    :param str out_dir: directory to save query results
    :param callable agg_fn: aggregation function
    :param callable qe_fn: query expansion function
    """

    data, image_names = load_and_aggregate_features(index_features, agg_fn)
    data, _ = run_feature_processing_pipeline(np.vstack(data), params=whiten_params)

    # Iterate queries, process them, rank results, and evaluate mAP
    aps = []
    for Q, query_name in load_features(queries_dir):
        Q = agg_fn(Q)

        # Normalize and PCA to final feature
        Q, _ = run_feature_processing_pipeline(Q, params=whiten_params)

        inds, dists = get_nn(Q, data)

        # perform query_expansion
        if qe_fn is not None:
            Q = qe_fn(Q, data, inds)
            inds, dists = get_nn(Q, data)

        ap = get_ap(inds, dists, query_name, image_names, groundtruth_dir, out_dir)
        aps.append(ap)

    return np.array(aps).mean()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--wt', dest='weighting', type=str, default='crow', help='weighting to apply for feature aggregation')

    parser.add_argument('--index_features', dest='index_features', type=str, default='oxford/pool5', help='directory containing raw features to index')
    parser.add_argument('--whiten_features', dest='whiten_features', type=str, default='paris/pool5', help='directory containing raw features to fit whitening')

    parser.add_argument('--queries', dest='queries', type=str, default='oxford/pool5_queries/', help='directory containing image files')
    parser.add_argument('--groundtruth', dest='groundtruth', type=str, default='oxford/groundtruth/', help='directory containing groundtruth files')
    parser.add_argument('--d', dest='d', type=int, default=128, help='dimension of final feature')
    parser.add_argument('--out', dest='out', type=str, default=None, help='optional path to save ranked output')
    parser.add_argument('--qe', dest='qe', type=int, default=0, help='perform query expansion with this many top results')
    args = parser.parse_args()

    # Select which aggregation function to apply
    if args.weighting == 'crow':
        agg_fn = apply_crow_aggregation
    else:
        agg_fn = apply_ucrow_aggregation

    if args.qe > 0:
        qe_fn = partial(simple_query_expansion, top_k=args.qe)
    else:
        qe_fn = None
        
    # compute whitening params
    whitening_params = fit_whitening(args.whiten_features, agg_fn, args.d)

    # compute aggregated features and run the evaluation
    mAP = run_eval(args.queries, args.groundtruth, args.index_features, whitening_params, args.out, agg_fn, qe_fn)
    print 'mAP: %f' % mAP

    exit(0)

