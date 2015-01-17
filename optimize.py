#!/usr/bin/env python
# encoding: utf-8

import json
import math

import numpy as np
import scipy.optimize

# The TRUTH table was extracted from b18, the terrains and trees renamed
# according to the one_world_legacy_lookup_table.cc. Our aim is to find
# parameters for the trees that approximate these values. Given that the
# terrains parameters make sense physically, the remaining values should be
# sensible.
TRUTH = json.load(open('./truth_table.json', 'r'))

# These are the parameters as they were in trunk revision 7357.
TERRAINS = json.load(open('./terrains.json', 'r'))
TREES = json.load(open('./trees.json', 'r'))

pow2 = lambda x: x * x

def probability_to_grow(terrain, tree):
    """This reproduces the model from terrain_affinity.cc."""
    sigma_fertility = (1. - tree['pickiness']) * 0.25 + 1e-2
    sigma_humidity = (1. - tree['pickiness']) * 0.25 + 1e-2
    sigma_temperature = (1. - tree['pickiness']) * 12.5 + 1e-1

    pure_gauss = math.exp(
        -pow2(tree['preferred_fertility'] - terrain['fertility']) / (2 * pow2(sigma_fertility)) -
        pow2(tree['preferred_humidity'] - terrain['humidity']) / (2 * pow2(sigma_humidity)) -
        pow2(tree['preferred_temperature'] - terrain['temperature']) / (2 * pow2(sigma_temperature)))

    advanced_gauss = pure_gauss * 1.1 + 0.05
    if advanced_gauss > 0.95:
        advanced_gauss = 0.95
    return advanced_gauss

def difference(trees, verbose=False):
    """Returns the squared differences of the probabilities in the TRUTH table
    and the calculated parameters using the parameters in 'trees' """
    squared_diffs = 0.
    for tree_name, terrains_for_tree in TRUTH.items():
        for terrain_name, b18_probability in terrains_for_tree.items():
            p = probability_to_grow(TERRAINS[terrain_name], trees[tree_name])
            diff = p - b18_probability
            if verbose:
                print "%s %s %.2f %.2f (%.2f)" % (
                        tree_name, terrain_name, b18_probability, p, diff)
            squared_diffs += pow2(diff)
    print "squared_diffs: %r" % (squared_diffs)
    return squared_diffs

def parameters_to_dictionary(parameters):
    rv = {}
    for idx, tree_name in enumerate(TREES):
        d = {}
        d['preferred_fertility'] = parameters[idx * 4 + 0]
        d['preferred_humidity'] = parameters[idx * 4 + 1]
        d['preferred_temperature'] = parameters[idx * 4 + 2]
        d['pickiness'] = parameters[idx * 4 + 3]
        rv[tree_name] = d
    return rv

def main():
    parameters = np.empty(len(TREES) * 4)
    for idx, tree_name in enumerate(TREES):
        parameters[idx * 4 + 0] = TREES[tree_name]['preferred_fertility']
        parameters[idx * 4 + 1] = TREES[tree_name]['preferred_humidity']
        parameters[idx * 4 + 2] = TREES[tree_name]['preferred_temperature']
        parameters[idx * 4 + 3] = TREES[tree_name]['pickiness']

    def minimize(params):
        return difference(parameters_to_dictionary(params))

    final = scipy.optimize.fmin(minimize, parameters)
    final_dictionary = parameters_to_dictionary(final)
    json.dump(final_dictionary, open("final_parameters.json", "w"), indent=4)

    print difference(final_dictionary, verbose=True)

if __name__ == '__main__':
    main()
