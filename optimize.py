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


SMALL = 1e-16

pow2 = lambda x: x * x


def probability_to_grow(weights, terrain, tree):
    """This reproduces the model from terrain_affinity.cc."""
    sigma_fertility = (1. - tree['pickiness'])
    sigma_humidity = (1. - tree['pickiness'])
    sigma_temperature = (1. - tree['pickiness'])

    return math.exp((
        -pow2((tree['preferred_fertility'] - terrain['fertility']) / (weights[0] * sigma_fertility))
        -pow2((tree['preferred_humidity'] - terrain['humidity']) / (weights[1] * sigma_humidity))
        -pow2((tree['preferred_temperature'] - terrain['temperature']) / (weights[2] * sigma_temperature))) / 2)

def difference(weights, terrains, trees, verbose=False):
    """Returns the squared differences of the probabilities in the
    TRUTH table and the calculated parameters using the parameters
    in 'terrains' and 'trees'. """
    squared_diffs = 0.
    max_diff, min_diff = -100, 100
    for tree_name, terrains_for_tree in TRUTH.items():
        for terrain_name, b18_probability in terrains_for_tree.items():
            p = probability_to_grow(weights, terrains[terrain_name], trees[tree_name])
            diff = p - b18_probability
            if verbose:
                max_diff = max(max_diff, diff)
                min_diff = min(min_diff, diff)
                print "%s %s %.2f %.2f (%.2f)" % (
                        tree_name, terrain_name, b18_probability, p, diff)
            squared_diffs += pow2(diff)
    if verbose:
        print "%.2f %.2f" % (min_diff, max_diff)
    print "squared_diffs: %r" % (squared_diffs)
    return squared_diffs

def parameters_to_dictionary(parameters):
    trees = {}
    terrains = {}
    idx = 3
    weights = parameters[:idx]
    for terrain_name in sorted(TERRAINS.keys()):
        d = {}
        d['fertility'] = parameters[idx]
        idx += 1
        d['humidity'] = parameters[idx]
        idx += 1
        d['temperature'] = parameters[idx]
        idx += 1
        terrains[terrain_name] = d

    for tree_name in sorted(TREES.keys()):
        d = {}
        d['preferred_fertility'] = parameters[idx]
        idx += 1
        d['preferred_humidity'] = parameters[idx]
        idx += 1
        d['preferred_temperature'] = parameters[idx]
        idx += 1
        d['pickiness'] = parameters[idx]
        idx += 1
        trees[tree_name] = d
    assert(idx == len(parameters))
    return weights, terrains, trees

def main():
    parameters = np.empty(len(TREES) * 4 + len(TERRAINS) * 3 + 3)
    bounds = []

    parameters[0] = 0.25
    bounds.append((SMALL, 1000))
    parameters[1] = 0.25
    bounds.append((SMALL, 1000))
    parameters[2] = 12.5
    bounds.append((SMALL, 1000))

    idx = 3
    for terrain_name in sorted(TERRAINS.keys()):
        parameters[idx] = TERRAINS[terrain_name]['fertility']
        idx += 1
        bounds.append((SMALL, 1 - SMALL))

        parameters[idx] = TERRAINS[terrain_name]['humidity']
        idx += 1
        bounds.append((SMALL, 1 - SMALL))

        parameters[idx] = TERRAINS[terrain_name]['temperature']
        idx += 1
        bounds.append((223, 1300.00))

    for tree_name in sorted(TREES.keys()):
        parameters[idx] = TREES[tree_name]['preferred_fertility']
        idx += 1
        bounds.append((SMALL, 1 - SMALL))

        parameters[idx] = TREES[tree_name]['preferred_humidity']
        idx += 1
        bounds.append((SMALL, 1 - SMALL))

        parameters[idx] = TREES[tree_name]['preferred_temperature']
        idx += 1
        # freezing cold to very hot
        bounds.append((223, 343.15))

        parameters[idx] = TREES[tree_name]['pickiness']
        idx += 1
        bounds.append((SMALL, 1 - SMALL))
    assert(idx == len(parameters))

    def minimize(params):
        weights, terrains, trees = parameters_to_dictionary(params)
        return difference(weights, terrains, trees)

    # final = scipy.optimize.fmin(minimize, parameters)
    result = scipy.optimize.fmin_l_bfgs_b(
            minimize,
            parameters,
            approx_grad = True,
            bounds = bounds,
            maxfun = 15000000,
            maxiter = 15000000
            )
    print "#sirver result: %r\n" % (result,)
    final = result[0]
    # final = scipy.optimize.minimize(
            # minimize, parameters, bounds=bounds).x
    # final = scipy.optimize.differential_evolution(
            # minimize, bounds, popsize=25).x
    final_weights, final_terrains, final_trees = parameters_to_dictionary(final)
    json.dump(final_terrains, open("final_terrains.json", "w"), indent=4)
    json.dump(final_trees, open("final_trees.json", "w"), indent=4)
    json.dump(final_weights.tolist(), open("final_weights.json", "w"), indent=4)
    print difference(final_weights, final_terrains, final_trees, verbose=True)

def verify():
    final_weights = json.load(open('optimize/final_weights.json', 'r'))
    final_terrains = json.load(open('optimize/final_terrains.json', 'r'))
    final_trees = json.load(open('optimize/final_trees.json', 'r'))

    print difference(final_weights, final_terrains, final_trees, verbose=True)

if __name__ == '__main__':
    main()
