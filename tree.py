"""Binary ID3 decision tree.

All the features including a target feature is binary."""

import collections
import functools
import itertools
import math
import types


class Tree(object):
    """Just a tree to make decisions."""
    def __init__(self, node, target):
        self.root_node = node
        self.target = target

    def make_decision(self, unclassified, node=None):
        """Decision process itself."""
        node = node or self.root_node
        try:
            gt_thr = unclassified[node['key']] >= node['threshold']
            if gt_thr is node['left_val']:
                return self.make_decision(unclassified, node['left'])
            elif gt_thr is node['right_val']:
                return self.make_decision(unclassified, node['right'])
        except KeyError:
            return node[self.target]
        raise ValueError('Invalid predicate value.')


def function_behaviour(class_):
    """Make a callable class behave as factory function"""
    def create_tree(learning_data, target):
        ct = class_(learning_data, target)
        return ct()
    create_tree.__decorated__ = class_ # Keep a class for more transparent view
    return create_tree


@function_behaviour
class create_tree(object):
    """Decision tree controller object.

    Control nodes. Maintain learning process and making decisions process."""


    def __init__(self, learning_data, target):
        """Sort a training data relatively to a target feature.
        Sort a training data relatively to a target feature to determine
        the most bound features: the smaller entropy at the data sorted
        relative to the target feature, the more bound the feature to the
        target."""

        self._verify_data(learning_data)

        self.target = target
        self.keys = self._get_verified_keys(learning_data)
        self.length = len(learning_data)
        self.thresholds = self._get_thresholds(learning_data)
        self.discret_data = self._get_discret_data(learning_data)
        self.data = learning_data
        self.root_node = None
        self._by_entropy = lambda x: x[0]

    def __call__(self):
        self._learn()
        return Tree(self.root_node, self.target)

    def _get_thresholds(self, data):
        keys = self.keys + [self.target]
        val_sum = reduce(
                lambda x, y: {k: float(x[k] + y[k]) for k in keys}, data)
        return {k: v/self.length for k, v in val_sum.iteritems()}

    @staticmethod
    def _verify_data(data):
        for i in set(itertools.chain(*(i.itervalues() for i in data))):
            if not isinstance(i, (types.IntType, types.FloatType)):
                raise ValueError(
                        'Inconsistent data: data is not numeric.')

    def _get_discret_data(self, data):
        """Check is data consistent."""
        to_disc = lambda x: {
                k: int(v >= self.thresholds[k]) for k, v in x.iteritems()}
        data = map(to_disc, data)
        return sorted(data, key=lambda x: x[self.target])

    def _get_verified_keys(self, data):
        """Check for data consistency and return keys."""
        try:
            keys = set(tuple(i.keys()) for i in data)
        except Exception:
            import pdb; pdb.set_trace()
        if len(keys) == 1:
            keys = list(keys.pop())
            keys.remove(self.target)
            return keys
        raise ValueError('Inconsistent data: the items have different keys.')

    def _get_probability(self, key, from_=None, to=None):
        """Get probability for a different values of a key on a slice."""
        if from_ is not None or to is not None:
            the_slice = self.discret_data[from_:to]
        else:
            the_slice = self.discret_data

        counter = {0: 0, 1: 0}
        for i in the_slice:
            if i[key] == 0:
                counter[0] += 1
            else:
                counter[1] += 1

        for i in 0, 1:
            yield counter[i] / float(len(the_slice))

        #the_slice = [i[key] for i in the_slice]
        #the_slice.sort()
        #slice_values = set(the_slice)
        #len_slice = len(the_slice)

        #if len(slice_values) == 1:
            #for i in (0, 1):
                #yield int(i == next(iter(slice_values)))
        #else:
            #_from = 0
            #t0 = len_slice
            #delim = t0 / 2
            #while not (the_slice[delim] == 1 and the_slice[delim-1] == 0):
                #if the_slice[delim] == 1:
                    #t0 = delim
                    #delim -= (t0 - _from)/2
                #elif the_slice[delim] == 0:
                    #_from = delim
                    #delim += (t0 - _from)/2

            #len_slice = float(len_slice)
            #for i in (delim, len_slice - delim):
                #yield i / len_slice

    def _count_entropy(self, key, from_, to):
        """Count Shannon entropy for a key on a slice."""
        probs = self._get_probability(key, from_, to)
        try:
            return sum(map(lambda p: -(p * math.log(p, 2)), probs))
        except ValueError:
            return None

    def _average_entropy(self, key, from_, to):
        """Average entropy for on a slice for a key."""
        def count(delimeter):
            entropy = filter(None,
                    (self._count_entropy(key, from_, delimeter),
                     self._count_entropy(key, delimeter, to))
                    )
            if len(entropy) == 1:
                return entropy[0], delimeter
            elif not entropy:
                return 0, delimeter
            else:
                return sum(entropy) / 2.0, delimeter
        return count

    def _min_index(self, from_, to):
        """Count average entropy for all allowed slices.

        Returns a minimum average entropy index and a prevailing
        values of the right and left side by the target key."""

        def count(key):
            ave_entropy = map(
                    self._average_entropy(key, from_, to),
                    xrange(from_+1, to-1))
            entp_dlm = min(ave_entropy, key=self._by_entropy)
            return entp_dlm + (key, )
        return count

    def _min_key(self, from_, to):
        """Key with a minimal entropy for a slice."""
        keys_by_entp = map(self._min_index(from_, to), self.keys)
        return min(keys_by_entp, key=self._by_entropy)

    def _min_leaf(self, leaf):
        """leaf node with a minimal entropy."""
        return self._min_key(leaf['from'], leaf['to']) + (leaf, )

    def _get_feature_values(self, key, from_, to, index):
        """The most probable value for a key on a node."""
        sum_by_key = lambda x, y: {key: x[key]+y[key]}
        left = reduce(sum_by_key, self.discret_data[from_:index])
        right = reduce(sum_by_key, self.discret_data[index:to])
        left_prob = left[key] / float(index - from_)
        right_prob = right[key] / float(to - index)
        left_v, right_v = (left_prob > right_prob, right_prob > left_prob)
        return left_v, right_v

    @staticmethod
    def _if_splitable(leaf):
        """Filter to determine a leafs able for further split."""
        return leaf['to'] - leaf['from'] > 3 and not 'leaf' in leaf

    def _learn(self):
        """Learn process itself."""
        self.root_node = {'from': 0, 'to': self.length}
        leafs = [self.root_node]

        while self.keys:
            splitable = map(self._min_leaf, filter(self._if_splitable, leafs))
            entropy, index, key, leaf = min(splitable, key=self._by_entropy)
            left_v, right_v = self._get_feature_values(
                    key, leaf['from'], leaf['to'], index)
            self.keys.remove(key)

            if entropy == 0 or left_v == right_v:
                leaf['leaf'] = True
            else:
                leaf['key'] = key
                leaf['threshold'] = self.thresholds[key]
                leaf['left'] = {'from': leaf['from'], 'to': index}
                leaf['right'] = {'from': index, 'to': leaf['to']}
                leaf['left_val'] = left_v
                leaf['right_val'] = right_v

                for branch in ('left', 'right'):
                    leafs.append(leaf[branch])

                leafs.remove(leaf)
                del leaf['from']
                del leaf['to']

        for leaf in leafs:
            leaf_data = self.data[leaf['from']: leaf['to']]
            by_target = lambda x: x[self.target]
            leaf[self.target] = (
                    min(leaf_data, key=by_target)[self.target],
                    max(leaf_data, key=by_target)[self.target])
            del leaf['from']
            del leaf['to']
