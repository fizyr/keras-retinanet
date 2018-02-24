#!/usr/bin/env python

import argparse
from keras_retinanet.utils import named_subparser


def make_parser():
    parser    = named_subparser.ArgumentParser()
    subparser = named_subparser.NamedSubparser('--food', '-f', required=True, repeated=False)
    parser.add_named_subparser(subparser)

    aap = argparse.ArgumentParser(prog='aap')
    aap.add_argument('--tail', action='store_true')
    aap.add_argument('--name')
    subparser.add_option('aap', aap)

    noot = argparse.ArgumentParser(prog='noot')
    noot.add_argument('--type', choices=['walnoot', 'cashew'], required=True)
    noot.add_argument('--size')
    subparser.add_option('noot', noot)

    return parser


def test_simple():
    parser = make_parser()

    parsed, unknown = parser.parse_known_args(['--food', 'aap', '--name', 'Sjaak', '--tail'])
    print(parsed)
    print(parsed.food)
    assert parsed.food.name == 'aap'
    assert parsed.food.args.name == 'Sjaak'
    assert parsed.food.args.tail
    assert not unknown

    parsed, unknown = parser.parse_known_args(['--food', 'aap', '--name', 'Henkie'])
    print(parsed)
    print(parsed.food)
    assert parsed.food.name == 'aap'
    assert parsed.food.args.name == 'Henkie'
    assert not parsed.food.args.tail
    assert not unknown

    parsed, unknown = parser.parse_known_args(['--food', 'noot', '--type', 'walnoot'])
    assert parsed.food.name == 'noot'
    assert parsed.food.args.type == 'walnoot'
    assert parsed.food.args.size is None
    assert not unknown


if __name__ == '__main__':
    print(repr(make_parser().parse_known_args()))
