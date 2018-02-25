#!/usr/bin/env python

import argparse
from keras_retinanet.utils.named_subparser import ArgumentParser, NamedSubparser


def make_parser():
    food_parser = NamedSubparser()

    aap = argparse.ArgumentParser(prog='aap')
    aap.add_argument('--tail', action='store_true')
    aap.add_argument('--name')
    food_parser.add_option('aap', aap)

    noot = argparse.ArgumentParser(prog='noot')
    noot.add_argument('--type', choices=['walnoot', 'cashew'], required=True)
    noot.add_argument('--size')
    food_parser.add_option('noot', noot)

    parser = ArgumentParser()
    parser.add_argument('--global1')
    parser.add_named_subparser(['--food', '-f'], subparser=food_parser, required=True, repeated=True, help="What to eat?")
    parser.add_argument('--global2')

    return parser


def test_simple():
    parser = make_parser()

    parsed, unknown = parser.parse_known_args(['--food', 'aap', '--name', 'Sjaak', '--tail'])
    print(parsed)
    print(parsed.food)
    assert len(parsed.food) == 1
    assert parsed.food[0].name == 'aap'
    assert parsed.food[0].args.name == 'Sjaak'
    assert parsed.food[0].args.tail
    assert not unknown

    parsed, unknown = parser.parse_known_args(['--food', 'aap', '--name', 'Henkie'])
    print(parsed)
    print(parsed.food)
    assert len(parsed.food) == 1
    assert parsed.food[0].name == 'aap'
    assert parsed.food[0].args.name == 'Henkie'
    assert not parsed.food[0].args.tail
    assert not unknown

    parsed, unknown = parser.parse_known_args(['--food', 'noot', '--type', 'walnoot'])
    assert len(parsed.food) == 1
    assert parsed.food[0].name == 'noot'
    assert parsed.food[0].args.type == 'walnoot'
    assert parsed.food[0].args.size is None
    assert not unknown


if __name__ == '__main__':
    print(repr(make_parser().parse_known_args()))
