"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse


class NamedSubparser:
    def __init__(self, *args, **kwargs):
        self.name     = args[0]
        self.names    = args
        self.dest     = kwargs.get('dest', self.name.lstrip('-').replace('-', '_'))
        self.required = kwargs.get('required', False)
        self.repeated = kwargs.get('repeated', False)
        self.help     = kwargs.get('help', None)
        self.options  = {}

    def choices(self):
        return list(self.options.keys())

    def add_option(self, value, parser):
        if value in self.options:
            raise KeyError("subparser option `{}' is already defined".format(value))
        self.options[value] = parser

    def find_indices(self, args):
        option_parser = argparse.ArgumentParser()
        option_parser.add_argument(*self.names, dest='option', action='store_true')
        for i, arg in enumerate(args):
            parsed, _ = option_parser.parse_known_args([arg])
            if parsed.option:
                yield i

    def parse(self, args):
        option_parser = argparse.ArgumentParser()
        option_parser.add_argument(*self.names, dest='type', choices=self.choices())
        namespace, _ = option_parser.parse_known_args(args[:2])
        selected     = namespace.type
        namespace, unknown = self.options[selected].parse_known_args(args[2:])
        return SubNamespace(selected, namespace), unknown


class SubNamespace:
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return 'SubNamespace(name={}, args={})'.format(repr(self.name), repr(self.args))


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.__super().__init__(*args, **kwargs)
        self.__args               = args
        self.__kwargs             = kwargs
        self.__kwargs['parents']  = [self]
        self.__kwargs['add_help'] = False
        self.__subparsers         = {}

    def __super(self):
        return super(ArgumentParser, self)

    def print_help(self, file=None):
        # TODO: Print subparser options help
        return self.__with_named_subparsers().print_help(file=file)

    def print_usage(self, file=None):
        return self.__with_named_subparsers().print_usage(file=file)

    def __with_named_subparsers(self):
        result = argparse.ArgumentParser(*self.__args, **self.__kwargs)
        for subparses in self.__subparsers.values():
            result.add_argument(subparses.name, choices=subparses.choices(), help=subparses.help)
        return result

    def add_named_subparser(self, subparser):
        if subparser.name in self.__subparsers:
            raise KeyError("named subparser `{}' is already defined".format(name))
        self.__subparsers[subparser.name] = subparser
        return subparser

    def __match_subparsers(self, args):
        """
        Build a list of matched and unmatched subparsers.
        Each subparser will be listed once for each match in the argument list.
        """
        matched   = []
        unmatched = []
        for subparser in self.__subparsers.values():
            indices = list(subparser.find_indices(args))
            if not indices:
                unmatched.append(subparser)
                continue
            for index in indices:
                matched.append((subparser, index))
        return matched, unmatched

    def __parse_known(self, args):
        matches, unmatched = self.__match_subparsers(args)
        for subparser in unmatched:
            yield subparser, None, None

        # If the first match is not the first argument,
        # everthing up to the match is unknown.
        first_match = matches[0][1]
        if first_match != 0:
            yield None, None, args[:first_match]

        # Add end index to each entry.
        for i in range(len(matches) - 1):
            matches[i] = matches[i][0], matches[i][1], matches[i + 1][1]
        matches[-1] = matches[-1][0], matches[-1][1], len(args)

        # Let all subparsers parse their arguments.
        matches = sorted(matches, key=lambda x: x[1])
        for subparser, start, end in matches:
            sub_namespace, unknown = subparser.parse(args[start:end])
            yield subparser, sub_namespace, unknown

    def parse_known_args(self, args=None, namespace=None):
        namespace, unknown = self.__super().parse_known_args(args, namespace)
        leftover           = []

        with_named_subparsers = self.__with_named_subparsers()

        # Set the default value for each named subparser.
        for subparser in self.__subparsers.values():
            if subparser.repeated:
                setattr(namespace, subparser.dest, [])
            else:
                setattr(namespace, subparser.dest, None)

        # Process all named subparsers.
        count = {}
        for subparser, sub_namespace, unknown in self.__parse_known(unknown):
            if unknown:
                leftover += unknown
            if subparser is None:
                continue
            if sub_namespace is None:
                if subparser.required:
                    with_named_subparsers.error('missing required argument: {}'.format(subparser.name))
                continue

            count.setdefault(subparser.dest, 0)
            count[subparser.dest] += 1
            if subparser.repeated:
                getattr(namespace, subparser.dest).append(sub_namespace)
            else:
                if count[subparser.dest] > 1:
                    with_named_subparsers.error('duplicate option: {}'.format(subparser.name))
                setattr(namespace, subparser.dest, sub_namespace)

        return namespace, leftover
