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
import sys


def _find_indices(values, container):
    for i, val in enumerate(container):
        if val in values:
            yield i


def _delete_indices(indices, container):
    for index in sorted(indices, reverse=True):
        del container[index]


def _extract_values(values, container):
    indices = _find_indices(values, container)
    values  = [container[i] for i in indices]
    _delete_indices(indices, container)
    return values


def normalize_args(args):
    """ Iterate over a normalized version of a command line argument list.

    Any argument of the form -name=value will be split into -name and value.
    All other elements are kept as is.
    """
    for arg in args:
        if not arg.startswith('-'):
            yield arg
        else:
            name, sep, value = arg.partition('=')
            yield name
            if sep:
                yield value


def find_indices(names, args):
    """ Find all indices in the argument list where the given names match.

    The argument list must be normalized by a call to normalize_args() already.
    """
    option_parser = argparse.ArgumentParser(add_help=False)
    option_parser.add_argument(*names, dest='option', action='count')
    for i, arg in enumerate(args):
        parsed, _ = option_parser.parse_known_args([arg])
        if parsed.option:
            yield i


class Argument:
    """ Class representing a command line argument that can be registered with a ArgumentParser. """
    def __init__(self, names, kwargs):
        self.names  = names
        self.name   = names[0]
        self.kwargs = kwargs

    def as_args_kwargs(self):
        """ Convert to args/kwargs to be passed to ArgumentParser.add_argument(). """
        return self.names, self.kwargs


class NamedSubparserArgument:
    """ Class representing a named subparser that can be registered with an ArgumentParser. """
    def __init__(self, names, subparser, dest=None, required=False, repeated=False, help=None):
        self.names     = names
        self.name      = self.names[0]
        self.subparser = subparser
        self.dest      = dest if dest is not None else self.name.lstrip('-').replace('-', '_')
        self.required  = required
        self.repeated  = repeated
        self.help      = help

    def as_args_kwargs(self):
        """ Convert to args/kwargs to be passed to ArgumentParser.add_argument().

        The args/kwargs returned are only for documentation purposes.
        The actual parsing needs more support from the ArgumentParser.
        Use add_named_subparser() for that.
        """
        return self.names, dict(
            choices  = self.subparser.choices(),
            dest     = self.dest,
            required = self.required,
            action   = 'append' if self.repeated else 'store',
            help     = self.help,
        )


class NamedSubparser:
    """ A named subparser acts as an enum argument.
    Each enum value is associated with another argument parser that will
    be used to parser the remaining arguments.

    It is up to the caller to split and normalize the argument list before feeding
    it to named_subparser.parse().
    """
    def __init__(self):
        """ Create a new named subparser. """
        self.options  = {}

    def choices(self):
        """ Get the allowed choices for the enum values.

        Each option added with add_option() will become one choice.
        """
        return list(self.options.keys())

    def add_option(self, value, parser):
        """ Add a option for the subparser. """
        if value in self.options:
            raise KeyError("subparser option `{}' is already defined".format(value))
        self.options[value] = parser

    def parse(self, names, args):
        """ Parse the arguments using the this named subparser.

        The arguments must be normalized by a call to normalize_args() already.

        The argument list should contain exactly one argument with any of the name from `names` with the enum value,
        followed by the arguments for the selected subparser.

        This function will parse all arguments given to it, so any splitting has to be done before calling this function.
        """
        option_parser = argparse.ArgumentParser(add_help=False)
        option_parser.add_argument(*names, dest='type', choices=self.choices())
        namespace, _ = option_parser.parse_known_args(args[:2])
        selected     = namespace.type
        namespace, unknown = self.options[selected].parse_known_args(args[2:])
        return SubNamespace(selected, namespace), unknown


class SubNamespace:
    """ Sub-namspace parsed by a named subparser. """
    def __init__(self, name, args):
        self.name = name
        self.args = args

    def __repr__(self):
        return 'SubNamespace(name={}, args={})'.format(repr(self.name), repr(self.args))


class ArgumentParser(argparse.ArgumentParser):
    """ ArgumentParser that supports named subparsers. """
    def __init__(self, *args, **kwargs):
        self.__args               = args
        self.__kwargs             = dict(kwargs)
        self.__add_help           = kwargs.get('add_help', True)
        self.__subparsers         = {}
        self.__doc_arguments      = []

        kwargs['add_help'] = False
        super(ArgumentParser, self).__init__(*args, **kwargs)

    def __with_named_subparsers(self):
        result = argparse.ArgumentParser(*self.__args, **self.__kwargs)
        for doc_arg in self.__doc_arguments:
            args, kwargs = doc_arg.as_args_kwargs()
            result.add_argument(*args, **kwargs)
        return result

    def print_help(self, file=None):
        # TODO: Print subparser options help
        return self.__with_named_subparsers().print_help(file=file)

    def print_usage(self, file=None):
        return self.__with_named_subparsers().print_usage(file=file)

    def add_argument(self, *args, **kwargs):
        # Keep a copy of each argument so we can rebuild a parser for the help text,
        # with all arguments in the right order.
        self.__doc_arguments.append(Argument(args, kwargs))
        super(ArgumentParser, self).add_argument(*args, **kwargs)

    def add_named_subparser(self, names, subparser=None, dest=None, required=False, repeated=False, help=None):
        name = names[0]
        if name in self.__subparsers:
            raise KeyError("named subparser `{}' is already defined".format(name))
        if subparser is None:
            subparser = NamedSubparser()

        subparser_arg = NamedSubparserArgument(names, subparser=subparser, dest=dest, required=required, repeated=repeated, help=help)
        self.__subparsers[name] = subparser_arg
        self.__doc_arguments.append(subparser_arg)
        return subparser

    def __match_subparsers(self, args):
        """
        Build a list of matched and unmatched subparsers.
        Each subparser will be listed once for each match in the argument list.
        """
        matched   = []
        unmatched = []
        for subparser_arg in self.__subparsers.values():
            indices = list(find_indices(subparser_arg.names, args))
            if not indices:
                unmatched.append(subparser_arg)
                continue
            for index in indices:
                matched.append((subparser_arg, index))
        return matched, unmatched

    def __parse_known(self, args):
        matches, unmatched = self.__match_subparsers(args)
        for subparser_arg in unmatched:
            yield subparser_arg, None, None

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
        for subparser_arg, start, end in matches:
            sub_namespace, unknown = subparser_arg.subparser.parse(subparser_arg.names, args[start:end])
            yield subparser_arg, sub_namespace, unknown

    def parse_known_args(self, args=None, namespace=None):
        namespace, unknown = super(ArgumentParser, self).parse_known_args(args, namespace)
        unknown            = list(normalize_args(unknown))
        leftover           = []

        # Create a subparser with options for the names subparser,
        # so we get the correct help output with errors.
        with_named_subparsers = self.__with_named_subparsers()

        # Set the default value for each named subparser.
        for subparser_arg in self.__subparsers.values():
            if subparser_arg.repeated:
                setattr(namespace, subparser_arg.dest, [])
            else:
                setattr(namespace, subparser_arg.dest, None)

        # Process all named subparsers.
        count = {}
        for subparser_arg, sub_namespace, unknown in self.__parse_known(unknown):
            if unknown:
                leftover += unknown
            if subparser_arg is None:
                continue
            if sub_namespace is None:
                if subparser_arg.required:
                    with_named_subparsers.error('missing required argument: {}'.format(subparser_arg.name))
                continue

            count.setdefault(subparser_arg.dest, 0)
            count[subparser_arg.dest] += 1
            if subparser_arg.repeated:
                getattr(namespace, subparser_arg.dest).append(sub_namespace)
            else:
                if count[subparser_arg.dest] > 1:
                    with_named_subparsers.error('duplicate option: {}'.format(subparser_arg.name))
                setattr(namespace, subparser_arg.dest, sub_namespace)

        # If none of the subparsers grabbed help, grab it.
        if self.__add_help and _extract_values(['--help', '-h'], leftover):
            self.print_help()
            sys.exit(0)

        return namespace, leftover
