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

import csv
import pytest
try:
    from io import StringIO
except ImportError:
    from stringio import StringIO

from keras_retinanet.preprocessing import csv_generator


def csv_str(string):
    if str == bytes:
        string = string.decode('utf-8')
    return csv.reader(StringIO(string))


def annotation(x1, y1, x2, y2, class_name):
    return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'class': class_name}


def test_read_classes():
    assert csv_generator._read_classes(csv_str('')) == {}
    assert csv_generator._read_classes(csv_str('a,1')) == {'a': 1}
    assert csv_generator._read_classes(csv_str('a,1\nb,2')) == {'a': 1, 'b': 2}


def test_read_classes_wrong_format():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,b,c'))
        except ValueError as e:
            assert str(e).startswith('line 1: format should be')
            raise
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,1\nb,c,d'))
        except ValueError as e:
            assert str(e).startswith('line 2: format should be')
            raise


def test_read_classes_malformed_class_id():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,b'))
        except ValueError as e:
            assert str(e).startswith("line 1: malformed class ID:")
            raise

    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,1\nb,c'))
        except ValueError as e:
            assert str(e).startswith('line 2: malformed class ID:')
            raise


def test_read_classes_duplicate_name():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_classes(csv_str('a,1\nb,2\na,3'))
        except ValueError as e:
            assert str(e).startswith('line 3: duplicate class name')
            raise


def test_read_annotations():
    classes = {'a': 1, 'b': 2, 'c': 4, 'd': 10}
    annotations = csv_generator._read_annotations(csv_str(
        'a.png,0,1,2,3,a'     '\n'
        'b.png,4,5,6,7,b'     '\n'
        'c.png,8,9,10,11,c'   '\n'
        'd.png,12,13,14,15,d' '\n'
    ), classes)
    assert annotations == {
        'a.png': [annotation( 0,  1,  2,  3, 'a')],
        'b.png': [annotation( 4,  5,  6,  7, 'b')],
        'c.png': [annotation( 8,  9, 10, 11, 'c')],
        'd.png': [annotation(12, 13, 14, 15, 'd')],
    }


def test_read_annotations_multiple():
    classes = {'a': 1, 'b': 2, 'c': 4, 'd': 10}
    annotations = csv_generator._read_annotations(csv_str(
        'a.png,0,1,2,3,a'     '\n'
        'b.png,4,5,6,7,b'     '\n'
        'a.png,8,9,10,11,c'   '\n'
    ), classes)
    assert annotations == {
        'a.png': [
            annotation(0, 1,  2,  3, 'a'),
            annotation(8, 9, 10, 11, 'c'),
        ],
        'b.png': [annotation(4, 5, 6, 7, 'b')],
    }


def test_read_annotations_wrong_format():
    classes = {'a': 1, 'b': 2, 'c': 4, 'd': 10}
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,1,2,3,a'), classes)
        except ValueError as e:
            assert str(e).startswith("line 1: format should be")
            raise

    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str(
                'a.png,0,1,2,3,a' '\n'
                'a.png,1,2,3,a'   '\n'
            ), classes)
        except ValueError as e:
            assert str(e).startswith("line 2: format should be")
            raise


def test_read_annotations_wrong_x1():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,a,0,1,2,a'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: malformed x1:")
            raise


def test_read_annotations_wrong_y1():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,0,a,1,2,a'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: malformed y1:")
            raise


def test_read_annotations_wrong_x2():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,0,1,a,2,a'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: malformed x2:")
            raise


def test_read_annotations_wrong_y2():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,0,1,2,a,a'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: malformed y2:")
            raise


def test_read_annotations_wrong_class():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,0,1,2,3,g'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: unknown class name:")
            raise


def test_read_annotations_invalid_bb_x():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,1,2,1,3,g'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: x2 (1) must be higher than x1 (1)")
            raise
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,9,2,5,3,g'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: x2 (5) must be higher than x1 (9)")
            raise


def test_read_annotations_invalid_bb_y():
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,1,2,3,2,a'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: y2 (2) must be higher than y1 (2)")
            raise
    with pytest.raises(ValueError):
        try:
            csv_generator._read_annotations(csv_str('a.png,1,8,3,5,a'), {'a': 1})
        except ValueError as e:
            assert str(e).startswith("line 1: y2 (5) must be higher than y1 (8)")
            raise


def test_read_annotations_empty_image():
    # Check that images without annotations are parsed.
    assert csv_generator._read_annotations(csv_str('a.png,,,,,\nb.png,,,,,'), {'a': 1}) == {'a.png': [], 'b.png': []}

    # Check that lines without annotations don't clear earlier annotations.
    assert csv_generator._read_annotations(csv_str('a.png,0,1,2,3,a\na.png,,,,,'), {'a': 1}) == {'a.png': [annotation(0, 1,  2,  3, 'a')]}
