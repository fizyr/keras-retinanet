import numpy as np

from keras_retinanet.preprocessing.csv_generator import CSVGenerator, Generator


class DfGenerator(CSVGenerator):
    """Custom generator intented to work with in-memory Pandas' dataframe."""
    def __init__(self, df, class_mapping, cols, base_dir='', **kwargs):
        """Initialization method.

        Arguments:
            df: Pandas DataFrame containing paths, labels, and bounding boxes.
            class_mapping: Dict mapping label_str to id_int.
            cols: Dict Mapping 'col_{filename/label/x1/y1/x2/y2} to corresponding df col.
        """
        self.base_dir = base_dir
        self.cols = cols
        self.classes = class_mapping
        self.labels = {v: k for k, v in self.classes.items()}

        self.image_data = self._read_data(df)
        self.image_names = list(self.image_data.keys())

        Generator.__init__(self, **kwargs)

    def _read_classes(self, df):
        return {row[0]: row[1] for _, row in df.iterrows()}

    def __len__(self):
        return len(self.image_names)

    def _read_data(self, df):
        data = {}
        for _, row in df.iterrows():
            img_file, class_name = row[self.cols['col_filename']], row[self.cols['col_label']]
            x1, y1 = row[self.cols['col_x1']], row[self.cols['col_y1']]
            x2, y2 = row[self.cols['col_x2']], row[self.cols['col_y2']]

            if img_file not in data:
                data[img_file] = []

            # Image without annotations
            if not isinstance(class_name, str) and np.isnan(class_name):
                continue

            data[img_file].append({
                'x1': int(x1), 'x2': int(x2),
                'y1': int(y1), 'y2': int(y2),
                'class': class_name
            })

        return data
