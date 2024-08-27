import json
import sys
import re
import numpy as np
import os
import itertools
import polars as pl

from typing import (
    Any,
    Iterable,
)

from polars.type_aliases import (
    IntoExprColumn,
)

RESERVED = ("path", "size", "exists")


def parse_filename_patterns(filename_pattern):
    """
    Convenience function for list or string arguments.

    Args:
        filename_pattern (str or list): A filename pattern or a list of filename patterns.

    Returns:
        list: A list of filename patterns.
    """
    if isinstance(filename_pattern, str):
        filename_patterns = [filename_pattern]
    else:
        filename_patterns = filename_pattern
    return filename_patterns


def extract_columns(strings):
    """
    Extract columns used in filename pattern using regex to handle
    adjacent characters.

    Args:
        strings (list): A list of filename patterns.

    Returns:
        list: A list of columns used in the filename patterns.

    Raises:
        ValueError: If all filename patterns do not have the same parameters.
    """
    if not strings:
        return []

    cols = None
    for pattern in strings:
        found_cols = set(re.findall(r"\{(.*?)\}", pattern))
        if cols is None:
            cols = found_cols
        else:
            if cols != found_cols:
                msg = "All filename patterns must have the same parameters."
                raise ValueError(msg)
    # TODO: check reserved?
    return list(cols)


def expand_params(params):
    """
    Create a list of dictionary rows, where all rows are the Cartesian product
    of values for each key.

    Args:
        params (dict): A dictionary where keys are parameter names and values
                       are lists of parameter values.

    Returns:
        list: A list of dictionaries, each representing a combination of parameters.

    Raises:
        ValueError: If any parameter key does not have values.
    """
    keys = list(params.keys())
    values = list(params.values())
    for i, value in enumerate(values):
        if not len(value):
            msg = (
                f"{keys[i]} did not have any values. This would "
                "cause the Cartesian product to be zero."
            )
            raise ValueError(msg)
    product = list(itertools.product(*values))
    expanded_data = [dict(zip(keys, combination)) for combination in product]
    return expanded_data


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def generate_paths(df, filename_patterns, directory=None, sep="__", filename_sep="_"):
    """
    Generate paths based on filename patterns.

    Args:
        data (DataFrame): A Polars DataFrame of all the (likely crossed) experimental combinations.
        filename_patterns (str or list): A string or list of strings representing the filename pattern(s).
        directory (str, optional): The base directory for the generated paths. If not provided, the current directory is used.
        sep (str, optional): The separator used between the key and value in the generated paths. Defaults to "__".

    Yields:
        tuple: A tuple containing the full path and the row dictionary.
    """
    # get the keys in the filename part of path
    filename_patterns = parse_filename_patterns(filename_patterns)
    filename_cols = extract_columns(filename_patterns)

    # get the keys in the directory part of path
    dir_keys = [
        key for key in df.columns if key not in filename_cols and key not in RESERVED
    ]

    df = df.clone()
    if "size" in df.columns:
        # handle the possibility the size column may be all nulls,
        # which messes up sorting
        df = df.with_columns(pl.col("size").fill_null(value=0))
    for row in df.sort(by=df.columns).iter_rows(named=True):
        if directory is None:
            dir_path = (
                ""
                if not dir_keys
                else os.path.join(*[f"{key}{sep}{row[key]}" for key in dir_keys])
            )
        else:
            dir_path = (
                directory
                if not dir_keys
                else os.path.join(
                    directory, *[f"{key}{sep}{row[key]}" for key in dir_keys]
                )
            )

        for pattern in sorted(filename_patterns):
            filename_parts = []
            for col in filename_cols:
                value = row[col]
                filename_part = f"{col}{sep}{value}"
                filename_parts.append(filename_part)
            filename = pattern.format(
                **{col: f"{col}{filename_sep}{row[col]}" for col in filename_cols}
            )
            full_path = os.path.join(dir_path, filename)
            yield full_path, row


def format_size(size):
    """
    Convert a file size to a human-readable format (KB, MB, GB).

    Args:
        size (int): The file size in bytes.

    Returns:
        str: The file size in a human-readable format.
    """
    for unit in ["bytes", "KB", "MB", "GB", "TB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def random_seed(rng=None):
    """
    Generate a random seed value.

    Args:
        rng (numpy.random.Generator, optional): A random number generator
             object. If not provided, numpy.random will be used.

    Returns:
        int: A random seed value.
    """
    # NOTE: borrowed from my other similar package, slimflow
    seed_max = sys.maxsize
    if rng is None:
        return np.random.randint(0, seed_max)
    return rng.integers(0, seed_max)


class GridDf:
    def __init__(self, params: dict, seed: int = None, dir: str = None):
        """
        Initialize the GridDf object.

        Args:
            params (dict): A dictionary of parameters for the grid.
            seed (int, optional): A seed for random number generation. Defaults to None.
        """
        if not isinstance(params, dict):
            msg = (
                "Argument 'params' must be a dict (perhaps "
                "you mean to use GridDf.from_tsv()?)."
            )
            raise ValueError(msg)
        self.params = params
        self.df = None
        self.filename_patterns = None
        self.dir = dir
        self.nreps = None
        self.seed = random_seed(None) if seed is None else seed
        self.rng = np.random.default_rng(self.seed)
        self.sep = None

    @staticmethod
    def from_tsv(filepath, dir=None):
        """
        Load parameters from a TSV file and deduce the parameters.

        Args:
            filepath (str): The path to the TSV file to load.

        Returns:
            GridDf: The updated GridDf object with parameters loaded from the TSV file.
        """
        with open(filepath, "r") as f:
            first_line = f.readline().strip()
            if first_line.startswith("#"):
                serialized_params = first_line.lstrip("# params: ").strip()
                params = json.loads(serialized_params)
            else:
                raise ValueError(
                    "TSV file does not contain serialized parameters as a comment line."
                )
        df = pl.read_csv(
            filepath, separator="\t", skip_rows=1, infer_schema_length=1000
        )
        grid = GridDf(params, dir=dir)
        grid.df = df
        return grid

    def write_tsv(self, filepath):
        """
        Write the current DataFrame to a TSV file with serialized parameters as a comment line.
        Args:
            filepath (str): The path to the TSV file to write.
        """
        self._ensure_cross_generated("write_tsv")
        serialized_params = json.dumps(self.params, cls=NumpyEncoder)
        comment_line = f"# params: {serialized_params}\n"
        with open(filepath, "w") as f:
            f.write(comment_line)
            f.write(self.df.write_csv(separator="\t"))

    def __repr__(self):
        """
        Return a string representation of the GridDf object.

        Returns:
            str: A string representation of the GridDf object.
        """
        param_strs = [
            f"   {key} ∈ {{{', '.join(map(str, values))}}}"
            for key, values in self.params.items()
            if key not in RESERVED
        ]
        param_str = "\n".join(param_strs)
        seed_str = f"Seed: {self.seed}"
        num_params = len(self.params)
        num_replicates = (
            self.df.select(pl.col("replicate")).unique().shape[0]
            if self.df is not None and "replicate" in self.df.columns
            else "NA"
        )

        repr_str = (
            f"GridDf Status:\n"
            f" Total number of parameters: {num_params}\n"
            f" Total number of replicates: {num_replicates}\n"
            f" {seed_str}\n"
            f" Parameters:\n{param_str}"
        )

        if self.df is not None and "exists" in self.df.columns:
            summary = self.summarize_files(self.df)
            repr_str += (
                f"\n\n Files Summary:\n"
                f"  Total files: {summary['total_files']}\n"
                f"  Existing files: {summary['existing_files']}\n"
                f"  Missing files: {summary['missing_files']}\n"
                f"  Total size of existing files: {format_size(summary['total_size'])}"
            )

        return repr_str

    def cross_product(self, nreps=None, use_seed=False):
        """
        Compute the Cartesian product of all columns of this `GridDf`,
        returning them in a new `GridDf`. Includes repetitions and seeding.

        Args:
            nreps (int, optional): The number of repetitions for each parameter combination.
                                   Defaults to None, which means no repetitions.
            use_seed (bool, optional): If True, add a unique seed for each repetition. If False,
                                       add a replicate number instead. Defaults to False. If runs
                                       may be added later, it is recommended this is set to False,
                                       since this guarantees that past files have the same seed.

        Returns:
            GridDf: The updated GridDf object with the cross product computed.
        """
        expanded_data = expand_params(self.params)

        if nreps:
            replicated_data = []
            for row in expanded_data:
                for rep in range(nreps):
                    row_copy = row.copy()
                    if use_seed:
                        row_copy["seed"] = random_seed(self.rng)
                    else:
                        row_copy["seed"] = rep
                    replicated_data.append(row_copy)
            expanded_data = replicated_data

        self.df = pl.DataFrame(expanded_data)
        self.nreps = nreps
        return self

    def _ensure_cross_generated(self, func):
        if self.df is None:
            msg = f"GridDf.{func}() can only be called after GridDf.cross_product()."
            raise ValueError(msg)

    def filter(
        self,
        *predicates: (
            IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any]
        ),
        **constraints: Any,
    ):
        """
        Filter the internal Polars DataFrame according to these criteria.
        """
        self._ensure_cross_generated("filter")
        self.df = self.df.filter(*predicates, **constraints)
        return self

    def path_pattern(self, filename_patterns=None, filename_sep="_"):
        """
        Generate the path pattern format string based on filename patterns.

        Args:
            filename_patterns (str or list): A string or list of strings representing the filename pattern(s).

        Returns:
            list: A list of path pattern format strings.
        """
        self._ensure_cross_generated("path_pattern")
        sep = self.sep
        dir = self.dir

        # Get the keys in the filename part of path
        filename_patterns = (
            self.filename_patterns if filename_patterns is None else filename_patterns
        )
        filename_patterns = parse_filename_patterns(filename_patterns)
        filename_cols = extract_columns(filename_patterns)

        # Get the keys in the directory part of path
        dir_keys = [
            key
            for key in self.df.columns
            if key not in filename_cols and key not in RESERVED
        ]

        path_patterns = []
        for pattern in sorted(filename_patterns):
            if dir is None:
                dir_path = (
                    ""
                    if not dir_keys
                    else os.path.join(*[f"{key}{sep}{{{key}}}" for key in dir_keys])
                )
            else:
                dir_path = (
                    dir
                    if not dir_keys
                    else os.path.join(
                        dir, *[f"{key}{sep}{{{key}}}" for key in dir_keys]
                    )
                )

            filename_format = re.sub(r"\{(.*?)\}", rf"\1{filename_sep}{{\1}}", pattern)
            path_pattern = os.path.join(dir_path, filename_format)
            path_patterns.append(path_pattern)

        return path_patterns

    def generate_path_items(
        self, filename_patterns, dir: str = None, sep: str = "__", filename_sep="_"
    ):
        """
        Generate paths based on filename patterns. This will *not* propagate the
        GridDf.df. If dir is specified as *not* None, this will overwrite the existing attribute.

        Args:
            filename_patterns (str or list): A string or list of strings representing the filename pattern(s).
            dir (str, optional): The base directory for the generated paths. Defaults to None.
            sep (str, optional): The separator used between the key and value in the generated paths. Defaults to "__".

        Returns:
            generator: A generator yielding tuples containing the full path and the row dictionary.
        """
        self._ensure_cross_generated("generate_paths")
        self.sep = sep
        if dir is not None:
            self.dir = dir
        paths = generate_paths(
            self.df, filename_patterns, self.dir, sep, filename_sep=filename_sep
        )
        return paths

    def _ensure_paths(self):
        try:
            paths = self.df["path"]
        except pl.exceptions.ColumnNotFoundError:
            raise ValueError(
                "The internal dataframe does not have a 'path' column. Generate paths."
            )
        return paths

    def paths(self):
        """
        Get a list of current paths.
        """
        paths = self._ensure_paths()
        return paths

    def generate_paths(
        self,
        filename_patterns,
        dir: str = None,
        sep: str = "__",
        split_filename=False,
        filename_sep: str = "_",
    ):
        """
        Generate paths based on filename patterns and return as a DataFrame.

        Args:
            filename_patterns (str or list): A string or list of strings representing the filename pattern(s).
            dir (str, optional): The base directory for the generated paths. Defaults to None.
            sep (str, optional): The separator used between the key and value in the generated paths. Defaults to "__".
            split_filename (bool, optional): If True, split the directory part and the file path. Defaults to False.

        Returns:
            GridDf
        """
        self._ensure_cross_generated("generate_paths_df")
        if dir is not None:
            self.dir = dir
        paths = list(
            self.generate_path_items(
                filename_patterns, self.dir, sep, filename_sep=filename_sep
            )
        )

        row_list = []
        for path, row in paths:
            if split_filename:
                directory, filename = os.path.split(path)
                row.update({"directory": directory, "filename": filename})
            row.update({"path": path})
            row_list.append(row)

        paths_df = pl.DataFrame(row_list)
        self.df = paths_df
        self.filename_patterns = filename_patterns
        self.sep = sep
        return self

    def query_files(self, local_dir=None):
        """
        Check if files exist and get their sizes, returning a DataFrame with this information.

        returns:
            GridDf
        """
        dir = self.dir
        paths = self._ensure_paths()

        exists = []
        sizes = []
        new_paths = []
        for path in paths:
            if local_dir is not None:
                path = os.path.join(local_dir, path)
            exists.append(os.path.exists(path))
            sizes.append(os.path.getsize(path) if os.path.exists(path) else None)
            new_paths.append(path)

        self.df = self.df.with_columns(
            [
                pl.Series("path", new_paths),
                pl.Series("exists", exists),
                pl.Series("size", sizes),
            ]
        )

        return self

    def summarize_files(self, df):
        """
        Generate a summary of the DataFrame created by `query_files`.

        Args:
            df (DataFrame): The DataFrame containing file paths, existence, and sizes.

        Returns:
            dict: A summary dictionary containing the total number of files, the number of existing files,
                  the number of missing files, and the total size of existing files.
        """
        total_files = df.shape[0]
        existing_files = df.filter(pl.col("exists") == True).shape[0]
        missing_files = total_files - existing_files
        total_size = (
            df.filter(pl.col("exists") == True)["size"].fill_null(value=0).sum()
        )

        summary = {
            "total_files": total_files,
            "existing_files": existing_files,
            "missing_files": missing_files,
            "total_size": total_size,
        }
        return summary

    def map_to_column(self, func, column_name):
        """
        Map a function over the file paths and put the results in
        a new DataFrame column.

        Args:
            func (callable): The function to apply to each file path.
            column_name (str): The name of the new column to store the results.

        Returns:
            GridDf: The updated GridDf object with the results in a new column.
        """
        if self.df is None:
            raise ValueError(
                "Files must be queried using `query_files()` before mapping function to files."
            )

        results = [
            func(row["path"]) if row["exists"] else None
            for row in self.df.iter_rows(named=True)
        ]

        self.df = self.df.with_columns(pl.Series(column_name, results))
        return self


def create_temp_files(file_paths):
    for path in file_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("test")


# Example usage and testing
if __name__ == "__main__":
    import tempfile

    def print_paths(paths):
        print("[", end="")
        for path in paths:
            print(f"{path},")
        print("]", end="")

    data = {"param1": [1, 2], "param2": [3, 4], "group": ["a", "b"]}

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        grid = GridDf(data).cross_product()
        file_paths = [
            path
            for path, _ in grid.generate_path_items(
                "data_{param1}_{param2}.tsv", dir=temp_dir
            )
        ]

        # Diagnostic print: Check generated paths
        print("Generated file paths:")
        for path in file_paths:
            print(path)

        # Create half the temp files
        create_temp_files(
            file_paths[: len(file_paths) // 2]
        )  # Create half of the files

        # Diagnostic print: Check files created
        print("Files created:")
        for path in file_paths[: len(file_paths) // 2]:
            print(path, os.path.exists(path), os.path.getsize(path))

        # Get files
        files_df = (
            grid.generate_paths("data_{param1}_{param2}.tsv", dir=temp_dir)
            .query_files()
            .df
        )
        # Diagnostic print: Check files DataFrame
        print("Files DataFrame after querying:")
        print(files_df)
        print(files_df["path"].to_list())

        # Check DataFrame
        # for path, row in grid.generate_path_items("data_{param1}_{param2}.tsv",
        #                                          dir=temp_dir):
        #    if os.path.exists(path):
        #        print(path, os.path.exists(path), os.path.getsize(path))
        #    else:
        #        print(f"does not exist: {path}")

        expected_df = pl.DataFrame(
            [
                {
                    **row,
                    "path": path,
                    "exists": os.path.exists(path),
                    "size": os.path.getsize(path) if os.path.exists(path) else None,
                }
                for path, row in grid.generate_path_items(
                    "data_{param1}_{param2}.tsv", dir=temp_dir
                )
            ]
        )
        # Diagnostic print: Check expected DataFrame
        print("Expected DataFrame:")
        print(expected_df)

        cols = ["path", "exists", "size", "param1", "param2"]
        assert files_df.select(cols).equals(expected_df.select(cols))

        # Summarize files
        summary = grid.summarize_files(files_df)
        expected_summary = {
            "total_files": len(file_paths),
            "existing_files": len(file_paths) // 2,
            "missing_files": len(file_paths) // 2,
            "total_size": sum(
                os.path.getsize(path) for path in file_paths[: len(file_paths) // 2]
            ),
        }
        # Diagnostic print: Check summary and expected summary
        print("Summary:")
        print(summary)
        print("Expected Summary:")
        print(expected_summary)

        assert summary == expected_summary

        grid.query_files()
        print(grid.path_pattern())
        # print(grid.df)
