import os
import pytest
import tempfile
import polars as pl

from grid_df.grid_df import (
    parse_filename_patterns,
    extract_columns,
    expand_params,
    GridDf,
    create_temp_files,
)

def test_parse_filename_patterns():
    assert parse_filename_patterns("data.tsv") == ["data.tsv"]
    assert parse_filename_patterns(["data1.tsv", "data2.tsv"]) == ["data1.tsv", "data2.tsv"]


def test_extract_columns():
    assert extract_columns([]) == []
    assert extract_columns(["data_{param1}.tsv"]) == ["param1"]
    assert set(extract_columns(["data_{param1}_{param2}.tsv"])) == set(["param1", "param2"])
    with pytest.raises(ValueError, match="All filename patterns must have the same parameters."):
        extract_columns(["data_{param1}.tsv", "data_{param2}.tsv"])


def test_expand_params():
    data = {
        'param1': [1, 2], 
        'param2': [3, 4],
        'filename_col': ['a', 'b']
    }
    expected = [{'param1': 1, 'param2': 3, 'filename_col': 'a'},
                {'param1': 1, 'param2': 3, 'filename_col': 'b'},
                {'param1': 1, 'param2': 4, 'filename_col': 'a'},
                {'param1': 1, 'param2': 4, 'filename_col': 'b'},
                {'param1': 2, 'param2': 3, 'filename_col': 'a'},
                {'param1': 2, 'param2': 3, 'filename_col': 'b'},
                {'param1': 2, 'param2': 4, 'filename_col': 'a'},
                {'param1': 2, 'param2': 4, 'filename_col': 'b'}]
    assert expand_params(data) == expected

    with pytest.raises(ValueError, match="param2 did not have any values."):
        expand_params({'param1': [1, 2], 'param2': [], 'filename_col': ['a', 'b']})



def test_grid_df_generate_path_items():
    data = {
        'param1': [1, 2],
        'param2': [3, 4],
        'group': ['a', 'b']
        }

    grid = GridDf(data)
    results = list(grid.cross_product().generate_path_items("data_{param2}.tsv"))
    expected = [('param1__1/group__a/data_param2__3.tsv', {'param1': 1, 'param2': 3, 'group': 'a'}),
                ('param1__1/group__b/data_param2__3.tsv', {'param1': 1, 'param2': 3, 'group': 'b'}),
                ('param1__1/group__a/data_param2__4.tsv', {'param1': 1, 'param2': 4, 'group': 'a'}),
                ('param1__1/group__b/data_param2__4.tsv', {'param1': 1, 'param2': 4, 'group': 'b'}),
                ('param1__2/group__a/data_param2__3.tsv', {'param1': 2, 'param2': 3, 'group': 'a'}),
                ('param1__2/group__b/data_param2__3.tsv', {'param1': 2, 'param2': 3, 'group': 'b'}),
                ('param1__2/group__a/data_param2__4.tsv', {'param1': 2, 'param2': 4, 'group': 'a'}),
                ('param1__2/group__b/data_param2__4.tsv', {'param1': 2, 'param2': 4, 'group': 'b'})]
    assert results == expected



def test_grid_df_generate_path_items_filter():
    data = {
        'param1': [1, 2],
        'param2': [3, 4],
        'group': ['a', 'b']
        }

    grid = GridDf(data)
    results = list(grid.cross_product().filter(param1=1).generate_path_items("data_{param2}.tsv"))
    expected = [('param1__1/group__a/data_param2__3.tsv', {'param1': 1, 'param2': 3, 'group': 'a'}),
                ('param1__1/group__b/data_param2__3.tsv', {'param1': 1, 'param2': 3, 'group': 'b'}),
                ('param1__1/group__a/data_param2__4.tsv', {'param1': 1, 'param2': 4, 'group': 'a'}),
                ('param1__1/group__b/data_param2__4.tsv', {'param1': 1, 'param2': 4, 'group': 'b'})]
    assert results == expected



def test_grid_df_generate_path_items_filter_two():

    data = {
        'param1': [1, 2],
        'param2': [3, 4, 21],
        'group': ['a', 'b']
    }
    print(expand_params(data))

    grid = GridDf(data)

    results = list(grid
                   .cross_product()
                   .filter(param1=1, param2=21)
                   .generate_path_items("data.tsv"))

    expected = [('param1__1/param2__21/group__a/data.tsv', {'param1': 1, 'param2': 21, 'group': 'a'}),
                ('param1__1/param2__21/group__b/data.tsv', {'param1': 1, 'param2': 21, 'group': 'b'})]

    assert results == expected


def test_grid_df_generate_path_items_filter_with_seed():
    data = {
        'param1': [1, 2],
        'param2': [3, 4, 21],
        'group': ['a', 'b']
    }

    grid = GridDf(data, seed=0)
    results = list(grid
                  .cross_product(nreps=2, use_seed=True)
                  .filter(param1=1, param2=21)
                  .generate_path_items("data.tsv"))


    expected = [('param1__1/param2__21/group__a/seed__5014055544817598430/data.tsv', {'param1': 1, 'param2': 21, 'group': 'a', 'seed': 5014055544817598430}),
                ('param1__1/param2__21/group__a/seed__8624520845998120949/data.tsv', {'param1': 1, 'param2': 21, 'group': 'a', 'seed': 8624520845998120949}),
                ('param1__1/param2__21/group__b/seed__25258205892266071/data.tsv', {'param1': 1, 'param2': 21, 'group': 'b', 'seed': 25258205892266071}),
                ('param1__1/param2__21/group__b/seed__7524920857253125029/data.tsv', {'param1': 1, 'param2': 21, 'group': 'b', 'seed': 7524920857253125029})]

    assert results == expected


def test_generate_paths():
    data = {
        'param1': [1, 2],
        'param2': [3, 4],
        'group': ['a', 'b']
    }

    grid = GridDf(data)
    grid.cross_product().generate_paths("data_{param2}.tsv")
    df = grid.df

    expected = pl.DataFrame([
        {'param1': 1, 'param2': 3, 'group': 'a', 'path': 'param1__1/group__a/data_param2__3.tsv'},
        {'param1': 1, 'param2': 3, 'group': 'b', 'path': 'param1__1/group__b/data_param2__3.tsv'},
        {'param1': 1, 'param2': 4, 'group': 'a', 'path': 'param1__1/group__a/data_param2__4.tsv'},
        {'param1': 1, 'param2': 4, 'group': 'b', 'path': 'param1__1/group__b/data_param2__4.tsv'},
        {'param1': 2, 'param2': 3, 'group': 'a', 'path': 'param1__2/group__a/data_param2__3.tsv'},
        {'param1': 2, 'param2': 3, 'group': 'b', 'path': 'param1__2/group__b/data_param2__3.tsv'},
        {'param1': 2, 'param2': 4, 'group': 'a', 'path': 'param1__2/group__a/data_param2__4.tsv'},
        {'param1': 2, 'param2': 4, 'group': 'b', 'path': 'param1__2/group__b/data_param2__4.tsv'},
    ])

    assert df.equals(expected)


def test_generate_paths_with_filter():
    data = {
        'param1': [1, 2],
        'param2': [3, 4],
        'group': ['a', 'b']
    }

    grid = GridDf(data)
    grid.cross_product().filter(param1=1).generate_paths("data_{param2}.tsv")
    df = grid.df

    expected = pl.DataFrame([
        {'param1': 1, 'param2': 3, 'group': 'a', 'path': 'param1__1/group__a/data_param2__3.tsv'},
        {'param1': 1, 'param2': 3, 'group': 'b', 'path': 'param1__1/group__b/data_param2__3.tsv'},
        {'param1': 1, 'param2': 4, 'group': 'a', 'path': 'param1__1/group__a/data_param2__4.tsv'},
        {'param1': 1, 'param2': 4, 'group': 'b', 'path': 'param1__1/group__b/data_param2__4.tsv'},
    ])

    assert df.equals(expected)


def test_generate_paths_with_split_filename():
    data = {
        'param1': [1, 2],
        'param2': [3, 4],
        'group': ['a', 'b']
    }

    grid = GridDf(data)
    grid = grid.cross_product().generate_paths("data_{param2}.tsv", split_filename=True)
    df = grid.df

    expected = pl.DataFrame([
        {'param1': 1, 'param2': 3, 'group': 'a', 'directory': 'param1__1/group__a', 'filename': 'data_param2__3.tsv', 'path': 'param1__1/group__a/data_param2__3.tsv'},
        {'param1': 1, 'param2': 3, 'group': 'b', 'directory': 'param1__1/group__b', 'filename': 'data_param2__3.tsv', 'path': 'param1__1/group__b/data_param2__3.tsv'},
        {'param1': 1, 'param2': 4, 'group': 'a', 'directory': 'param1__1/group__a', 'filename': 'data_param2__4.tsv', 'path': 'param1__1/group__a/data_param2__4.tsv'},
        {'param1': 1, 'param2': 4, 'group': 'b', 'directory': 'param1__1/group__b', 'filename': 'data_param2__4.tsv', 'path': 'param1__1/group__b/data_param2__4.tsv'},
        {'param1': 2, 'param2': 3, 'group': 'a', 'directory': 'param1__2/group__a', 'filename': 'data_param2__3.tsv', 'path': 'param1__2/group__a/data_param2__3.tsv'},
        {'param1': 2, 'param2': 3, 'group': 'b', 'directory': 'param1__2/group__b', 'filename': 'data_param2__3.tsv', 'path': 'param1__2/group__b/data_param2__3.tsv'},
        {'param1': 2, 'param2': 4, 'group': 'a', 'directory': 'param1__2/group__a', 'filename': 'data_param2__4.tsv', 'path': 'param1__2/group__a/data_param2__4.tsv'},
        {'param1': 2, 'param2': 4, 'group': 'b', 'directory': 'param1__2/group__b', 'filename': 'data_param2__4.tsv', 'path': 'param1__2/group__b/data_param2__4.tsv'},
    ])

    assert df.equals(expected)


def test_generate_paths_with_filter_and_split_filename():
    data = {
        'param1': [1, 2],
        'param2': [3, 4, 21],
        'group': ['a', 'b']
    }

    grid = GridDf(data)
    grid.cross_product().filter(param1=1, param2=21).generate_paths("data.tsv", split_filename=True)
    df = grid.df

    expected = pl.DataFrame([
        {'param1': 1, 'param2': 21, 'group': 'a', 'directory': 'param1__1/param2__21/group__a', 'filename': 'data.tsv', 'path': 'param1__1/param2__21/group__a/data.tsv'},
        {'param1': 1, 'param2': 21, 'group': 'b', 'directory': 'param1__1/param2__21/group__b', 'filename': 'data.tsv', 'path': 'param1__1/param2__21/group__b/data.tsv'},
    ])

    assert df.equals(expected)

def test_query_files_and_summarize_files():
    # Define data
    data = {
        'param1': [1, 2],
        'param2': [3, 4],
        'group': ['a', 'b']
    }

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        grid = GridDf(data).cross_product()
        file_paths = [path for path, _ in 
                      grid.generate_path_items("data_{param1}_{param2}.tsv",
                                                                   dir=temp_dir)]

        # Diagnostic print: Check generated paths
        print("Generated file paths:")
        for path in file_paths:
            print(path)

        # Create half the temp files
        create_temp_files(file_paths[:len(file_paths)//2])  # Create half of the files

        # Diagnostic print: Check files created
        print("Files created:")
        for path in file_paths[:len(file_paths)//2]:
            print(path, os.path.exists(path), os.path.getsize(path))

        # Get files
        files_df = (grid
                    .generate_paths("data_{param1}_{param2}.tsv", dir=temp_dir)
                    .query_files().df)
        # Diagnostic print: Check files DataFrame
        print("Files DataFrame after querying:")
        print(files_df)
        print(files_df['path'].to_list())

        # Check DataFrame
        #for path, row in grid.generate_path_items("data_{param1}_{param2}.tsv",
        #                                          dir=temp_dir):
        #    if os.path.exists(path):
        #        print(path, os.path.exists(path), os.path.getsize(path))
        #    else:
        #        print(f"does not exist: {path}")

        expected_df = pl.DataFrame([
            {**row, 'path': path,
             'exists': os.path.exists(path),
             'size': os.path.getsize(path) if os.path.exists(path) else None}
            for path, row in grid.generate_path_items("data_{param1}_{param2}.tsv",
                                                      dir=temp_dir)
        ])
        # Diagnostic print: Check expected DataFrame
        print("Expected DataFrame:")
        print(expected_df)

        cols = ['path', 'exists', 'size', 'param1', 'param2']
        assert files_df.select(cols).equals(expected_df.select(cols))

        # Summarize files
        summary = grid.summarize_files(files_df)
        expected_summary = {
            'total_files': len(file_paths),
            'existing_files': len(file_paths) // 2,
            'missing_files': len(file_paths) // 2,
            'total_size': sum(os.path.getsize(path) for path in file_paths[:len(file_paths)//2])
        }
        # Diagnostic print: Check summary and expected summary
        print("Summary:")
        print(summary)
        print("Expected Summary:")
        print(expected_summary)

        assert summary == expected_summary

