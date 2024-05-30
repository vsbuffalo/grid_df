# GridDf Documentation

GridDf is a Python library and tool that simplifies running computational
experiments over some sort of grid (e.g. of parameters, files, options, etc).
GridDf is like another library I wrote,
*[SlimFlow](https://github.com/vsbuffalo/slimflow/)* that simplifies running
simulations, but is more general purpose.

## Quick Example

```python
>>> from grid_df import GridDf

>>> parameters = dict(theta=[1, 2], rho=[1, 2],
                      eps=np.logspace(-3, -1, 5))

>>> grid = GridDf(parameters)
>>> grid
GridDf Status:
Total number of parameters: 3
Total number of replicates: N/A
Seed: 31
Parameters:
  theta ∈ {1, 2}
  rho ∈ {1, 2}
  eps ∈ {0.001, 0.0031622776601683794, 0.01, 0.03162277660168379, 0.1}


>>> grid = GridDf(parameters, seed=31)
>>> grid.cross_product(nreps=10) # stores results in place
>>> grid.df # treatment/sample dataframe
shape: (200, 4)
┌───────┬─────┬───────┬─────────────────────┐
│ theta ┆ rho ┆ eps   ┆ seed                │
│ ---   ┆ --- ┆ ---   ┆ ---                 │
│ i64   ┆ i64 ┆ f64   ┆ i64                 │
╞═══════╪═════╪═══════╪═════════════════════╡
│ 1     ┆ 1   ┆ 0.001 ┆ 8330289625267613134 │
│ 1     ┆ 1   ┆ 0.001 ┆ 624221792569208323  │
│ 1     ┆ 1   ┆ 0.001 ┆ 6204846579103848259 │
│ 1     ┆ 1   ┆ 0.001 ┆ 4358455627292652714 │
│ 1     ┆ 1   ┆ 0.001 ┆ 6248168414806147411 │
│ …     ┆ …   ┆ …     ┆ …                   │
│ 2     ┆ 2   ┆ 0.1   ┆ 8391934664739369655 │
│ 2     ┆ 2   ┆ 0.1   ┆ 433194606698393151  │
│ 2     ┆ 2   ┆ 0.1   ┆ 6677485992861345647 │
│ 2     ┆ 2   ┆ 0.1   ┆ 1531321398586782866 │
│ 2     ┆ 2   ┆ 0.1   ┆ 2497551049888867841 │
└───────┴─────┴───────┴─────────────────────┘

>>> # re-generate grid, and add on a filter (using Polars).
>>> import polars as pl
>>> grid.cross_product(nreps=10).filter(pl.col("eps") < 1e-2, rho=2)
>>> grid.df
shape: (40, 4)
┌───────┬─────┬──────────┬─────────────────────┐
│ theta ┆ rho ┆ eps      ┆ seed                │
│ ---   ┆ --- ┆ ---      ┆ ---                 │
│ i64   ┆ i64 ┆ f64      ┆ i64                 │
╞═══════╪═════╪══════════╪═════════════════════╡
│ 1     ┆ 2   ┆ 0.001    ┆ 1928002274283433492 │
│ 1     ┆ 2   ┆ 0.001    ┆ 6203347984245720503 │
│ 1     ┆ 2   ┆ 0.001    ┆ 607414143343001079  │
│ 1     ┆ 2   ┆ 0.001    ┆ 3559219570239069049 │
│ 1     ┆ 2   ┆ 0.001    ┆ 6483318352466503760 │
│ …     ┆ …   ┆ …        ┆ …                   │
│ 2     ┆ 2   ┆ 0.003162 ┆ 3817825041754741265 │
│ 2     ┆ 2   ┆ 0.003162 ┆ 3845773742203971537 │
│ 2     ┆ 2   ┆ 0.003162 ┆ 4928324071964660663 │
│ 2     ┆ 2   ┆ 0.003162 ┆ 5080691066214215007 │
│ 2     ┆ 2   ┆ 0.003162 ┆ 1520904907558386    │
└───────┴─────┴──────────┴─────────────────────┘

```

This workflow pattern can be thought of as **expand-filter**. At the very most,
a computational experiment will have a fully factorial design: some quantity
will be calculated on the Cartesian product of all variables (and their
replicates). In some case, certain parameter combinations may be invalid or the
experimenter may want to narrow the search space to reduce computational
overhead. Certain combinations can be programmatically and explicitly filtered
away with with the `filter()` method which is simply passed to Polars dataframe
of all results.

### Generating file paths

When the right parameter combinations have been constructed, we then generate
the file paths. The schema that is used is like
`dir/theta__1/rho__0.001/filename_1928002274283433492.tsv`, where parameter
keys and values are concatenated according to `sep` (default is `__`) to make the 
directory component. The user specifies the filename pattern, which *can* include
*any* of the columns (including seed), but doesn't necessarily need to. The filename
also can have many columns. All columns *not* in the filename pattern will go into 
the directory part of the path. Here is a simple example:

```python
>>> parameters = dict(theta=[1, 2], rho=[1, 2], eps=np.logspace(-3, -1, 5))
>>> grid = GridDf(parameters, seed=31)
>>> df = (grid
      .cross_product(nreps=10)
      .filter(rho=1)
      .generate_paths("{theta}__{seed}.tsv", dir="results"))
>>> print(df['path'].to_list()[:2])
['results/rho__1/eps__0.001/theta__1__seed__365780996487948558.tsv', 
 'results/rho__1/eps__0.001/theta__1__seed__472337310787310937.tsv']
```

You can see that all parameters except `theta` and the seed are used to build
the file path for each result. Then you can write the samples as TSV:

```python
>>> grid.write_tsv("samples.tsv")
```

You probably wouldn't want ever want to not put the seed column into the
filename, but `GridDf` would allow it. You can see here how it would 
structure the filepath:


```python
>>> grid = GridDf(parameters, seed=31)
>>> df = (grid
          .cross_product(nreps=10)
          .filter(rho=1)
          .generate_paths("{theta}__{eps}.tsv", dir="results"))
>>> print(df['path'].to_list()[0])
'results/rho__1/seed__365780996487948558/theta__1__eps__0.001.tsv'
```

Note that the parameter dictionary can be written as a YAML configuation file,
and then read in. This suggests a workflow like: each experiment specifies a
set of parameters in a YAML file. The combinations are created, filtered, and
the filepaths are generated and written to a local TSV file of the samples.
This is then read in by something like Snakemake (or, Snakemake can call
`grid_df` directly), which uses the file paths to run the computational
experiments or calculations.

### Collecting and processing the resulting files

With the resulting files generated (e.g. from Snakemake), we can then load
in the TSV of expected samples, and process it.

```python
>>> grid = GridDf.from_tsv("samples.tsv")
>>> grid
>>> grid.df
shape: (100, 5)
┌───────┬─────┬───────┬─────────────────────┬─────────────────────────────────┐
│ theta ┆ rho ┆ eps   ┆ seed                ┆ path                            │
│ ---   ┆ --- ┆ ---   ┆ ---                 ┆ ---                             │
│ i64   ┆ i64 ┆ f64   ┆ i64                 ┆ str                             │
╞═══════╪═════╪═══════╪═════════════════════╪═════════════════════════════════╡
│ 1     ┆ 1   ┆ 0.001 ┆ 365780996487948558  ┆ rho__1/eps__0.001/theta__1__se… │
│ 1     ┆ 1   ┆ 0.001 ┆ 472337310787310937  ┆ rho__1/eps__0.001/theta__1__se… │
│ 1     ┆ 1   ┆ 0.001 ┆ 624221792569208323  ┆ rho__1/eps__0.001/theta__1__se… │
│ 1     ┆ 1   ┆ 0.001 ┆ 1628203149862637576 ┆ rho__1/eps__0.001/theta__1__se… │
│ 1     ┆ 1   ┆ 0.001 ┆ 2786919589362691562 ┆ rho__1/eps__0.001/theta__1__se… │
│ …     ┆ …   ┆ …     ┆ …                   ┆ …                               │
│ 2     ┆ 1   ┆ 0.1   ┆ 5845317608633120295 ┆ rho__1/eps__0.1/theta__2__seed… │
│ 2     ┆ 1   ┆ 0.1   ┆ 7283582347823813098 ┆ rho__1/eps__0.1/theta__2__seed… │
│ 2     ┆ 1   ┆ 0.1   ┆ 8326942101501739696 ┆ rho__1/eps__0.1/theta__2__seed… │
│ 2     ┆ 1   ┆ 0.1   ┆ 8417947907201963168 ┆ rho__1/eps__0.1/theta__2__seed… │
│ 2     ┆ 1   ┆ 0.1   ┆ 8967082047165422699 ┆ rho__1/eps__0.1/theta__2__seed… │
└───────┴─────┴───────┴─────────────────────┴─────────────────────────────────┘
```

Then these files can be queried, which loads their file status and size into
the dataframe. This gives a small summary, and change the dataframe:

```python
>>> grid.query_files()
GridDf Status:
 Total number of parameters: 3
 Total number of replicates: NA
 Seed: 1233895214273657537
 Parameters:
   param1 ∈ {1, 2}
   param2 ∈ {3, 4}
   group ∈ {a, b}

 Files Summary:
  Total files: 8
  Existing files: 4
  Missing files: 4
  Total size of existing files: 16.00 bytes

>>> grid.df
shape: (8, 6)
┌────────┬────────┬───────┬─────────────────────────────────┬────────┬──────┐
│ param1 ┆ param2 ┆ group ┆ path                            ┆ exists ┆ size │
│ ---    ┆ ---    ┆ ---   ┆ ---                             ┆ ---    ┆ ---  │
│ i64    ┆ i64    ┆ str   ┆ str                             ┆ bool   ┆ i64  │
╞════════╪════════╪═══════╪═════════════════════════════════╪════════╪══════╡
│ 1      ┆ 3      ┆ a     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ true   ┆ 4    │
│ 1      ┆ 3      ┆ b     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ true   ┆ 4    │
│ 1      ┆ 4      ┆ a     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ true   ┆ 4    │
│ 1      ┆ 4      ┆ b     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ true   ┆ 4    │
│ 2      ┆ 3      ┆ a     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ false  ┆ null │
│ 2      ┆ 3      ┆ b     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ false  ┆ null │
│ 2      ┆ 4      ┆ a     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ false  ┆ null │
│ 2      ┆ 4      ┆ b     ┆ /var/folders/4w/tx_sszv90dlbrx… ┆ false  ┆ null │
└────────┴────────┴───────┴─────────────────────────────────┴────────┴──────┘
```

