# TAOConvert #

## Description ##

A utility to assist users in converting their semianalytic data
into a format that can be easily imported into TAO.

## Installation ##

Please install by running the `setup.py` command:

```bash
git clone https://github.com/CAS-eResearch/TAOImport.git
cd TAOImport
python setup.py install
```

You may need to prefix the command with `sudo` if you are installing
system-wide on a Linux machine. Otherwise, use `python setup.py install --user` 
to install on a per user basis.

## Preparing your input data ##

For the Spectral Energy Distribution (SED) of galaxies to be calculated correctly
within TAO, the following properties must be provided:
* Decomposition of Star Formation Rate into Bulge and Disk (in terms of input fields `sfrdisk` and `sfrbulge`)
* Decomposition of Merger Types into these 4 choices:
  - 0 -> None
  - 1 -> Major Merger
  - 2 -> Minor Merger
  - 3 -> Disk Instability
  - 4 -> Disrupt to ICS
* Snapshot at which merger occurs
* The primary galaxy into with the merger occurs (i.e., the index of galaxy that survives after the merger)
 
For a merging secondary galaxy, all these properties *must* be recorded at the last output of the secondary galaxy.

## Getting help. ##

The conversion process is accessed by running the `taoconvert` script
after installation. Help can be accessed on the command line via the
`-h` or `--help` option:

```bash
taoconvert -h
```

## Control Scripts ##

Control over how the conversion utility creates TAO input data is given
through a user created control script. The control script is simply a
Python script that is read by `taoconvert`. The script can be placed in
the current working directory and called `taoconv.py`, or can be specified
on the command line with the `-s` or `--script` option:

```bash
taoconvert -s examples/sage.py
```

The control script should consist of a class that inherits from `tao.Converter`,
and should override appropriate methods to implement the following features:

 1. Extra arguments to be specified on the command line.
 2. Simulation parameters.
 3. Snapshot redshifts.
 4. The data mapping.
 5. A tree iterator.

### Extra Arguments ###

The extra arguments are a way to allow transient information to be given
to the converter that relates directly to a specific data type. For example,
when converting SAGE data one needs to provide the path in which the
output trees are stored, the SAGE parameter file, the simulation box size, and
an a-list file.

To specify arguments, one must override the class method `add_arguments`:

```python
class SAGEConverter(tao.Converter):
  @classmethod
  def add_arguments(self, parser):
    parser.add_argument(
      '--trees-dir',
      default='.',
      help='location of SAGE trees'
    )
```

In the instance methods of the overridden converter class, arguments parsed
from the command-line can be accessed via the `args` member.

### Simulation Parameters ###

Simulation parameters are given as a dictionary of values. The required values are:

 * The simulation box size, `box_size`.
 * The Hubble constant, `hubble`.
 * OmegaM, `omega_m`.
 * OmegaL, `omega_l`.

To supply these values override the instance method `get_simulation_data`:

```python
class SAGEConverter(tao.Converter):
  def get_simulation_data(self):
    return {
      'box_size': 62.5,
      'hubble': 0.71,
      'omega_m': 0.25,
      'omega_l': 0.75,
    }
```

### Snapshot Redshifts ###

Snapshot redshifts are specified by setting a list-like value to the
`snapshot_redshifts` variable name. For example:

```python
snapshot_redshifts = [127, 94, 32, 0]
```

### Data Mappings ###

Data mappings are the mechanism to allow conversion of the source data
values into the required data values of TAO. There are three kinds of data
mapping, basic mappings, complex mappings, and direct mappings.

A basic mapping specifies
what is essentially a renaming of the input data fields to match the
TAO fields. For example, TAO requires position to be given as three
different fields named `posx`, `posy`, and `posz`. If
the source data stores positions as `x`, `y`, and `z` then they can use
a basic mapping to change to the correct name. Basic mappings may be
specified by instantiating a `tao.Mapping` class and providing a dictionary
of mapped values, like this:

```python
mapping = tao.Mapping({
  'posx': 'x',
  'posy': 'y',
  'posz': 'z',
})
```

A complex mapping may be needed to perform any transformation on the
data in order to prepare it for TAO. For example, the positions are
required to be given as Mpc/h. If the source data is stored just as
Mpc then a complex mapping would be required to change the units.
Complex mappings are specified by inheriting from the `tao.Mapping`
class and providing methods to convert data. The methods are named
`map_field_name`, where `field_name` is to be replaced by the name
of the TAO field you wish to calculate. For example, to implement the
position mappings discussed above:

```python
class MyMapping(tao.Mapping):
  def map_posx(tree):
    x = tree['x']
    return x/h
  def map_posy(tree):
    y = tree['y']
    return y/h
  def map_posz(tree):
    z = tree['z']
    return z/h

mapping = MyMapping()
```

The argument given to the mapping methods is a NumPy compound array
containing the loaded source tree.

The final kind of mapping specifies a direct copy of source data. This
is given as a second argument to the instantiation of the mapping class,
which is required to be a list of source field names to be transfered
as is to the output data. For example:

``` python
mapping = tao.Mapping({
  # basic mappings
}, [
  'field_0',
  'field_1',
])
```

### Tree Iterator ###

The tree iterator is the primary means of taking the source trees
from disk and arranging them in a structure useful to the conversion.
We need the source data to be loaded one merger tree at a time and
added to the output. Merger trees need to be represented as a compound
NumPy array of galaxies included in each tree. The structure of the
tree is represented by a property on each galaxy that gives the
0-based index of the descendant galaxy in the same tree. For example,
a tree with three galaxies, A, B, and C, for which B and C merge into
A, could be represented as an array of descendant indices such as:

```python
# A B C #
descendants = [2, 2, -1]
```

Here the numbers are indices into the descendant array, and -1 indicates
there is no descendant. Internal ordering in the loaded merger tree does
not matter so long as hierarchy storage via the descendant value is
maintained.

The interface for iterating over trees is a function called `iterate_trees`.
It accepts a single argument, `args`, which is the result of parsing the
command line arguments. The function should be implemented as a Python
generator, yielding as many results as there are trees to convert.

As an example, loading positions and a descendant value from multiple
binary files could be implemented as:

```python
def iterate_trees(args):
  dtype = np.dtype([
    ('posx', 'f'),
    ('posy', 'f'),
    ('posz', 'f'),
    ('descendant', 'i'),
  ])
  yield np.fromfile('tree_0.dat', dtype)
  yield np.fromfile('tree_1.dat', dtype)
```

## Examples ##

There are a few examples of control scripts provided within the `examples`
directory.

## Running the Conversion Script for SAGE 2.0 ##

If you have [SAGE 2.0](https://github.com/darrencroton/sage), **and** you want to 
import the standard fields into [TAO](http://tao.asvo.org.au/tao), then these are the steps:

1. Install this repo
2. Run this command-line from the `tao/examples` directory:
```shell
taoconvert -s sage.py --trees-dir=/path/to/sage/output/directory/ --parameters=/path/to/sage/parameter-file --box-size=BOXSIZE --a-list=/path/to/file/with/scales -o /path/to/output/basefilename --sim-name="SIMULATION" --model-name="SAGE 2.0"
```

Note, this assumes that the output of SAGE are stored in files called `model_z*` files. If you have altered the output filename for SAGE, then you will have to edit `sage.py` and replace `model_z` with the appropriate filebase. 


