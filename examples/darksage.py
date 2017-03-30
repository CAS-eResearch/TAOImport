"""Convert DARK SAGE output to TAO.

A control script to be used with `taoconvert` to convert DARK SAGE output
binary data into HDF5 input for TAO.

"""
from __future__ import print_function
import re, os
import numpy as np
import tao
from collections import OrderedDict
from tqdm import tqdm

class DARKSAGEConverter(tao.Converter):
    """Subclasses tao.Converter to perform SAGE output conversion."""

    def __init__(self, *args, **kwargs):
        src_fields_dict = OrderedDict([
                ('StellarMass', {
                        'type': np.float32,
                        'label': "Total Stellar Mass",
                        'description': "Total stellar mass of galaxy",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",
                        'order': 0,
                        }),
               ('StellarDiscMass', {
                    'type': np.float32,
                        'label': "Disk Stellar Mass",
                        'description': "Total stellar mass within the disk, including the pseudobulge",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",
                        'order': 1,
                        }),
                ('MergerBulgeMass', {
                        'type': np.float32,
                        'label': "Merger-driven Bulge Mass",
                        'description': "Stellar mass in the merger-driven bulge of the galaxy",
                        'units': '10^10 Msun/h',
                        'group': "Galaxy Masses",                        
                        'order': 2,
                        }),
               ('InstabilityBulgeMass', {
                    'type': np.float32,
                    'label': "Instability-driven Bulge Mass",
                    'description': "Stellar mass in the instability-driven bulge of the galaxy",
                    'units': '10^10 Msun/h',
                    'group': "Galaxy Masses",
                    'order': 3,
                    }),
               ('PseudoBulgeMass', {
                    'type': np.float32,
                    'label': "Pseudobulge Mass",
                    'description': "Stellar mass in the disk inside 0.2 * Cooling Scale Radius",
                    'units': "10^10 Msun/h",
                    'group': "Galaxy Masses",
                    'order': 4,
                    }),
               ('BlackHoleMass', {
                        'type': np.float32,
                        'label': "Black Hole Mass",
                        'units': "10^10 Msun/h",
                        'description': "Mass of the black hole",
                        'group': "Galaxy Masses",
                        'order': 5,
                        }),
                ('ColdGas', {
                        'type': np.float32,
                        'label': "Cold Gas Mass",
                        'units': "10^10 Msun/h",
                        'description': "Cold-gas mass of the galaxy",
                        'group': "Galaxy Masses",
                        'order': 6,
                        }),
               ('HImass', {
                        'type': np.float32,
                        'label': "HI Mass",
                        'description': "Atomic-hydrogen mass of the galaxy",
                        'group': "Galaxy Masses",
                        'units': "10^10 Msun/h",
                        'order': 7,
                        }),
               ('H2mass', {
                        'type': np.float32,
                        'label': "H2 Mass",
                        'description': "Molecular-hydrogen mass of the galaxy",
                        'group': "Galaxy Masses",
                        'units': "10^10 Msun/h",
                        'order': 8,
                        }),
                ('HotGas', {
                        'type': np.float32,
                        'label': "Hot Gas Mass",
                        'units': "10^10 Msun/h",
                        'description': "Hot-gas mass around the galaxy",
                        'group': "Galaxy Masses",                        
                        'order': 9,
                        }),
                ('EjectedMass', {
                        'type': np.float32,
                        'label': "Ejected Gas Mass",
                        'units': "10^10 Msun/h",
                        'description': "Mass of gas ejected from both the galaxy and the halo",
                        'group': "Galaxy Masses",
                        'order': 10,
                        }),
                ('ICS', {
                        'type': np.float32,
                        'label': "Intracluster Stars Mass",
                        'units': "10^10 Msun/h",
                        'description': "Stellar mass dispersed in the halo, not in any particular galaxy",
                        'group': "Galaxy Masses",                        
                        'order': 11,
                        }),
                ('MetalsStellarMass', {
                        'type': np.float32,
                        'label': "Metals Total Stellar Mass",
                        'description': "Mass of metals in the galaxy stars (bulge + disk)",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 12,
                        }),
               ('MetalsStellarDiscMass', {
                        'type': np.float32,
                        'label': "Metals Stellar Disk Mass",
                        'description': "Mass of metals in the stellar disk, including the pseudobulge",                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",
                        'order': 13,
                        }),
                ('MetalsMergerBulgeMass', {
                        'type': np.float32,
                        'label': "Metals Merger Bulge Mass",
                        'description': "Mass of metals in stars in the merger-driven bulge",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 14,
                        }),
                ('MetalsInstabilityBulgeMass', {
                        'type': np.float32,
                        'label': "Metals Instability Bulge Mass",
                        'description': "Mass of metals in stars in the instability-driven bulge",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",
                        'order': 15,
                        }),
               ('MetalsPseudoBulge', {
                        'type': np.float32,
                        'label': "Metals Pseudobulge Mass",
                        'description': "Mass of metals in the disk inside 0.2 * Cooling Scale Radius",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",
                        'order': 16,
                        }),
                ('MetalsColdGas', {
                        'type': np.float32,
                        'label': "Metals Cold Gas Mass",
                        'description': "Mass of metals in the cold gas phase",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 17,
                        }),
                ('MetalsHotGas', {
                        'type': np.float32,
                        'label': "Metals Hot Gas Mass",
                        'description': "Mass of metals in the hot gas phase",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 18,
                        }),
                ('MetalsEjectedMass', {
                        'type': np.float32,
                        'label': "Metals Ejected Gas Mass",
                        'description': "Mass of metals in the gas ejected from the galaxy",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 19,
                        }),
                ('MetalsICS', {
                        'type': np.float32,
                        'label': "Metals Intracluster Stars Mass",
                        'description': "Mass of metals in stars dispersed throughout the halo",
                        'units': "10^10 Msun/h",
                        'group': "Galaxy Masses",                        
                        'order': 20,
                        }),
                ('ObjectType', {
                        'type': np.int32,
                        'label': "Galaxy Classification",
                        'description': "Galaxy type: 0-central, 1-satellite",
                        'group': "Galaxy Properties",
                        'order': 21,
                        }),
                ('DiskScaleRadius', {
                        'type': np.float32,
                        'label': "Cooling Scale Radius",
                        'description': "Scale radius used in determining how freshly cooled gas is distributed",
                        'group': "Galaxy Properties",
                        'units': "Mpc/h",
                        'order': 22,
                        }),
               ('r50', {
                        'type': np.float32,
                        'label': "Half-Mass Radius",
                        'description': "Radius enclosing 50% of the stellar disk content",
                        'group': "Galaxy Properties",
                        'units': "Mpc/h",
                        'order': 23,
                        }),
               ('r90', {
                        'type': np.float32,
                        'label': "90-Percent Radius",
                        'description': "Radius enclosing 90% of the stellar disk content",
                        'group': "Galaxy Properties",
                        'units': "Mpc/h",
                        'order': 24,
                        }),
                ('RadiusHI', {
                         'type': np.float32,
                         'label': "HI Radius",
                         'description': "Radius where the surface density of HI crosses 1 Msun/pc^2",
                         'group': "Galaxy Properties",
                         'units': "Mpc/h",
                         'order': 25,
                        }),
               ('RadiusTrans', {
                        'type': np.float32,
                        'label': "HI/H2 Transition Radius",
                        'description': "Radius where the surface density of HI and H2 equate",
                        'group': "Galaxy Properties",
                        'units': "Mpc/h",
                        'order': 26,
                        }),
               ('rSFR', {
                        'type': np.float32,
                        'label': "Star Formation Radius",
                        'description': "Radius enclosing 50% of the disk's star formation activity",
                        'group': "Galaxy Properties",
                        'units': "Mpc/h",
                        'order': 27,
                        }),
               ('TotSfr', {
                    'type': np.float32,
                    'label': "Total Star Formation Rate",
                    'description': "Total star formation rate (disk + bulge)",
                    'group': "Galaxy Properties",
                    'units': "Msun/year",
                    'order': 28,
                    }),
               ('dZStar', {
                        'type': np.float32,
                        'label': "Stellar Metallicity Gradient",
                        'description': "Metallicity gradient of the stellar disk between the half-mass and 90-percent radii",
                        'group': "Galaxy Properties",
                        'units': "dex/kpc*h",
                        'order': 29,
                        }),
               ('dZGas', {
                        'type': np.float32,
                        'label': "Gas Metallicity Gradient",
                        'description': "Metallicity gradient of the gas disk between the half-mass and 90-percent radii",
                        'group': "Galaxy Properties",
                        'units': "dex/kpc*h",
                        'order': 30,
                        }),
                ('Cooling', {
                        'type': np.float32,
                        'label': "Hot Gas Cooling Rate",
                        'description': "Net cooling rate of hot gas in the halo",
                        'group': "Galaxy Properties",
                        'units': "log10(erg/s)",
                        'order': 31,
                        }),
                ('Heating', {
                        'type': np.float32,
                        'label': "AGN Heating Rate",
                        'description': "Gross heating rate from the active galactic nucleus",
                        'group': "Galaxy Properties",
                        'units': "log10(erg/s)",
                        'order': 32,
                        }),
                ('TimeofLastMajorMerger', {
                        'type': np.float32,
                        'label': "Time of Last Major Merger",
                        'description': "Look-back time (from z=0) of last major merger",
                        'group': "Galaxy Properties",
                        'units': "Myr/h",
                        'order': 33,
                         }),
                ('OutflowRate', {
                        'type': np.float32,
                        'label': "Supernova Cold Gas Outflow Rate",
                        'description': "Cold-gas outflow rate from stellar feedback",
                        'group': "Galaxy Properties",
                        'units': "Msun/yr",
                        'order': 34,
                        }),
                 ('jStarDisc', {
                      'type': np.float32,
                      'label': "j Stellar Disk",
                      'description': "Specific angular momentum of the entire stellar disk, including the pseudobulge",
                      'group': "Galaxy Properties",
                      'units': "kpc/h * km/s",
                      'order': 35,
                      }),
                 ('jPseudoBulge', {
                      'type': np.float32,
                      'label': "j PseudoBulge",
                      'description': "Specific angular momentum of disk stars inside 0.2 * Cooling Scale Radius",
                      'group': "Galaxy Properties",
                      'units': "kpc/h * km/s",
                      'order': 36,
                      }),
                 ('jGas', {
                      'type': np.float32,
                      'label': "j Cold Gas",
                      'description': "Specific angular momentum of the entire cold gas disk",
                      'group': "Galaxy Properties",
                      'units': "kpc/h * km/s",
                      'order': 37,
                  }),
                 ('jHI', {
                      'type': np.float32,
                      'label': "j HI",
                      'description': "Specific angular momentum of atomic hydrogen in the gas disk",
                      'group': "Galaxy Properties",
                      'units': "kpc/h * km/s",
                      'order': 38,
                  }),
                 ('jH2', {
                      'type': np.float32,
                      'label': "j H2",
                      'description': "Specific angular momentum of molecular hydrogen in the gas disk",
                      'group': "Galaxy Properties",
                      'units': "kpc/h * km/s",
                      'order': 39,
                  }),
                ('SpinStars_x', {
                      'type': np.float32,
                      'label': "X Spin of Stellar Disk",
                      'description': "Normalised x-axis component of the stellar disk spin vector",
                      'group': "Galaxy Properties",
                      'order': 40,
                  }),
                 ('SpinStars_y', {
                      'type': np.float32,
                      'label': "Y Spin of Stellar Disk",
                      'description': "Normalised y-axis component of the stellar disk spin vector",
                      'group': "Galaxy Properties",
                      'order': 41,
                  }),
                 ('SpinStars_z', {
                      'type': np.float32,
                      'label': "Z Spin of Stellar Disk",
                      'description': "Normalised z-axis component of the stellar disk spin vector",
                      'group': "Galaxy Properties",
                      'order': 42,
                  }),
                 ('SpinGas_x', {
                      'type': np.float32,
                      'label': "X Spin of Gas Disk",
                      'description': "Normalised x-axis component of the gas disk spin vector",
                      'group': "Galaxy Properties",
                      'order': 43,
                  }),
                 ('SpinGas_y', {
                      'type': np.float32,
                      'label': "Y Spin of Gas Disk",
                      'description': "Normalised y-axis component of the gas disk spin vector",
                      'group': "Galaxy Properties",
                      'order': 44,
                  }),
                 ('SpinGas_z', {
                      'type': np.float32,
                      'label': "Z Spin of Gas Disk",
                      'description': "Normalised z-axis component of the gas disk spin vector",
                      'group': "Galaxy Properties",
                      'order': 45,
                  }),
                 ('Mvir', {
                        'type': np.float32,
                        'label': "Mvir",
                        'description': "Virial mass of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "10^10 Msun/h",
                        'order': 46,
                        }),
                ('Rvir', {
                        'type': np.float32,
                        'label': "Rvir",
                        'description': "Physical virial radius of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "Mpc/h",
                        'order': 47,
                        }),
                ('Vvir', {
                        'type': np.float32,
                        'label': "Vvir",
                        'description': "Virial speed of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 48,
                        }),
                ('Vmax', {
                        'type': np.float32,
                        'label': "Vmax",
                        'description': "Maximum circular speed of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 49,
                        }),
                ('Vpeak', {
                        'type': np.float32,
                        'label': "Vpeak",
                        'description': "Maximum circular velocity attained "
                        "in the assembly history (susceptible to spikes "
                        "during mergers)",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 50,
                        }),
                ('VelDisp', {
                        'type': np.float32,
                        'label': "Velocity Dispersion",
                        'description': "Velocity dispersion of the (sub)halo",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 51,
                        }),
                ('Spin_x', {
                        'type': np.float32,
                        'label': "jX Halo",
                        'description': "X-component of the (sub)halo's specific angular momentum",
                        'group': "Halo Properties",
                        'units': 'Mpc/h * km/s',
                        'order': 52,
                        }),
                ('Spin_y', {
                        'type': np.float32,
                        'label': "jY Halo",
                        'description': "Y-component of the (sub)halo's specific angular momentum",
                        'group': "Halo Properties",
                        'units': 'Mpc/h * km/s',
                        'order': 53,
                        }),
                ('Spin_z', {
                        'type': np.float32,
                        'label': "jZ Halo",
                        'description': "Z-component of the (sub)halo's specific angular momentum",
                        'group': "Halo Properties",
                        'units': 'Mpc/h * km/s',
                        'order': 54,
                        }),
                ('Len', {
                        'type': np.int32,
                        'label': "Total Particles",
                        'description': "Total number of simulation particles "\
                            "in the (sub)halo",
                        'group': "Halo Properties",
                        'order': 55,
                        }),
                 ('LenMax', {
                      'type': np.int32,
                      'label': "Maximum Number of Particles over History",
                      'description': "Maximum number of simulation particles in the (sub)halo over its entire existence up until this point",
                      'group': "Halo Properties",
                      'order': 56,
                  }),
                ('CentralMvir', {
                        'type': np.float32,
                        'label': "Central Galaxy Mvir",
                        'description': "Virial mass of the central-galaxy halo",
                        'group': "Halo Properties",
                        'units': "10^10 Msun/h",
                        'order': 57,
                        }),
                ('infallMvir', {
                        'type': np.float32,
                        'label': "Subhalo Mvir at Infall",
                        'description': "Virial mass of the (sub)halo at infall",
                        'group': "Halo Properties",
                        'units': "10^10 Msun/h",                        
                        'order': 58,
                        }),
                ('infallVvir', {
                        'type': np.float32,
                        'label': "Subhalo Vvir at Infall",
                        'description': "Virial speed of the (sub)halo at infall",
                        'group': "Halo Properties",
                        'units': "km/s",
                        'order': 59,
                        }),
                ('infallVmax', {
                        'type': np.float32,
                        'label': "Subhalo Vmax at Infall",
                        'description': "Maximum circular velocity of the (sub)halo at infall",
                        'group': "Halo Properties",
                        'units': "km/s",                        
                        'order': 60,
                        }),
                ('Pos_x', {
                        'type': np.float32,
                        'label': "X",
                        'description': "Co-moving X position of the (sub)halo",
                        'group': "Positions & Velocities",
                        'units': "Mpc/h",
                        'order': 61,
                        }),
                ('Pos_y', {
                        'type': np.float32,
                        'label': "Y",
                        'description': "Co-moving Y position of the (sub)halo",
                        'group': "Positions & Velocities",
                        'units': "Mpc/h",
                        'order': 62,
                        }),
                ('Pos_z', {
                        'type': np.float32,
                        'label': "Z",
                        'description': "Co-moving Z position of the (sub)halo",
                        'group': "Positions & Velocities",
                        'units': "Mpc/h",
                        'order': 63,
                        }),
                ('Vel_x', {
                        'type': np.float32,
                        'label': "X Velocity",
                        'description': "X component of the galaxy/(sub)halo velocity",
                        'group': "Positions & Velocities",
                        'units': "km/s",
                        'order': 64,
                        }),
                ('Vel_y', {
                        'type': np.float32,
                        'label': "Y Velocity",
                        'description': "Y component of the galaxy/(sub)halo velocity",
                        'group': "Positions & Velocities",
                        'units': "km/s",
                        'order': 65,
                        }),
                ('Vel_z', {
                        'type': np.float32,
                        'label': "Z Velocity",
                        'description': "Z component of the galaxy/(sub)halo velocity",
                        'group': "Positions & Velocities",
                        'units': "km/s",
                        'order': 66,
                        }),
                ('SnapNum', {
                        'type': np.int32,
                        'label': "Snapshot Number",
                        'description': "Snapshot number in the simulation",
                        'group': "Simulation",
                        'order': 67,
                        }),
                ('GalaxyIndex', {
                        'type': np.int64,
                        'label': "Galaxy ID",
                        'description': "A unique ID that stays with the galaxy/(sub)halo for its entire history",
                        'group': "Simulation",
                        'order': 68,
                        }),
                ('CentralGalaxyIndex', {
                        'type': np.int64,
                        'label': "Central Galaxy ID",
                        'description': "The unique Galaxy ID of the central galaxy this galaxy/subhalo belongs to",
                        'group': "Simulation",
                        'order': 69,
                        }),
                ('HaloIndex', {
                        'type': np.int32,
                        'label': "Halo Index",
                        'description': "An ID for the (sub)halo passed through from the original simulation",
                        'group': "Simulation",
                        'order': 70,
                        }),
                ('SimulationHaloIndex', {
                        'type': np.int32,
                        'label': "Simulation Halo ID",
                        'description': "Halo index within the tree file",
                        'group': "Simulation",
                        'order': 71,
                        }),
                ('TreeIndex', {
                        'type': np.int32,
                        'label': "Tree Index",
                        'description': "The index for the simulation tree file that this (sub)halo belongs to",
                        'group': "Simulation",
                        'order': -1,
                        }),
                 ('DiscRadii_0', {
                  'type': np.float32,
                  'label': "Annulus 1 Inner Radius",
                  'description': "Radius of the inner edge of annulus 1",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_1', {
                      'type': np.float32,
                      'label': "Annulus 1 Outer Radius",
                      'description': "Radius of the outer edge of annulus 1",
                      'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_2', {
                  'type': np.float32,
                  'label': "Annulus 2 Outer Radius",
                  'description': "Radius of the outer edge of annulus 2",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_3', {
                  'type': np.float32,
                  'label': "Annulus 3 Outer Radius",
                  'description': "Radius of the outer edge of annulus 3",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_4', {
                  'type': np.float32,
                  'label': "Annulus 4 Outer Radius",
                  'description': "Radius of the outer edge of annulus 4",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_5', {
                  'type': np.float32,
                  'label': "Annulus 5 Outer Radius",
                  'description': "Radius of the outer edge of annulus 5",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_6', {
                  'type': np.float32,
                  'label': "Annulus 6 Outer Radius",
                  'description': "Radius of the outer edge of annulus 6",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_7', {
                  'type': np.float32,
                  'label': "Annulus 7 Outer Radius",
                  'description': "Radius of the outer edge of annulus 7",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_8', {
                  'type': np.float32,
                  'label': "Annulus 8 Outer Radius",
                  'description': "Radius of the outer edge of annulus 8",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_9', {
                  'type': np.float32,
                  'label': "Annulus 9 Outer Radius",
                  'description': "Radius of the outer edge of annulus 9",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_10', {
                  'type': np.float32,
                  'label': "Annulus 10 Outer Radius",
                  'description': "Radius of the outer edge of annulus 10",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_11', {
                  'type': np.float32,
                  'label': "Annulus 11 Outer Radius",
                  'description': "Radius of the outer edge of annulus 11",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_12', {
                  'type': np.float32,
                  'label': "Annulus 12 Outer Radius",
                  'description': "Radius of the outer edge of annulus 12",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_13', {
                  'type': np.float32,
                  'label': "Annulus 13 Outer Radius",
                  'description': "Radius of the outer edge of annulus 13",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_14', {
                  'type': np.float32,
                  'label': "Annulus 14 Outer Radius",
                  'description': "Radius of the outer edge of annulus 14",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_15', {
                  'type': np.float32,
                  'label': "Annulus 15 Outer Radius",
                  'description': "Radius of the outer edge of annulus 15",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_16', {
                  'type': np.float32,
                  'label': "Annulus 16 Outer Radius",
                  'description': "Radius of the outer edge of annulus 16",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_17', {
                  'type': np.float32,
                  'label': "Annulus 17 Outer Radius",
                  'description': "Radius of the outer edge of annulus 17",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_18', {
                  'type': np.float32,
                  'label': "Annulus 18 Outer Radius",
                  'description': "Radius of the outer edge of annulus 18",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_19', {
                  'type': np.float32,
                  'label': "Annulus 19 Outer Radius",
                  'description': "Radius of the outer edge of annulus 19",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_20', {
                  'type': np.float32,
                  'label': "Annulus 20 Outer Radius",
                  'description': "Radius of the outer edge of annulus 20",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_21', {
                  'type': np.float32,
                  'label': "Annulus 21 Outer Radius",
                  'description': "Radius of the outer edge of annulus 21",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_22', {
                  'type': np.float32,
                  'label': "Annulus 22 Outer Radius",
                  'description': "Radius of the outer edge of annulus 22",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_23', {
                  'type': np.float32,
                  'label': "Annulus 23 Outer Radius",
                  'description': "Radius of the outer edge of annulus 23",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_24', {
                  'type': np.float32,
                  'label': "Annulus 24 Outer Radius",
                  'description': "Radius of the outer edge of annulus 24",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_25', {
                  'type': np.float32,
                  'label': "Annulus 25 Outer Radius",
                  'description': "Radius of the outer edge of annulus 25",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_26', {
                  'type': np.float32,
                  'label': "Annulus 26 Outer Radius",
                  'description': "Radius of the outer edge of annulus 26",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_27', {
                  'type': np.float32,
                  'label': "Annulus 27 Outer Radius",
                  'description': "Radius of the outer edge of annulus 27",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_28', {
                  'type': np.float32,
                  'label': "Annulus 28 Outer Radius",
                  'description': "Radius of the outer edge of annulus 28",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_29', {
                  'type': np.float32,
                  'label': "Annulus 29 Outer Radius",
                  'description': "Radius of the outer edge of annulus 29",
                  'group': "Internal",
                  'units': "Mpc/h",
                  'order': -1,
                  }),
                 ('DiscRadii_30', {
                      'type': np.float32,
                      'label': "Annulus 30 Outer Radius",
                      'description': "Radius of the outer edge of annulus 30",
                      'group': "Internal",
                      'units': "Mpc/h",
                      'order': -1,
                  }),
                 ('DiscStars_1', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 1",
                  'description': "Mass contained within annulus 1 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_2', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 2",
                  'description': "Mass contained within annulus 2 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_3', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 3",
                  'description': "Mass contained within annulus 3 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_4', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 4",
                  'description': "Mass contained within annulus 4 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_5', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 5",
                  'description': "Mass contained within annulus 5 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_6', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 6",
                  'description': "Mass contained within annulus 6 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_7', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 7",
                  'description': "Mass contained within annulus 7 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_8', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 8",
                  'description': "Mass contained within annulus 8 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_9', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 9",
                  'description': "Mass contained within annulus 9 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_10', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 10",
                  'description': "Mass contained within annulus 10 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_11', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 11",
                  'description': "Mass contained within annulus 11 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_12', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 12",
                  'description': "Mass contained within annulus 12 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_13', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 13",
                  'description': "Mass contained within annulus 13 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_14', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 14",
                  'description': "Mass contained within annulus 14 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_15', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 15",
                  'description': "Mass contained within annulus 15 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_16', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 16",
                  'description': "Mass contained within annulus 16 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_17', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 17",
                  'description': "Mass contained within annulus 17 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_18', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 18",
                  'description': "Mass contained within annulus 18 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_19', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 19",
                  'description': "Mass contained within annulus 19 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_20', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 20",
                  'description': "Mass contained within annulus 20 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_21', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 21",
                  'description': "Mass contained within annulus 21 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_22', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 22",
                  'description': "Mass contained within annulus 22 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_23', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 23",
                  'description': "Mass contained within annulus 23 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_24', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 24",
                  'description': "Mass contained within annulus 24 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_25', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 25",
                  'description': "Mass contained within annulus 25 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_26', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 26",
                  'description': "Mass contained within annulus 26 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_27', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 27",
                  'description': "Mass contained within annulus 27 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_28', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 28",
                  'description': "Mass contained within annulus 28 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_29', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 29",
                  'description': "Mass contained within annulus 29 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStars_30', {
                  'type': np.float32,
                  'label': "Stellar Mass Annulus 30",
                  'description': "Mass contained within annulus 30 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_1', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 1",
                  'description': "Mass contained within annulus 1 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_2', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 2",
                  'description': "Mass contained within annulus 2 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_3', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 3",
                  'description': "Mass contained within annulus 3 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_4', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 4",
                  'description': "Mass contained within annulus 4 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_5', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 5",
                  'description': "Mass contained within annulus 5 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_6', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 6",
                  'description': "Mass contained within annulus 6 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_7', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 7",
                  'description': "Mass contained within annulus 7 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_8', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 8",
                  'description': "Mass contained within annulus 8 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_9', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 9",
                  'description': "Mass contained within annulus 9 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_10', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 10",
                  'description': "Mass contained within annulus 10 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_11', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 11",
                  'description': "Mass contained within annulus 11 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_12', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 12",
                  'description': "Mass contained within annulus 12 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_13', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 13",
                  'description': "Mass contained within annulus 13 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_14', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 14",
                  'description': "Mass contained within annulus 14 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_15', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 15",
                  'description': "Mass contained within annulus 15 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_16', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 16",
                  'description': "Mass contained within annulus 16 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_17', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 17",
                  'description': "Mass contained within annulus 17 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_18', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 18",
                  'description': "Mass contained within annulus 18 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_19', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 19",
                  'description': "Mass contained within annulus 19 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_20', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 20",
                  'description': "Mass contained within annulus 20 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_21', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 21",
                  'description': "Mass contained within annulus 21 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_22', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 22",
                  'description': "Mass contained within annulus 22 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_23', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 23",
                  'description': "Mass contained within annulus 23 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_24', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 24",
                  'description': "Mass contained within annulus 24 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_25', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 25",
                  'description': "Mass contained within annulus 25 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_26', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 26",
                  'description': "Mass contained within annulus 26 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_27', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 27",
                  'description': "Mass contained within annulus 27 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_28', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 28",
                  'description': "Mass contained within annulus 28 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_29', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 29",
                  'description': "Mass contained within annulus 29 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGas_30', {
                  'type': np.float32,
                  'label': "Gas Mass Annulus 30",
                  'description': "Mass contained within annulus 30 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_1', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 1",
                  'description': "Mass in the form of atomic hydrogen in annulus 1 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_2', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 2",
                  'description': "Mass in the form of atomic hydrogen in annulus 2 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_3', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 3",
                  'description': "Mass in the form of atomic hydrogen in annulus 3 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_4', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 4",
                  'description': "Mass in the form of atomic hydrogen in annulus 4 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_5', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 5",
                  'description': "Mass in the form of atomic hydrogen in annulus 5 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_6', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 6",
                  'description': "Mass in the form of atomic hydrogen in annulus 6 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_7', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 7",
                  'description': "Mass in the form of atomic hydrogen in annulus 7 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_8', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 8",
                  'description': "Mass in the form of atomic hydrogen in annulus 8 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_9', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 9",
                  'description': "Mass in the form of atomic hydrogen in annulus 9 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_10', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 10",
                  'description': "Mass in the form of atomic hydrogen in annulus 10 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_11', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 11",
                  'description': "Mass in the form of atomic hydrogen in annulus 11 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_12', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 12",
                  'description': "Mass in the form of atomic hydrogen in annulus 12 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_13', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 13",
                  'description': "Mass in the form of atomic hydrogen in annulus 13 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_14', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 14",
                  'description': "Mass in the form of atomic hydrogen in annulus 14 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_15', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 15",
                  'description': "Mass in the form of atomic hydrogen in annulus 15 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_16', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 16",
                  'description': "Mass in the form of atomic hydrogen in annulus 16 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_17', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 17",
                  'description': "Mass in the form of atomic hydrogen in annulus 17 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_18', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 18",
                  'description': "Mass in the form of atomic hydrogen in annulus 18 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_19', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 19",
                  'description': "Mass in the form of atomic hydrogen in annulus 19 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_20', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 20",
                  'description': "Mass in the form of atomic hydrogen in annulus 20 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_21', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 21",
                  'description': "Mass in the form of atomic hydrogen in annulus 21 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_22', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 22",
                  'description': "Mass in the form of atomic hydrogen in annulus 22 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_23', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 23",
                  'description': "Mass in the form of atomic hydrogen in annulus 23 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_24', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 24",
                  'description': "Mass in the form of atomic hydrogen in annulus 24 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_25', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 25",
                  'description': "Mass in the form of atomic hydrogen in annulus 25 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_26', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 26",
                  'description': "Mass in the form of atomic hydrogen in annulus 26 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_27', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 27",
                  'description': "Mass in the form of atomic hydrogen in annulus 27 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_28', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 28",
                  'description': "Mass in the form of atomic hydrogen in annulus 28 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_29', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 29",
                  'description': "Mass in the form of atomic hydrogen in annulus 29 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscHI_30', {
                  'type': np.float32,
                  'label': "HI Mass Annulus 30",
                  'description': "Mass in the form of atomic hydrogen in annulus 30 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_1', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 1",
                  'description': "Mass in the form of molecular hydrogen in annulus 1 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_2', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 2",
                  'description': "Mass in the form of molecular hydrogen in annulus 2 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_3', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 3",
                  'description': "Mass in the form of molecular hydrogen in annulus 3 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_4', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 4",
                  'description': "Mass in the form of molecular hydrogen in annulus 4 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_5', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 5",
                  'description': "Mass in the form of molecular hydrogen in annulus 5 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_6', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 6",
                  'description': "Mass in the form of molecular hydrogen in annulus 6 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_7', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 7",
                  'description': "Mass in the form of molecular hydrogen in annulus 7 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_8', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 8",
                  'description': "Mass in the form of molecular hydrogen in annulus 8 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_9', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 9",
                  'description': "Mass in the form of molecular hydrogen in annulus 9 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_10', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 10",
                  'description': "Mass in the form of molecular hydrogen in annulus 10 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_11', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 11",
                  'description': "Mass in the form of molecular hydrogen in annulus 11 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_12', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 12",
                  'description': "Mass in the form of molecular hydrogen in annulus 12 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_13', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 13",
                  'description': "Mass in the form of molecular hydrogen in annulus 13 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_14', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 14",
                  'description': "Mass in the form of molecular hydrogen in annulus 14 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_15', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 15",
                  'description': "Mass in the form of molecular hydrogen in annulus 15 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_16', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 16",
                  'description': "Mass in the form of molecular hydrogen in annulus 16 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_17', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 17",
                  'description': "Mass in the form of molecular hydrogen in annulus 17 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_18', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 18",
                  'description': "Mass in the form of molecular hydrogen in annulus 18 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_19', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 19",
                  'description': "Mass in the form of molecular hydrogen in annulus 19 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_20', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 20",
                  'description': "Mass in the form of molecular hydrogen in annulus 20 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_21', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 21",
                  'description': "Mass in the form of molecular hydrogen in annulus 21 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_22', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 22",
                  'description': "Mass in the form of molecular hydrogen in annulus 22 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_23', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 23",
                  'description': "Mass in the form of molecular hydrogen in annulus 23 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_24', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 24",
                  'description': "Mass in the form of molecular hydrogen in annulus 24 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_25', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 25",
                  'description': "Mass in the form of molecular hydrogen in annulus 25 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_26', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 26",
                  'description': "Mass in the form of molecular hydrogen in annulus 26 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_27', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 27",
                  'description': "Mass in the form of molecular hydrogen in annulus 27 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_28', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 28",
                  'description': "Mass in the form of molecular hydrogen in annulus 28 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_29', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 29",
                  'description': "Mass in the form of molecular hydrogen in annulus 29 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscH2_30', {
                  'type': np.float32,
                  'label': "H2 Mass Annulus 30",
                  'description': "Mass in the form of molecular hydrogen in annulus 30 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscSFR_1', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 1",
                  'description': "Average star formation rate over last time-step within annulus 1 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_2', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 2",
                  'description': "Average star formation rate over last time-step within annulus 2 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_3', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 3",
                  'description': "Average star formation rate over last time-step within annulus 3 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_4', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 4",
                  'description': "Average star formation rate over last time-step within annulus 4 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_5', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 5",
                  'description': "Average star formation rate over last time-step within annulus 5 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_6', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 6",
                  'description': "Average star formation rate over last time-step within annulus 6 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_7', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 7",
                  'description': "Average star formation rate over last time-step within annulus 7 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_8', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 8",
                  'description': "Average star formation rate over last time-step within annulus 8 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_9', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 9",
                  'description': "Average star formation rate over last time-step within annulus 9 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_10', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 10",
                  'description': "Average star formation rate over last time-step within annulus 10 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_11', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 11",
                  'description': "Average star formation rate over last time-step within annulus 11 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_12', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 12",
                  'description': "Average star formation rate over last time-step within annulus 12 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_13', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 13",
                  'description': "Average star formation rate over last time-step within annulus 13 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_14', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 14",
                  'description': "Average star formation rate over last time-step within annulus 14 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_15', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 15",
                  'description': "Average star formation rate over last time-step within annulus 15 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_16', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 16",
                  'description': "Average star formation rate over last time-step within annulus 16 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_17', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 17",
                  'description': "Average star formation rate over last time-step within annulus 17 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_18', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 18",
                  'description': "Average star formation rate over last time-step within annulus 18 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_19', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 19",
                  'description': "Average star formation rate over last time-step within annulus 19 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_20', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 20",
                  'description': "Average star formation rate over last time-step within annulus 20 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_21', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 21",
                  'description': "Average star formation rate over last time-step within annulus 21 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_22', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 22",
                  'description': "Average star formation rate over last time-step within annulus 22 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_23', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 23",
                  'description': "Average star formation rate over last time-step within annulus 23 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_24', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 24",
                  'description': "Average star formation rate over last time-step within annulus 24 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_25', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 25",
                  'description': "Average star formation rate over last time-step within annulus 25 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_26', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 26",
                  'description': "Average star formation rate over last time-step within annulus 26 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_27', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 27",
                  'description': "Average star formation rate over last time-step within annulus 27 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_28', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 28",
                  'description': "Average star formation rate over last time-step within annulus 28 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_29', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 29",
                  'description': "Average star formation rate over last time-step within annulus 29 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscSFR_30', {
                  'type': np.float32,
                  'label': "Star Formation Rate Annulus 30",
                  'description': "Average star formation rate over last time-step within annulus 30 of the gas disc",
                  'group': "Internal",
                  'units': "Msun/yr",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_1', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 1",
                  'description': "Mass of metals contained within annulus 1 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_2', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 2",
                  'description': "Mass of metals contained within annulus 2 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_3', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 3",
                  'description': "Mass of metals contained within annulus 3 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_4', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 4",
                  'description': "Mass of metals contained within annulus 4 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_5', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 5",
                  'description': "Mass of metals contained within annulus 5 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_6', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 6",
                  'description': "Mass of metals contained within annulus 6 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_7', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 7",
                  'description': "Mass of metals contained within annulus 7 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_8', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 8",
                  'description': "Mass of metals contained within annulus 8 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_9', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 9",
                  'description': "Mass of metals contained within annulus 9 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_10', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 10",
                  'description': "Mass of metals contained within annulus 10 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_11', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 11",
                  'description': "Mass of metals contained within annulus 11 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_12', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 12",
                  'description': "Mass of metals contained within annulus 12 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_13', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 13",
                  'description': "Mass of metals contained within annulus 13 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_14', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 14",
                  'description': "Mass of metals contained within annulus 14 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_15', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 15",
                  'description': "Mass of metals contained within annulus 15 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_16', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 16",
                  'description': "Mass of metals contained within annulus 16 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_17', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 17",
                  'description': "Mass of metals contained within annulus 17 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_18', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 18",
                  'description': "Mass of metals contained within annulus 18 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_19', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 19",
                  'description': "Mass of metals contained within annulus 19 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_20', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 20",
                  'description': "Mass of metals contained within annulus 20 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_21', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 21",
                  'description': "Mass of metals contained within annulus 21 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_22', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 22",
                  'description': "Mass of metals contained within annulus 22 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_23', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 23",
                  'description': "Mass of metals contained within annulus 23 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_24', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 24",
                  'description': "Mass of metals contained within annulus 24 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_25', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 25",
                  'description': "Mass of metals contained within annulus 25 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_26', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 26",
                  'description': "Mass of metals contained within annulus 26 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_27', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 27",
                  'description': "Mass of metals contained within annulus 27 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_28', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 28",
                  'description': "Mass of metals contained within annulus 28 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_29', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 29",
                  'description': "Mass of metals contained within annulus 29 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscStarsMetals_30', {
                  'type': np.float32,
                  'label': "Metals Stellar Mass Annulus 30",
                  'description': "Mass of metals contained within annulus 30 of the stellar disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_1', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 1",
                  'description': "Mass of metals contained within annulus 1 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_2', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 2",
                  'description': "Mass of metals contained within annulus 2 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_3', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 3",
                  'description': "Mass of metals contained within annulus 3 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_4', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 4",
                  'description': "Mass of metals contained within annulus 4 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_5', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 5",
                  'description': "Mass of metals contained within annulus 5 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_6', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 6",
                  'description': "Mass of metals contained within annulus 6 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_7', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 7",
                  'description': "Mass of metals contained within annulus 7 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_8', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 8",
                  'description': "Mass of metals contained within annulus 8 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_9', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 9",
                  'description': "Mass of metals contained within annulus 9 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_10', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 10",
                  'description': "Mass of metals contained within annulus 10 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_11', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 11",
                  'description': "Mass of metals contained within annulus 11 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_12', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 12",
                  'description': "Mass of metals contained within annulus 12 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_13', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 13",
                  'description': "Mass of metals contained within annulus 13 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_14', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 14",
                  'description': "Mass of metals contained within annulus 14 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_15', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 15",
                  'description': "Mass of metals contained within annulus 15 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_16', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 16",
                  'description': "Mass of metals contained within annulus 16 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_17', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 17",
                  'description': "Mass of metals contained within annulus 17 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_18', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 18",
                  'description': "Mass of metals contained within annulus 18 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_19', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 19",
                  'description': "Mass of metals contained within annulus 19 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_20', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 20",
                  'description': "Mass of metals contained within annulus 20 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_21', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 21",
                  'description': "Mass of metals contained within annulus 21 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_22', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 22",
                  'description': "Mass of metals contained within annulus 22 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_23', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 23",
                  'description': "Mass of metals contained within annulus 23 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_24', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 24",
                  'description': "Mass of metals contained within annulus 24 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_25', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 25",
                  'description': "Mass of metals contained within annulus 25 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_26', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 26",
                  'description': "Mass of metals contained within annulus 26 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_27', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 27",
                  'description': "Mass of metals contained within annulus 27 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_28', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 28",
                  'description': "Mass of metals contained within annulus 28 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_29', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 29",
                  'description': "Mass of metals contained within annulus 29 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('DiscGasMetals_30', {
                  'type': np.float32,
                  'label': "Metals Gas Mass Annulus 30",
                  'description': "Mass of metals contained within annulus 30 of the gas disc",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('mergeIntoID', {
                        'type': np.int32,
                        'label': "Descendant Galaxy Index",
                        'description': "Index for the descendant galaxy "\
                            "after a merger",
                        'group': "Internal",
                        'order': 50,
                        }),
                ('mergeIntoSnapNum', {
                        'type': np.int32,
                        'label': "Descendant Snapshot",
                        'description': "Snapshot for the descendant galaxy",
                        'group': "Internal",
                        'order': 51,
                        }),
                ('mergeType', {
                        'type': np.int32,
                        'label': "Merger Type",
                        'description': "Merger type: "\
                            "0=none; 1=minor merger; 2=major merger; "\
                            "3=disk instability; 4=disrupt to ICS",
                        'group': "Internal",
                        'order': 52,
                        }),
                ('dT', {
                        'type': np.float32,
                        'label': "Galaxy Age",
                        'group': "Internal",
                        'order': 53,
                        }),
               ('SfrDisk', {
                        'type': np.float32,
                        'label': "Star Formation Rate in the Disk",
                        'description': "Star formation rate in the disk",
                        'group': "Internal",
                        'units': "Msun/year",
                        'order': -1,
                        }),
               ('SfrBulge', {
                        'type': np.float32,
                        'label': "Star formation Rate in the Bulge",
                        'description': "Star formation rate in the bulge",
                        'group': "Internal",
                        'units': "Msun/year",
                        'order': -1,
                        }),
               ('SfrDiskZ', {
                        'type': np.float32,
                        'label': "Avg. Metallicity of Star-forming Disk Gas",
                        'description': "Metallicty of star forming disk gas "\
                        "(averaged over timesteps between two snapshots)"\
                        "(Mass of metals)/(Mass of star forming disk gas)",
                        'group': "Internal",
                        'units': "fraction",
                        'order': -1,
                        }),
               ('SfrBulgeZ', {
                        'type': np.float32,
                        'label': "Avg. Metallicity of Star-forming Bulge Gas",
                        'description': "Metallicty of star forming bulge gas "\
                        "(averaged over timesteps between two snapshots)"\
                        "(Mass of metals)/(Mass of star forming bulge gas)",
                        'group': "Internal",
                        'units': "fraction",
                        'order': -1,
                        }),
                ('BulgeRadius', {
                     'type': np.float32,
                     'label': "Bulge Radius",
                     'description': "Approximate radius of the bulge",
                     'group': "Internal",
                     'units': "Mpc/h",
                     'order': -1,
                     }),
                 ('SpinInstabilityBulge_x', {
                        'type': np.float32,
                        'label': "X Spin Instability Bulge",
                        'description': "X component of normalised spin vector for instability-drive bulge",
                        'group': "Internal",
                        'order': -1,
                  }),
                 ('SpinInstabilityBulge_y', {
                      'type': np.float32,
                      'label': "Y Spin Instability Bulge",
                      'description': "Y component of normalised spin vector for instability-drive bulge",
                      'group': "Internal",
                      'order': -1,
                  }),
                 ('SpinInstabilityBulge_z', {
                      'type': np.float32,
                      'label': "Z Spin Instability Bulge",
                      'description': "Z component of normalised spin vector for instability-drive bulge",
                      'group': "Internal",
                      'order': -1,
                  }),
                 ('SpinMergerBulge_x', {
                      'type': np.float32,
                      'label': "X Spin Merger Bulge",
                      'description': "X component of normalised spin vector for merger-drive bulge",
                      'group': "Internal",
                      'order': -1,
                  }),
                 ('SpinMergerBulge_y', {
                      'type': np.float32,
                      'label': "Y Spin Merger Bulge",
                      'description': "Y component of normalised spin vector for merger-drive bulge",
                      'group': "Internal",
                      'order': -1,
                  }),
                 ('SpinMergerBulge_z', {
                      'type': np.float32,
                      'label': "Z Spin Merger Bulge",
                      'description': "Z component of normalised spin vector for merger-drive bulge",
                      'group': "Internal",
                      'order': -1,
                  }),
                 ('StarsInSitu', {
                      'type': np.float32,
                      'label': "Passive Stellar Mass",
                      'description': "Mass of stars formed through passive star formation from H2",
                      'group': "Internal",
                      'units': "10^10 Msun/h",
                      'order': -1,
                  }),
                 ('StarsInstability', {
                      'type': np.float32,
                      'label': "Instability Stellar Mass",
                      'description': "Mass of stars formed through disc instabilities",
                      'group': "Internal",
                      'units': "10^10 Msun/h",
                        'order': -1,
                  }),
                 ('StarsMergeBurst', {
                      'type': np.float32,
                      'label': "Merge Burst Stellar Mass",
                      'description': "Mass of stars formed through merger-driven starbursts",
                      'group': "Internal",
                      'units': "10^10 Msun/h",
                      'order': -1,
                  }),
                 ('AccretedGasMass', {
                  'type': np.float32,
                  'label': "Accreted Gas Mass",
                  'description': "Total mass of all gas that has cooled onto all progenitors of this galaxy",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('EjectedSNGasMass', {
                  'type': np.float32,
                  'label': "SN Ejected Gas Mass",
                  'description': "Total mass of all gas reheated by stellar feedback for all progenitors of this galaxy",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('EjectedQuasarGasMass', {
                  'type': np.float32,
                  'label': "SN Ejected Gas Mass",
                  'description': "Total mass of all gas reheated by quasar AGN feedback for all progenitors of this galaxy",
                  'group': "Internal",
                  'units': "10^10 Msun/h",
                  'order': -1,
                  }),
                 ('TotInstabEvents', {
                  'type': np.int32,
                  'label': "Total Number of Instability Events",
                  'description': "Total number of instability events for the main progenitor of this galaxy",
                  'group': "Internal",
                  'order': -1,
                  }),
                 ('TotInstabEventsGas', {
                  'type': np.int32,
                  'label': "Total Number of Gas Instability Events",
                  'description': "Total number of instability events in the gas disc for the main progenitor of this galaxy",
                  'group': "Internal",
                  'order': -1,
                  }),
                 ('TotInstabEventsStar', {
                  'type': np.int32,
                  'label': "Total Number of Stellar Instability Events",
                  'description': "Total number of instability events in the stellar disc for the main progenitor of this galaxy",
                  'group': "Internal",
                  'order': -1,
                  }),
                 ('TotInstabAnnuliGas', {
                  'type': np.int32,
                  'label': "Historical Sum of Unstable Gas Annuli",
                  'description': "Total number of gas annuli that have undergone an instability over the main progenitor's history",
                  'group': "Internal",
                  'order': -1,
                  }),
                 ('TotInstabAnnuliStar', {
                  'type': np.int32,
                  'label': "Historical Sum of Unstable Stellar Annuli",
                  'description': "Total number of stellar annuli that have undergone an instability over the main progenitor's history",
                  'group': "Internal",
                  'order': -1,
                  }),
                 ('FirstUnstableAvGas', {
                  'type': np.float32,
                  'label': "Average First Unstable Gas Annulus",
                  'description': "The lowest annulus to be unstable when an instability occurs in the gas disc, averaged over the history of the main progenitor",
                  'group': "Internal",
                  'order': -1,
                  }),
                  ('FirstUnstableAvStar', {
                   'type': np.float32,
                   'label': "Average First Unstable Stellar Annulus",
                   'description': "The lowest annulus to be unstable when an instability occurs in the stellar disc, averaged over the history of the main progenitor",
                   'group': "Internal",
                   'order': -1,
                  }),
                  ('TotSinkGas_1', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 1",
                   'description': "Total gas moved out of annulus 1 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_2', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 2",
                   'description': "Total gas moved out of annulus 2 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_3', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 3",
                   'description': "Total gas moved out of annulus 3 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_4', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 4",
                   'description': "Total gas moved out of annulus 4 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_5', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 5",
                   'description': "Total gas moved out of annulus 5 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_6', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 6",
                   'description': "Total gas moved out of annulus 6 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_7', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 7",
                   'description': "Total gas moved out of annulus 7 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_8', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 8",
                   'description': "Total gas moved out of annulus 8 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_9', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 9",
                   'description': "Total gas moved out of annulus 9 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_10', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 10",
                   'description': "Total gas moved out of annulus 10 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_11', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 11",
                   'description': "Total gas moved out of annulus 11 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_12', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 12",
                   'description': "Total gas moved out of annulus 12 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_13', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 13",
                   'description': "Total gas moved out of annulus 13 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_14', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 14",
                   'description': "Total gas moved out of annulus 14 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_15', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 15",
                   'description': "Total gas moved out of annulus 15 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_16', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 16",
                   'description': "Total gas moved out of annulus 16 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_17', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 17",
                   'description': "Total gas moved out of annulus 17 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_18', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 18",
                   'description': "Total gas moved out of annulus 18 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_19', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 19",
                   'description': "Total gas moved out of annulus 19 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_20', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 20",
                   'description': "Total gas moved out of annulus 20 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_21', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 21",
                   'description': "Total gas moved out of annulus 21 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_22', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 22",
                   'description': "Total gas moved out of annulus 22 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_23', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 23",
                   'description': "Total gas moved out of annulus 23 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_24', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 24",
                   'description': "Total gas moved out of annulus 24 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_25', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 25",
                   'description': "Total gas moved out of annulus 25 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_26', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 26",
                   'description': "Total gas moved out of annulus 26 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_27', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 27",
                   'description': "Total gas moved out of annulus 27 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_28', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 28",
                   'description': "Total gas moved out of annulus 28 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_29', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 29",
                   'description': "Total gas moved out of annulus 29 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkGas_30', {
                   'type': np.float32,
                   'label': "Moved Gas From Instabilities Annulus 30",
                   'description': "Total gas moved out of annulus 30 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_1', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 1",
                   'description': "Total stellar mass moved out of annulus 1 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_2', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 2",
                   'description': "Total stellar mass moved out of annulus 2 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_3', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 3",
                   'description': "Total stellar mass moved out of annulus 3 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_4', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 4",
                   'description': "Total stellar mass moved out of annulus 4 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_5', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 5",
                   'description': "Total stellar mass moved out of annulus 5 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_6', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 6",
                   'description': "Total stellar mass moved out of annulus 6 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_7', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 7",
                   'description': "Total stellar mass moved out of annulus 7 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_8', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 8",
                   'description': "Total stellar mass moved out of annulus 8 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_9', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 9",
                   'description': "Total stellar mass moved out of annulus 9 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_10', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 10",
                   'description': "Total stellar mass moved out of annulus 10 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_11', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 11",
                   'description': "Total stellar mass moved out of annulus 11 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_12', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 12",
                   'description': "Total stellar mass moved out of annulus 12 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_13', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 13",
                   'description': "Total stellar mass moved out of annulus 13 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_14', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 14",
                   'description': "Total stellar mass moved out of annulus 14 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_15', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 15",
                   'description': "Total stellar mass moved out of annulus 15 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_16', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 16",
                   'description': "Total stellar mass moved out of annulus 16 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_17', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 17",
                   'description': "Total stellar mass moved out of annulus 17 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_18', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 18",
                   'description': "Total stellar mass moved out of annulus 18 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_19', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 19",
                   'description': "Total stellar mass moved out of annulus 19 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_20', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 20",
                   'description': "Total stellar mass moved out of annulus 20 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_21', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 21",
                   'description': "Total stellar mass moved out of annulus 21 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_22', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 22",
                   'description': "Total stellar mass moved out of annulus 22 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_23', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 23",
                   'description': "Total stellar mass moved out of annulus 23 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_24', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 24",
                   'description': "Total stellar mass moved out of annulus 24 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_25', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 25",
                   'description': "Total stellar mass moved out of annulus 25 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_26', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 26",
                   'description': "Total stellar mass moved out of annulus 26 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_27', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 27",
                   'description': "Total stellar mass moved out of annulus 27 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_28', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 28",
                   'description': "Total stellar mass moved out of annulus 28 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_29', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 29",
                   'description': "Total stellar mass moved out of annulus 29 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   }),
                  ('TotSinkStar_30', {
                   'type': np.float32,
                   'label': "Moved Stars From Instabilities Annulus 30",
                   'description': "Total stellar mass moved out of annulus 30 from instabilities",
                   'group': "Internal",
                   'units': "10^10 Msun/h",
                   'order': -1,
                   })
                 ])
        
        self.src_fields_dict = src_fields_dict
        super(DARKSAGEConverter, self).__init__(*args, **kwargs)

    @classmethod
    def add_arguments(cls, parser):
        """Adds extra arguments required for SAGE conversion.

        Extra arguments required for conversion are:
          1. The location of the SAGE output trees.
          2. The simulation box size.
          3. The list of expansion factors (a-list).
          4. The SAGE parameters file.
          5. The name of the simulation (dark matter/hydro).
          6. The name of the galaxy formation model (simulation name in case of hydro)
        """

        parser.add_argument('--trees-dir', default='.',
                            help='location of SAGE trees')
        parser.add_argument('--box-size', help='simulation box-size')
        parser.add_argument('--a-list', help='a-list file')
        parser.add_argument('--parameters', help='SAGE parameter file')
        parser.add_argument('--sim-name', help='name of the dark matter or '
                            'hydro simulation')
        parser.add_argument('--model-name', help='name of the SAM. Set to '
                            'simulation name for a hydro sim')

    def get_simulation_data(self):
        """Extract simulation data.

        Extracts the simulation data from the SAGE parameter file and
        returns a dictionary containing the values.
        """

        if not self.args.box_size:
            raise tao.ConversionError('Must specify a box-size')
        if not self.args.parameters:
            raise tao.ConversionError('Must specify a SAGE parameter file')
        par = open(self.args.parameters, 'r').read()
        hubble = np.float32(re.search(r'Hubble_h\s+(\d*\.?\d*)',
                                      par, re.I).group(1))
        if hubble < 1.0:
            hubble = hubble * 100.0
        msg = 'Hubble parameter must be in physical units (not little h)'
        assert hubble > 1.0, msg
        hubble = str(hubble)
        omega_m = re.search(r'Omega\s+(\d*\.?\d*)', par, re.I).group(1)
        omega_l = re.search(r'OmegaLambda\s+(\d*\.?\d*)', par, re.I).group(1)
        
        return {
            'box_size': self.args.box_size,
            'hubble': hubble,
            'omega_m': omega_m,
            'omega_l': omega_l,
        }

    def get_jbins(self):
        par = open(self.args.parameters, 'r').read()
        FirstBin = np.float32(re.search(r'FirstBin\s+(\d*\.?\d*)', par, re.I).group(1))
        ExponentBin = np.float32(re.search(r'ExponentBin\s+(\d*\.?\d*)', par, re.I).group(1))
        DiscBinEdge = np.append(0, np.array([FirstBin*ExponentBin**i for i in range(30)]))
        j_bin = (DiscBinEdge[1:]+DiscBinEdge[:-1])/2.
        h = np.float32(re.search(r'Hubble_h\s+(\d*\.?\d*)', par, re.I).group(1))
        return j_bin, h
    
    def get_snapshot_redshifts(self):
        """Parse and convert the expansion factors.

        Uses the expansion factors to calculate snapshot redshifts. Returns
        a list of redshifts in order of snapshots.
        """

        if not self.args.a_list:
            raise tao.ConversionError('Must specify a filename for the a-list')
        redshifts = []
        with open(self.args.a_list, 'r') as file:
            for line in file:
                redshifts.append(1.0 / float(line) - 1.0)
        return redshifts

    def get_mapping_table(self):
        """Returns a mapping from TAO fields to SAGE fields."""

        mapping = {'posx': 'Pos_x',
                   'posy': 'Pos_y',
                   'posz': 'Pos_z',
                   'velx': 'Vel_x',
                   'vely': 'Vel_y',
                   'velz': 'Vel_z',
                   'snapnum': 'SnapNum',
                   'mergetype': 'mergeType',
                   'dt': 'dT',
                   'sfrdisk': 'SfrDisk',
                   'sfrbulge': 'SfrBulge',
                   'sfrdiskz': 'SfrDiskZ',
                   'sfrbulgez': 'SfrBulgeZ',
                   'coldgas': 'ColdGas',
                   'metalscoldgas': 'MetalsColdGas',
                   'diskscaleradius': 'DiskScaleRadius',
                   'objecttype': 'ObjectType'
                   }

        return mapping

    def get_extra_fields(self):
        """Returns a list of SAGE fields and types to include."""
        wanted_field_keys = [
            'GalaxyIndex',
            'CentralGalaxyIndex',
            'SimulationHaloIndex',
            'mergeIntoID',
            'mergeIntoSnapNum',
            'Spin_x',
            'Spin_y',
            'Spin_z',
            'Len',
            'LenMax',
            'Mvir',
            'CentralMvir',
            'Rvir',
            'Vvir',
            'Vmax',
            'VelDisp',
            'StellarMass',
            'MergerBulgeMass',
            'InstabilityBulgeMass',
            'HotGas',
            'EjectedMass',
            'BlackHoleMass',
            'ICS',
            'MetalsStellarMass',
            'MetalsMergerBulgeMass',
            'MetalsInstabilityBulgeMass',
            'MetalsHotGas',
            'MetalsEjectedMass',
            'MetalsICS',
            'MetalsStellarDiscMass', 'MetalsPseudoBulge',
            'Cooling',
            'Heating',
            'TimeofLastMajorMerger',
            'OutflowRate',
            'infallMvir',
            'infallVvir',
            'infallVmax',
            'TotSfr',
            'Vpeak',
             'HImass',
             'H2mass',
             'StellarDiscMass', 'PseudoBulgeMass',
             'SpinStars_x', 'SpinStars_y', 'SpinStars_z',
             'SpinGas_x', 'SpinGas_y', 'SpinGas_z',
             'jStarDisc', 'jPseudoBulge', 'jGas', 'jHI', 'jH2',
             'RadiusHI', 'RadiusTrans', 'r50', 'r90', 'rSFR',
             'dZStar', 'dZGas'
        ]

        fields = OrderedDict()
        for k in wanted_field_keys:
            try:
                fields[k] = self.src_fields_dict[k]
            except KeyError:
                try:
                    fields[k] = self.src_fields_dict[k.lower()]
                except:
                    raise

        return fields

    def map_descendant(self, tree):
        """Calculate the SAGE tree structure.

        SAGE does not output the descendant information in its tree files
        in a directly usable format. To calculate it we need to capitalise
        on the snapshot ordering of the input data, the GalaxyIndex field,
        and the mergeIntoID field.
        """
        descs = np.empty(len(tree), np.int32)
        descs.fill(-1)

        """
        Now my attempt at this mapping descendants
        First, sort the entire tree into using GalaxyIndex as
        primary key and then snapshot number as secondary key.
        This sorted indices will naturally flow a galaxy from
        earlier times (lower snapshot numbers) to later times (larger
        snapshot number)
        """
        sorted_ind = np.argsort(tree, order=('GalaxyIndex', 'SnapNum'))
        all_gal_idx = tree['GalaxyIndex']
        for ii, idx in enumerate(all_gal_idx[sorted_ind]):
            jj = ii + 1
            if (jj < len(tree)) and (idx == all_gal_idx[sorted_ind[jj]]):
                assert descs[sorted_ind[ii]] == -1
                # assert tree['SnapNum'][sorted_ind[jj]] > \
                #     tree['SnapNum'][sorted_ind[ii]]
                # assert tree['GalaxyIndex'][sorted_ind[ii]] == \
                #     tree['GalaxyIndex'][sorted_ind[jj]]
                descs[sorted_ind[ii]] = sorted_ind[jj]

        # Run validation on descendants
        # for ii, desc in enumerate(descs):
        #     if desc == -1:
        #         this_galidx = tree['GalaxyIndex'][ii]
        #         this_snapnum = tree['SnapNum'][ii]

        #         # No descendant -> there can not be any galaxy
        #         # with the same galaxy index at a higher snapshot
        #         ind = (np.where((tree['GalaxyIndex'] == this_galidx) &
        #                         (tree['SnapNum'] > this_snapnum)))[0]
        #         msg = "desc == -1 but real descendant = {0}\n".format(ind)
        #         if len(ind) != 0:
        #             print("tree['GalaxyIndex'][{0}] = {1} at snapshot = {2} "
        #                   "should be a descendant for ii = {3} with idx = {4} "
        #                   "at snapshot = {5}".format(
        #                     ind, tree['GalaxyIndex'][ind],
        #                     tree['SnapNum'][ind], ii,
        #                     this_galidx, this_snapnum))
        #         assert len(ind) == 0, msg
        #     else:
        #         assert tree['SnapNum'][desc] > tree['SnapNum'][ii]
        #         assert tree['GalaxyIndex'][desc] == tree['GalaxyIndex'][ii]

        return descs

    def Vpeak(self, tree):
        """
        Calculates the max. of Vmax during the halo history
        """
        vpeak = np.empty(len(tree), np.float32)

        # By pre-filling vpeak with Vmax, I don't have to
        # worry about cases where there are no descendants
        # (although the code should cover that case)
        vpeak[:] = tree['Vmax']

        sorted_ind = np.argsort(tree, order=('GalaxyIndex', 'SnapNum'))
        all_vmax = tree['Vmax']
        all_gal_idx = tree['GalaxyIndex']
        vmax = []
        curr_idx = all_gal_idx[sorted_ind[0]]
        for ii, idx in enumerate(all_gal_idx[sorted_ind]):
            if curr_idx != idx:
                vmax = []
                curr_idx = idx

            vmax.append(all_vmax[sorted_ind[ii]])
            vpeak[sorted_ind[ii]] = max(vmax)

        return vpeak

    def totsfr(self, tree):
        """ Calculate the total star formation rate.

        Just sum the disk and bulge star formation rates
        """
        return tree['SfrDisk'] + tree['SfrBulge']
                 
    def totHI(self, tree):
        arr = tree['DiscHI_1']
        for i in range(2,31): arr += tree['DiscHI_'+str(i)]
        return arr

    def totH2(self, tree):
        arr = tree['DiscH2_1']
        for i in range(2,31): arr += tree['DiscH2_'+str(i)]
        return arr
                 
    def PseudoBulgeMass(self, tree):
        DiscMassArr = np.zeros((len(tree),30))
        DiscRadii = np.zeros((len(tree),30))
        for i in range(1,31):
            DiscMassArr[:,i-1] = tree['DiscStars_'+str(i)]
            DiscRadii[:,i-1] = tree['DiscRadii_'+str(i)]
        DiscRadii_norm = (DiscRadii.T / (0.2*tree['DiskScaleRadius'])).T
        DiscRadii_norm[tree['InstabilityBulgeMass']==0,:] = 1
        DiscMassArr[np.where(DiscRadii_norm>=1)] = 0
        return np.sum(DiscMassArr, axis=1)
                 
    def jStarDisc(self, tree, j_bin):
        arr = np.zeros(len(tree))
        DiscMass = tree['StellarMass'] - tree['InstabilityBulgeMass'] - tree['MergerBulgeMass']
        for i in range(1,31): arr += tree['DiscStars_'+str(i)] * j_bin[i-1]
        arr[DiscMass>0] = arr[DiscMass>0]/DiscMass[DiscMass>0]
        return arr
                 
    def jPseudoBulge(self, tree, j_bin):
        DiscMassArr = np.zeros((len(tree),30))
        DiscRadii = np.zeros((len(tree),30))
        DiscAngMom = np.zeros((len(tree),30))
        for i in range(1,31):
            DiscMassArr[:,i-1] = tree['DiscStars_'+str(i)]
            DiscRadii[:,i-1] = tree['DiscRadii_'+str(i)]
        DiscRadii_norm = (DiscRadii.T / (0.2*tree['DiskScaleRadius'])).T
        DiscRadii_norm[tree['InstabilityBulgeMass']==0,:] = 1
        DiscMassArr[np.where(DiscRadii_norm>=1)] = 0
        BulgeJ = np.sum((DiscMassArr * j_bin), axis=1)
        BulgeMass = np.sum(DiscMassArr, axis=1)
        BulgeMass[BulgeJ<=0] = 1. # prevents divide by zero in next line
        Bulge_j = BulgeJ / BulgeMass
        return Bulge_j
                 
    def jGas(self, tree, j_bin):
        arr = np.zeros(len(tree))
        DiscMass = 1.0*tree['ColdGas']
        for i in range(1,31): arr += tree['DiscGas_'+str(i)] * j_bin[i-1]
        arr[DiscMass>0] = arr[DiscMass>0]/DiscMass[DiscMass>0]
        return arr
                
    def jHI(self, tree, j_bin):
        arr = np.zeros(len(tree))
        DiscMass = np.zeros(len(tree))
        for i in range(1,31):
            arr += tree['DiscHI_'+str(i)] * j_bin[i-1]
            DiscMass += tree['DiscHI_'+str(i)]
        arr[DiscMass>0] = arr[DiscMass>0]/DiscMass[DiscMass>0]
        return arr
                 
    def jH2(self, tree, j_bin):
        arr = np.zeros(len(tree))
        DiscMass = np.zeros(len(tree))
        for i in range(1,31):
            arr += tree['DiscH2_'+str(i)] * j_bin[i-1]
            DiscMass += tree['DiscH2_'+str(i)]
        arr[DiscMass>0] = arr[DiscMass>0]/DiscMass[DiscMass>0]
        return arr
                
    def RadiusHI(self, tree, h):
        DiscRadii = np.zeros((len(tree),30))
        SigmaHI = np.zeros((len(tree),30))
        for i in range(1,31):
            DiscRadii[:,i-1] = np.sqrt((tree['DiscRadii_'+str(i)]**2+tree['DiscRadii_'+str(i-1)]**2)/2)
            SigmaHI[:,i-1] = tree['DiscHI_'+str(i)]/(np.pi*(tree['DiscRadii_'+str(i)]**2-tree['DiscRadii_'+str(i-1)]**2))*1e-2*h
        (row, col) = np.where(SigmaHI>1.0)
        filt = np.append(np.diff(row)>0, True)
        row, col = row[filt], col[filt]
        arr = np.zeros(len(tree))
        arr[row] = DiscRadii[row,col]
        row, col = row[col<29], col[col<29]
        arr[row] = arr[row] + (0-np.log10(SigmaHI[row,col]))/np.log10(SigmaHI[row,col+1]/SigmaHI[row,col]) * (DiscRadii[row,col+1]-DiscRadii[row,col])
        return arr


    def RadiusTrans(self, tree):
        ratio = np.zeros((len(tree),30))
        DiscRadii = np.zeros((len(tree),30))
        for i in range(1,31):
            HI = tree['DiscHI_'+str(i)]
            H2 = tree['DiscH2_'+str(i)]
            w = np.where((H2>0)&(HI>0))
            ratio[w,i-1] = HI[w]/H2[w]
            DiscRadii[:,i-1] = np.sqrt((tree['DiscRadii_'+str(i)]**2+tree['DiscRadii_'+str(i-1)]**2)/2)
        (row, col) = np.where(ratio>1.0)
        ind = np.searchsorted(row, np.unique(row))
        row, col = row[ind], col[ind]
        arr = np.zeros(len(tree))
        arr[row] = DiscRadii[row,col]
        row, col = row[col>0], col[col>0]
        arr[row] = arr[row] - (np.log10(ratio[row,col])-0)/np.log10(ratio[row,col]/ratio[row,col-1]) * (DiscRadii[row,col]-DiscRadii[row,col-1])
        return arr
    
    def r50(self, tree):
        DiscTot = tree['StellarMass'] - tree['InstabilityBulgeMass'] - tree['MergerBulgeMass']
        DiscArr = np.zeros((len(tree),30))
        DiscRadii = np.zeros((len(tree),30))
        for i in range(1,31):
            DiscArr[:,i-1] = tree['DiscStars_'+str(i)]
            DiscRadii[:,i-1] = tree['DiscRadii_'+str(i)]
        ratio = np.cumsum(DiscArr, axis=1)
        ratio[DiscTot>0] = (ratio[DiscTot>0].T/DiscTot[DiscTot>0]).T # Not actually "ratio" until this line
        (row, col) = np.where(ratio>=0.5)
        ind = np.searchsorted(row, np.unique(row))
        row, col = row[ind], col[ind]
        arr = np.zeros(len(tree))
        arr[row] = DiscRadii[row,col]
        row, col = row[col>0], col[col>0]
        arr[row] = arr[row] - (ratio[row,col]-0.5)/(ratio[row,col]-ratio[row,col-1]) * (DiscRadii[row,col]-DiscRadii[row,col-1])
        return arr
                 
    def r90(self, tree):
        DiscTot = tree['StellarMass'] - tree['InstabilityBulgeMass'] - tree['MergerBulgeMass']
        DiscArr = np.zeros((len(tree),30))
        DiscRadii = np.zeros((len(tree),30))
        for i in range(1,31):
            DiscArr[:,i-1] = tree['DiscStars_'+str(i)]
            DiscRadii[:,i-1] = tree['DiscRadii_'+str(i)]
        ratio = np.cumsum(DiscArr, axis=1)
        ratio[DiscTot>0] = (ratio[DiscTot>0].T/DiscTot[DiscTot>0]).T # Not actually "ratio" until this line
        (row, col) = np.where(ratio>=0.9)
        ind = np.searchsorted(row, np.unique(row))
        row, col = row[ind], col[ind]
        arr = np.zeros(len(tree))
        arr[row] = DiscRadii[row,col]
        row, col = row[col>0], col[col>0]
        arr[row] = arr[row] - (ratio[row,col]-0.9)/(ratio[row,col]-ratio[row,col-1]) * (DiscRadii[row,col]-DiscRadii[row,col-1])
        return arr

    def rSFR(self, tree):
        DiscTot = tree['SfrDisk']
        DiscArr = np.zeros((len(tree),30))
        DiscRadii = np.zeros((len(tree),30))
        for i in range(1,31):
            DiscArr[:,i-1] = tree['DiscSFR_'+str(i)]
            DiscRadii[:,i-1] = tree['DiscRadii_'+str(i)]
        ratio = np.cumsum(DiscArr, axis=1)
        ratio[DiscTot>0] = (ratio[DiscTot>0].T/DiscTot[DiscTot>0]).T # Not actually "ratio" until this line
        (row, col) = np.where(ratio>=0.5)
        ind = np.searchsorted(row, np.unique(row))
        row, col = row[ind], col[ind]
        arr = np.zeros(len(tree))
        arr[row] = DiscRadii[row,col]
        row, col = row[col>0], col[col>0]
        arr[row] = arr[row] - (ratio[row,col]-0.5)/(ratio[row,col]-ratio[row,col-1]) * (DiscRadii[row,col]-DiscRadii[row,col-1])
        return arr
                 
    def StellarDiscMass(self,tree):
        return tree['StellarMass'] - tree['InstabilityBulgeMass'] - tree['MergerBulgeMass']
    
    def dZStar(self,tree):
        grad = np.zeros(len(tree))
        for g in xrange(len(tree)):
            DiscMass = tree['StellarMass'][g] - tree['InstabilityBulgeMass'][g] - tree['MergerBulgeMass'][g]
            if DiscMass<=0: continue
            val = 0
            rad = np.array([])
            Z = np.array([])
            for i in range(1,31):
                val += tree['DiscStars_'+str(i)][g]/DiscMass
                if val>0.5 and tree['DiscStars_'+str(i)][g]>0 and tree['DiscStarsMetals_'+str(i)][g]>0:
                    rad = np.append(rad, np.sqrt((tree['DiscRadii_'+str(i)][g]**2+tree['DiscRadii_'+str(i-1)][g]**2)/2)*1e3)
                    Z = np.append(Z, np.log10(tree['DiscStarsMetals_'+str(i)][g]/tree['DiscStars_'+str(i)][g]))
                if val>0.9:
                    break
            if len(rad)>2:
                p = np.polyfit(rad, Z, 1)
                grad[g] = p[0]
        return grad

    def dZGas(self,tree):
        grad = np.zeros(len(tree))
        for g in xrange(len(tree)):
            DiscMass = tree['StellarMass'][g] - tree['InstabilityBulgeMass'][g] - tree['MergerBulgeMass'][g]
            if tree['ColdGas'][g]<=0 or DiscMass<=0: continue
            val = 0
            rad = np.array([])
            Z = np.array([])
            for i in range(1,31):
                val += tree['DiscStars_'+str(i)][g]/DiscMass
                if val>0.5 and tree['DiscGas_'+str(i)][g]>0 and tree['DiscGasMetals_'+str(i)][g]>0:
                    rad = np.append(rad, (tree['DiscRadii_'+str(i)][g]+tree['DiscRadii_'+str(i-1)][g])/2*1e3)
                    Z = np.append(Z, np.log10(tree['DiscGasMetals_'+str(i)][g]/tree['DiscGas_'+str(i)][g]))
                if val>0.9:
                    break
            if len(rad)>2:
                p = np.polyfit(rad, Z, 1)
                grad[g] = p[0]
        return grad
    
    def MetalsStellarDiscMass(self, tree):
        return tree['MetalsStellarMass'] - tree['MetalsInstabilityBulgeMass'] - tree['MetalsMergerBulgeMass']
    
    def MetalsPseudoBulge(self, tree):
        DiscMassArr = np.zeros((len(tree),30))
        DiscRadii = np.zeros((len(tree),30))
        for i in range(1,31):
            DiscMassArr[:,i-1] = tree['DiscStarsMetals_'+str(i)]
            DiscRadii[:,i-1] = tree['DiscRadii_'+str(i)]
        DiscRadii_norm = (DiscRadii.T / (0.2*tree['DiskScaleRadius'])).T
        DiscRadii_norm[tree['InstabilityBulgeMass']==0,:] = 1
        DiscMassArr[np.where(DiscRadii_norm>=1)] = 0
        return np.sum(DiscMassArr, axis=1)

    def map_dt(self, tree):
        """Convert SAGE dT values to Gyrs"""
        return tree['dT'] * 1e-3

    def map_tree_files_to_cores(self, group_strings):
        """
        Splits up the input tree files across cores (for MPI jobs)
        Otherwise, simply returns the input array of `group_strings`
        (group_strings are the core that SAGE processed the corresponding
        input tree file on)

        
        Returns: numpy array of file numbers that this core needs to process

        """
        if self.MPI is None:
            return  np.array(group_strings, dtype=np.int64)
        
        comm = self.MPI.COMM_WORLD
        rank = comm.rank
        ncores = comm.size

        # Easiest way to split is simply to divide the files over the cores
        nfiles = len(group_strings)
        if ncores > nfiles:
            msg = "Error: There are only {0} input files that need to be "\
                "converted but there are {1} parallel tasks. Please use {0} "\
                "tasks at the most(`mpirun -np {0} taoconvert ...`)"\
                .format(nfiles, ncores)
            raise ValueError(msg)
            
        
        nfiles_per_core = nfiles // ncores
        rem = nfiles % ncores
        nfiles_assigned=0
        for icore in xrange(ncores):
            nfiles_this_core = nfiles_per_core
            if rem > 0:
                nfiles_this_core += 1
                rem -=1

            if icore == rank:
                group_nums_this_core = np.arange(nfiles_assigned,
                                                 nfiles_assigned + nfiles_this_core,
                                                 step=1,
                                                 dtype=np.int64)

            # Once icore == rank has been triggered, the following line
            # does not have any impact on the return value. However,
            # this line serves as a check that the logic is correct
            nfiles_assigned += nfiles_this_core

        assert nfiles == nfiles_assigned
        assert rem == 0

        return group_nums_this_core

    
    def iterate_trees(self):
        """Iterate over SAGE trees."""

        file_order = ['ObjectType',
                      'GalaxyIndex',
                      'HaloIndex',
                      'SimulationHaloIndex',
                      'TreeIndex',
                      'SnapNum',
                      'CentralGalaxyIndex',
                      'CentralMvir',
                      'mergeType',
                      'mergeIntoID',
                      'mergeIntoSnapNum',
                      'dT',
                      'Pos_x', 'Pos_y', 'Pos_z',
                      'Vel_x', 'Vel_y', 'Vel_z',
                      'Spin_x', 'Spin_y', 'Spin_z',
                      'Len',
                      'LenMax',
                      'Mvir',
                      'Rvir',
                      'Vvir',
                      'Vmax',
                      'VelDisp',
                      'DiscRadii_0', 'DiscRadii_1', 'DiscRadii_2', 'DiscRadii_3', 'DiscRadii_4', 'DiscRadii_5', 'DiscRadii_6', 'DiscRadii_7', 'DiscRadii_8', 'DiscRadii_9', 'DiscRadii_10', 'DiscRadii_11', 'DiscRadii_12', 'DiscRadii_13', 'DiscRadii_14', 'DiscRadii_15', 'DiscRadii_16', 'DiscRadii_17', 'DiscRadii_18', 'DiscRadii_19', 'DiscRadii_20', 'DiscRadii_21', 'DiscRadii_22', 'DiscRadii_23', 'DiscRadii_24', 'DiscRadii_25', 'DiscRadii_26', 'DiscRadii_27', 'DiscRadii_28', 'DiscRadii_29', 'DiscRadii_30',
                      'ColdGas',
                      'StellarMass',
                      'MergerBulgeMass',
                      'InstabilityBulgeMass',
                      'HotGas',
                      'EjectedMass',
                      'BlackHoleMass',
                      'ICS',
                      'DiscGas_1', 'DiscGas_2', 'DiscGas_3', 'DiscGas_4', 'DiscGas_5', 'DiscGas_6', 'DiscGas_7', 'DiscGas_8', 'DiscGas_9', 'DiscGas_10', 'DiscGas_11', 'DiscGas_12', 'DiscGas_13', 'DiscGas_14', 'DiscGas_15', 'DiscGas_16', 'DiscGas_17', 'DiscGas_18', 'DiscGas_19', 'DiscGas_20', 'DiscGas_21', 'DiscGas_22', 'DiscGas_23', 'DiscGas_24', 'DiscGas_25', 'DiscGas_26', 'DiscGas_27', 'DiscGas_28', 'DiscGas_29', 'DiscGas_30',
                      'DiscStars_1', 'DiscStars_2', 'DiscStars_3', 'DiscStars_4', 'DiscStars_5', 'DiscStars_6', 'DiscStars_7', 'DiscStars_8', 'DiscStars_9', 'DiscStars_10', 'DiscStars_11', 'DiscStars_12', 'DiscStars_13', 'DiscStars_14', 'DiscStars_15', 'DiscStars_16', 'DiscStars_17', 'DiscStars_18', 'DiscStars_19', 'DiscStars_20', 'DiscStars_21', 'DiscStars_22', 'DiscStars_23', 'DiscStars_24', 'DiscStars_25', 'DiscStars_26', 'DiscStars_27', 'DiscStars_28', 'DiscStars_29', 'DiscStars_30',
                      'SpinStars_x', 'SpinStars_y', 'SpinStars_z',
                      'SpinGas_x', 'SpinGas_y', 'SpinGas_z',
                      'SpinInstabilityBulge_x', 'SpinInstabilityBulge_y', 'SpinInstabilityBulge_z',
                      'SpinMergerBulge_x', 'SpinMergerBulge_y', 'SpinMergerBulge_z',
                      'StarsInSitu',
                      'StarsInstability',
                      'StarsMergeBurst',
                      'DiscHI_1', 'DiscHI_2', 'DiscHI_3', 'DiscHI_4', 'DiscHI_5', 'DiscHI_6', 'DiscHI_7', 'DiscHI_8', 'DiscHI_9', 'DiscHI_10', 'DiscHI_11', 'DiscHI_12', 'DiscHI_13', 'DiscHI_14', 'DiscHI_15', 'DiscHI_16', 'DiscHI_17', 'DiscHI_18', 'DiscHI_19', 'DiscHI_20', 'DiscHI_21', 'DiscHI_22', 'DiscHI_23', 'DiscHI_24', 'DiscHI_25', 'DiscHI_26', 'DiscHI_27', 'DiscHI_28', 'DiscHI_29', 'DiscHI_30',
                      'DiscH2_1', 'DiscH2_2', 'DiscH2_3', 'DiscH2_4', 'DiscH2_5', 'DiscH2_6', 'DiscH2_7', 'DiscH2_8', 'DiscH2_9', 'DiscH2_10', 'DiscH2_11', 'DiscH2_12', 'DiscH2_13', 'DiscH2_14', 'DiscH2_15', 'DiscH2_16', 'DiscH2_17', 'DiscH2_18', 'DiscH2_19', 'DiscH2_20', 'DiscH2_21', 'DiscH2_22', 'DiscH2_23', 'DiscH2_24', 'DiscH2_25', 'DiscH2_26', 'DiscH2_27', 'DiscH2_28', 'DiscH2_29', 'DiscH2_30',
                      'DiscSFR_1', 'DiscSFR_2', 'DiscSFR_3', 'DiscSFR_4', 'DiscSFR_5', 'DiscSFR_6', 'DiscSFR_7', 'DiscSFR_8', 'DiscSFR_9', 'DiscSFR_10', 'DiscSFR_11', 'DiscSFR_12', 'DiscSFR_13', 'DiscSFR_14', 'DiscSFR_15', 'DiscSFR_16', 'DiscSFR_17', 'DiscSFR_18', 'DiscSFR_19', 'DiscSFR_20', 'DiscSFR_21', 'DiscSFR_22', 'DiscSFR_23', 'DiscSFR_24', 'DiscSFR_25', 'DiscSFR_26', 'DiscSFR_27', 'DiscSFR_28', 'DiscSFR_29', 'DiscSFR_30',
                      'AccretedGasMass',
                      'EjectedSNGasMass',
                      'EjectedQuasarGasMass',
                      'TotInstabEvents',
                      'TotInstabEventsGas',
                      'TotInstabEventsStar',
                      'TotInstabAnnuliGas',
                      'TotInstabAnnuliStar',
                      'FirstUnstableAvGas',
                      'FirstUnstableAvStar',
                      'TotSinkGas_1', 'TotSinkGas_2', 'TotSinkGas_3', 'TotSinkGas_4', 'TotSinkGas_5', 'TotSinkGas_6', 'TotSinkGas_7', 'TotSinkGas_8', 'TotSinkGas_9', 'TotSinkGas_10', 'TotSinkGas_11', 'TotSinkGas_12', 'TotSinkGas_13', 'TotSinkGas_14', 'TotSinkGas_15', 'TotSinkGas_16', 'TotSinkGas_17', 'TotSinkGas_18', 'TotSinkGas_19', 'TotSinkGas_20', 'TotSinkGas_21', 'TotSinkGas_22', 'TotSinkGas_23', 'TotSinkGas_24', 'TotSinkGas_25', 'TotSinkGas_26', 'TotSinkGas_27', 'TotSinkGas_28', 'TotSinkGas_29', 'TotSinkGas_30',
                      'TotSinkStar_1', 'TotSinkStar_2', 'TotSinkStar_3', 'TotSinkStar_4', 'TotSinkStar_5', 'TotSinkStar_6', 'TotSinkStar_7', 'TotSinkStar_8', 'TotSinkStar_9', 'TotSinkStar_10', 'TotSinkStar_11', 'TotSinkStar_12', 'TotSinkStar_13', 'TotSinkStar_14', 'TotSinkStar_15', 'TotSinkStar_16', 'TotSinkStar_17', 'TotSinkStar_18', 'TotSinkStar_19', 'TotSinkStar_20', 'TotSinkStar_21', 'TotSinkStar_22', 'TotSinkStar_23', 'TotSinkStar_24', 'TotSinkStar_25', 'TotSinkStar_26', 'TotSinkStar_27', 'TotSinkStar_28', 'TotSinkStar_29', 'TotSinkStar_30',
                      'MetalsColdGas',
                      'MetalsStellarMass',
                      'MetalsMergerBulgeMass',
                      'MetalsInstabilityBulgeMass',
                      'MetalsHotGas',
                      'MetalsEjectedMass',
                      'MetalsICS',
                      'DiscGasMetals_1', 'DiscGasMetals_2', 'DiscGasMetals_3', 'DiscGasMetals_4', 'DiscGasMetals_5', 'DiscGasMetals_6', 'DiscGasMetals_7', 'DiscGasMetals_8', 'DiscGasMetals_9', 'DiscGasMetals_10', 'DiscGasMetals_11', 'DiscGasMetals_12', 'DiscGasMetals_13', 'DiscGasMetals_14', 'DiscGasMetals_15', 'DiscGasMetals_16', 'DiscGasMetals_17', 'DiscGasMetals_18', 'DiscGasMetals_19', 'DiscGasMetals_20', 'DiscGasMetals_21', 'DiscGasMetals_22', 'DiscGasMetals_23', 'DiscGasMetals_24', 'DiscGasMetals_25', 'DiscGasMetals_26', 'DiscGasMetals_27', 'DiscGasMetals_28', 'DiscGasMetals_29', 'DiscGasMetals_30',
                      'DiscStarsMetals_1', 'DiscStarsMetals_2', 'DiscStarsMetals_3', 'DiscStarsMetals_4', 'DiscStarsMetals_5', 'DiscStarsMetals_6', 'DiscStarsMetals_7', 'DiscStarsMetals_8', 'DiscStarsMetals_9', 'DiscStarsMetals_10', 'DiscStarsMetals_11', 'DiscStarsMetals_12', 'DiscStarsMetals_13', 'DiscStarsMetals_14', 'DiscStarsMetals_15', 'DiscStarsMetals_16', 'DiscStarsMetals_17', 'DiscStarsMetals_18', 'DiscStarsMetals_19', 'DiscStarsMetals_20', 'DiscStarsMetals_21', 'DiscStarsMetals_22', 'DiscStarsMetals_23', 'DiscStarsMetals_24', 'DiscStarsMetals_25', 'DiscStarsMetals_26', 'DiscStarsMetals_27', 'DiscStarsMetals_28', 'DiscStarsMetals_29', 'DiscStarsMetals_30',
                      'SfrDisk',
                      'SfrBulge',
                      'SfrDiskZ',
                      'SfrBulgeZ',
                      'DiskScaleRadius',
                      'BulgeRadius',
                      'Cooling',
                      'Heating',
                      'TimeofLastMajorMerger',
                      'OutflowRate',
                      'infallMvir',
                      'infallVvir',
                      'infallVmax'
                      ]

        ordered_dtype = []
        for k in file_order:
            field_dict = self.src_fields_dict[k]
            ordered_dtype.append((k, field_dict['type']))

        j_bin, h = self.get_jbins()
    
        computed_fields = {'TotSfr': self.totsfr, 'Vpeak': self.Vpeak, 'HImass': self.totHI, 'H2mass': self.totH2, 'PseudoBulgeMass': self.PseudoBulgeMass, 'jStarDisc': self.jStarDisc, 'jPseudoBulge': self.jPseudoBulge, 'jGas': self.jGas, 'jHI': self.jHI, 'jH2': self.jH2, 'RadiusHI': self.RadiusHI, 'RadiusTrans': self.RadiusTrans, 'r50': self.r50, 'r90': self.r90, 'rSFR': self.rSFR, 'StellarDiscMass': self.StellarDiscMass, 'dZStar': self.dZStar, 'dZGas': self.dZGas, 'MetalsStellarDiscMass': self.MetalsStellarDiscMass, 'MetalsPseudoBulge': self.MetalsPseudoBulge}
        computed_field_list = []
        for f in computed_fields:
            if f not in field_dict.keys():
                assert "Computed field = {0} must still be defined "\
                    "in the module level field_dict".format(f)

            computed_field_list.append((f, field_dict['type']))

        # print("ordered_dtype = {0}".format(ordered_dtype))
        from_file_dtype = np.dtype(ordered_dtype, align=True)
        print("from file type = {0}".format(from_file_dtype))
        print("sizeof(file_dtype) = {0}".format(from_file_dtype.itemsize))
        assert from_file_dtype.itemsize == 1544, "Size of datatypes do not match"
        ordered_dtype.extend(computed_field_list)
        src_type = np.dtype(ordered_dtype)
        # print("src_type = {0}".format(src_type))

        entries = [e for e in os.listdir(self.args.trees_dir)
                   if os.path.isfile(os.path.join(self.args.trees_dir, e))]
        entries = [e for e in entries if e.startswith('model_z')]
        redshift_strings = list(set([re.match(r'model_z(\d+\.?\d*)_\d+', e).group(1)
                                     for e in entries]))
        group_strings = list(set([re.match(r'model_z\d+\.?\d*_(\d+)', e).group(1)
                                  for e in entries]))

        group_strings.sort(lambda x, y: -1 if int(x) < int(y) else 1)
        redshift_strings.sort(lambda x, y: 1 if float(x) < float(y) else -1)

        totntrees = 0L
        for group in group_strings:
            # redshift array is sorted -> pick the last redshift
            redshift = redshift_strings[-1]
            fn = 'model_z{0}_{1}'.format(redshift, group)
            with open(os.path.join(self.args.trees_dir, fn), 'rb') as f:
                n_trees = np.fromfile(f, np.uint32, 1)[0]
                totntrees += n_trees

        # If this is an MPI job, divide up the tasks
        group_nums_this_core = self.map_tree_files_to_cores(group_strings)
        root_process = self.MPI is None or \
            (self.MPI is not None and self.MPI.COMM_WORLD.rank == 0)

        for group in group_nums_this_core:
            files = []
            for redshift in redshift_strings:
                fn = 'model_z{0}_{1}'.format(redshift, group)
                files.append(open(os.path.join(self.args.trees_dir, fn), 'rb'))

            n_trees = [np.fromfile(f, np.uint32, 1)[0] for f in files][0]
            n_gals = [np.fromfile(f, np.uint32, 1)[0] for f in files]
            chunk_sizes = [np.fromfile(f, np.uint32, n_trees) for f in files]
            tree_sizes = sum(chunk_sizes)

            pbar = lambda x : tqdm(x) if root_process else x 
            for ii in pbar(xrange(n_trees)):
                tree_size = tree_sizes[ii]
                tree = np.empty(tree_size, dtype=src_type)
                offs = 0
                for jj in xrange(len(chunk_sizes)):
                    chunk_size = chunk_sizes[jj][ii]
                    data = np.fromfile(files[jj], from_file_dtype, chunk_size)
                    tree[offs:offs + chunk_size] = data
                    offs += chunk_size

                for fieldname, conversion_function in computed_fields.items():
                    if fieldname in ['jStarDisc', 'jPseudoBulge', 'jGas', 'jHI', 'jH2']:
                        tree[fieldname] = conversion_function(tree, j_bin)
                    elif fieldname in ['RadiusHI']:
                        tree[fieldname] = conversion_function(tree, h)
                    else:
                        tree[fieldname] = conversion_function(tree)

                # Reset the negative values for TimeofLastMajorMerger and
                # TimeofLastMinorMerger.
                for f in ['TimeofLastMajorMerger']:
                    timeofmerger = tree[f]
                    ind = (np.where(timeofmerger < 0.0))[0]
                    tree[f][ind] = -1.0
                    
                # Check any fields with negative values that shouldn't have them
                check_fields = [
                                    'Mvir',
                                    'CentralMvir',
                                    'Rvir',
                                    'Vvir',
                                    'Vmax',
                                    'VelDisp',
                                    'StellarMass',
                                    'MergerBulgeMass',
                                    'InstabilityBulgeMass',
                                    'HotGas',
                                    'EjectedMass',
                                    'BlackHoleMass',
                                    'ICS',
                                    'MetalsStellarMass',
                                    'MetalsMergerBulgeMass',
                                    'MetalsInstabilityBulgeMass',
                                    'MetalsHotGas',
                                    'MetalsEjectedMass',
                                    'MetalsICS',
                                    'MetalsStellarDiscMass', 'MetalsPseudoBulge',
                                    'Cooling',
                                    'Heating',
                                    'OutflowRate',
                                    'infallMvir',
                                    'infallVvir',
                                    'infallVmax',
                                    'TotSfr',
                                    'Vpeak',
                                    'HImass',
                                    'H2mass',
                                    'StellarDiscMass', 'PseudoBulgeMass',
                                    'jStarDisc', 'jPseudoBulge', 'jGas', 'jHI', 'jH2',
                                    'RadiusHI', 'RadiusTrans', 'r50', 'r90', 'rSFR',
                                    'ColdGas', 'MetalsColdGas', 'DiskScaleRadius'
                                ]
                for field in check_fields:
                    filt = (tree[field]<0) + (True-np.isfinite(tree[field]))
                    tree[field][filt] = 0.0

                assert min(tree['TimeofLastMajorMerger']) >= -1.0, \
                    "TimeofLastMajorMerger should contain -1.0 to indicate "\
                    "no known last major merger"
#                assert min(tree['TimeofLastMinorMerger']) >= -1.0, \
#                    "TimeofLastMinorMerger should contain -1.0 to indicate "\
#                    "no known last minor merger"



		# dT for first snapshot might be a problem
		ind = (np.where(tree['dT'] < 0.0))[0]
		if len(ind)>0:
                 tree['dT'][ind] = 8.317
 		assert min(tree['dT']) > 0, "Time between snapshots should be positive"

                # First validate ID's.
                for f in ['ObjectType', 'GalaxyIndex', 'CentralGalaxyIndex']:
                    if min(tree[f]) < 0:
                        print("ERROR; min(tree[{0}]) = {1} should be non-zero"
                              .format(f, min(tree[f])))
                        
                # Validate central galaxy index (unique id, generated by sage)
                ind = (np.where(tree['ObjectType'] == 0))[0]
                if not bool(np.all(tree['GalaxyIndex'][ind] ==
                                   tree['CentralGalaxyIndex'][ind])):
                    print("tree[GalaxyIndex][ind] = {0}".format(tree['GalaxyIndex'][ind]))
                    print("tree[CentralGalaxyIndex][ind] = {0}".format(tree['CentralGalaxyIndex'][ind]))

                assert bool(np.all(tree['GalaxyIndex'][ind] ==
                              tree['CentralGalaxyIndex'][ind])), \
                    "Central Galaxy Index must equal Galaxy Index for centrals"
                              
                
                yield tree

            for file in files:
                file.close()
