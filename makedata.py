from nova.dataprocessing import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--type", help="images, sparse_images, vertices, directions or energies", type=str, default='energies')
parser.add_argument("--fr", help="First file to process", type=int, default=0)
parser.add_argument("--to", help="Process until this file", type=int, default=10)
parser.add_argument("--lmdb", action="store_true", default=False)
parser.add_argument("--dir", type=str, default='/baldig/physicsprojects/nova/data/raw/flat_df/')
parser.add_argument("--target_dir", type=str, default='/baldig/physicsprojects/nova/data/flat/')
parser.add_argument("--tag", type=str, default='')
parser.add_argument("--not_hdf", action="store_true")
parser.add_argument("--nan_ok", action="store_true", default=False)

args = parser.parse_args()

raw_dir = args.dir

datasets = {'images': EventImages(),
            'energies': EventEnergies(),
            'vertices': EventVertices3D(),
            'true-vertices': TrueVertices(),
            'electron_images': ElectronImages(),
            'small_electron_images': SmallElectronImages(),
            'electron_energies': ElectronEnergies(),
            'electron_vertices': ElectronVertices(),
            'electron_direction': ElectronDirection(),
            'electron_reco_direction': ElectronRecoDirection(),
            'event_direction': EventDirection(),
            'event_reco_direction': EventRecoDirection()}

if args.lmdb:
    store_lmdb(data_objects=[EventImages(), EventEnergies(), EventVertices()],
               fr_=args.fr, to_= args.to)   
else:
    store_data(data_object=datasets.get(args.type),
               name='{}'.format(args.type),
               fr_=args.fr, to_= args.to,
               raw_dir=raw_dir,
               tag=args.tag,
               hdf=not args.not_hdf,
               target_dir=args.target_dir,
               assert_no_nan=not args.nan_ok)