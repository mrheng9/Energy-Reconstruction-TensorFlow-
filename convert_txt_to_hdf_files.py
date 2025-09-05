import os
import nova
import nova.dataprocessing

path = "/baldig/physicsprojects/nova/data/raw/add_flat_text/"
newpath = "/baldig/physicsprojects/nova/data/raw/add_flat_df/"
fnames = [fname for fname in os.listdir(path) if fname.endswith('.txt')]
# ftag = 'fdflatnuesg'
# idx_offset = 0

for f in fnames:
    print("Processing: {}".format(f))
    if f.startswith('fdcellnuesg_'):
        idx = int(f[12:-9])
        ftag = 'fdcellnuesg'
        idx_offset = 1000
    else:
        idx = int(f[13:-9])
        ftag = 'fdcellnuesg2'
        idx_offset = 2000
    df = nova.dataprocessing.get_dataframe(number=idx, dir=path, tag=ftag)
    df.to_hdf(path_or_buf=os.path.join(newpath, str(idx + idx_offset) + '.h5'),
              key='df',
              complevel=9)