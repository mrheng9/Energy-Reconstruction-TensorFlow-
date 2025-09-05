import os
import argparse
import sherpa
import datetime
from sherpa.schedulers import LocalScheduler,SGEScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--max_concurrent',
                    help='Number of concurrent processes',
                    type=int, default=1)
FLAGS = parser.parse_args()

# Iterate algorithm accepts dictionary containing lists of possible values. 
hp_space = {'lr': [1e-2, 5e-3, 1e-3, 5e-4],
            'batch_size': [32, 64]
            }
parameters = sherpa.Parameter.grid(hp_space)

alg = sherpa.algorithms.GridSearch()
stopping_rule = sherpa.algorithms.MedianStoppingRule(min_iterations=35, min_trials=10)

# Files
f = './nova_train.py --optimizer adam' # Python script to run.
outdir = './output_adam' 
outdir += ''.join(['_' + k for k in hp_space.keys()])
outdir += str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Submit to SGE queue.
P = 'arcus.p'
q = 'arcus-ubuntu.q'
l = "hostname=\'(arcus-5)\'"
env = '/home/lhertel/profiles/python3env.profile'
opt = '-N example -P {} -q {} -l {}'.format(P, q, l)
sched = SGEScheduler(environment=env, submit_options=opt, output_dir=outdir)

# Optimize
rval = sherpa.optimize(parameters=parameters,
                       algorithm=alg,
                       stopping_rule=stopping_rule,
                       output_dir=outdir,
                       lower_is_better=True,
                       filename=f,
                       scheduler=sched,
                       max_concurrent=FLAGS.max_concurrent)
print()
print('Best results:')
print(rval)

