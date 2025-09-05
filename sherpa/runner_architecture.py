import sherpa
import sherpa.schedulers
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--local', help='Run locally', action='store_true',
                    default=False)
parser.add_argument('--max_concurrent',
                    help='Number of concurrent processes',
                    type=int, default=1)
parser.add_argument('-P',
                    help="Specifies the project to which this  job  is  assigned.",
                    default='arcus_gpu.p')
parser.add_argument('-q',
                    help='Defines a list of cluster queues or queue instances which may be used to execute this job.',
                    default='arcus.q')
parser.add_argument('-l', help='the given resource list.',
                    default="hostname=\'(arcus-1|arcus-2|arcus-3|arcus-4|arcus-5|arcus-6|arcus-7|arcus-8|arcus-9)\'")
parser.add_argument('--env', help='Your environment path.',
                    default='/home/lhertel/profiles/python3env.profile',
                    type=str)
FLAGS = parser.parse_args()


# Define Hyperparameter ranges
# parameters = [sherpa.Choice('pooling', ['average', 'max', 'fullyconv']),
#               sherpa.Choice('activation', ['prelu', 'relu', 'elu']),
#               sherpa.Choice('skipconnection', ['yes', 'no']),
#               sherpa.Choice('kernel_init', ['he_normal', 'he_uniform', 'glorot_uniform', 'glorot_normal']),
#               sherpa.Ordinal('num_blocks', [1,2,3]),
#               sherpa.Ordinal('num_top_blocks', [1,2,3]),
#               sherpa.Ordinal('filter_number', [32, 64, 128]),
#               sherpa.Ordinal('input_scaling', [0.1, 0.5, 1., 5., 10.])]
parameters = [sherpa.Choice('pooling', ['average', 'max']),
              sherpa.Choice('activation', ['prelu', 'relu', 'elu']),
              sherpa.Choice('skipconnection', ['yes', 'no']),
              sherpa.Choice('kernel_init', ['he_normal', 'he_uniform', 'glorot_uniform']),
              sherpa.Ordinal('num_blocks', [1,2]),
              sherpa.Ordinal('num_top_blocks', [1,2]),
              sherpa.Ordinal('filter_number', [32, 64]),
              sherpa.Ordinal('input_scaling', [0.5, 1., 5.])]

algorithm = sherpa.algorithms.LocalSearch(seed_configuration={'pooling': 'max',
                                                              'activation': 'relu',
                                                              'skipconnection': 'no',
                                                              'kernel_init': 'glorot_uniform',
                                                              'num_blocks': 1,
                                                              'num_top_blocks': 1,
                                                              'filter_number': 32,
                                                              'input_scaling': 1.})

# The scheduler
if not FLAGS.local:
    env = FLAGS.env
    opt = '-N novaArchitecture -P {} -q {} -l {} -l gpu=1'.format(FLAGS.P, FLAGS.q, FLAGS.l)
    scheduler = sherpa.schedulers.SGEScheduler(environment=env, submit_options=opt)
else:
    scheduler = sherpa.schedulers.LocalScheduler()

# Running it all
sherpa.optimize(algorithm=algorithm,
                scheduler=scheduler,
                parameters=parameters,
                lower_is_better=True,
                filename="train_sherpa.py",
                max_concurrent=FLAGS.max_concurrent,
                output_dir='./output_architecture_search_refreshed_2')