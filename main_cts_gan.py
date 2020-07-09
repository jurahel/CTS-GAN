"""Time-series Generative Adversarial Networks (TimeGAN) Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Updated for the new CTS-GAN by Justin Hellermann

-----------------------------

main_timegan.py

(1) Import data
(2) Generate synthetic data
(3) Evaluate the performances in three ways
  - Visualization (t-SNE, PCA)
  - Discriminative score
  - Predictive score
"""

## Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# 1. TimeGAN model
from cts_gan import timegan
# 2. Data loading
from data_loading import real_data_loading#, sine_data_generation
# 3. Metrics
#from metrics.discriminative_metrics import discriminative_score_metrics
from metrics.predictive_metrics_cts_gan import predictive_score_metrics
from metrics.visualization_metrics_cts_gan import visualization


def main(args):
    """Main function for timeGAN experiments.
  
  Args:
    - data_name: sine, stock, or energy
    - seq_len: sequence length
    - Network parameters (should be optimized for different datasets)
      - module: gru, lstm, or lstmLN
      - hidden_dim: hidden dimensions
      - num_layer: number of layers
      - iteration: number of training iterations
      - batch_size: the number of samples in each batch
    - metric_iteration: number of iterations for metric computation
  
  Returns:
    - ori_data: original data
    - generated_data: generated synthetic data
    - metric_results: discriminative and predictive scores
  """
    ## Data loading
    if args.data_name in ['stock', 'energy']:
        ori_data = real_data_loading(args.seq_len)

    print(args.data_name + ' dataset is ready.')

    ## Synthetic data generation by TimeGAN
    # Set newtork parameters
    parameters = dict()
    parameters['module'] = args.module
    parameters['hidden_dim'] = args.hidden_dim
    parameters['num_layer'] = args.num_layer
    parameters['iterations'] = args.iteration
    parameters['batch_size'] = args.batch_size

    generated_data = timegan(ori_data, parameters)
    np.save('generated_data.npy', generated_data)
    np.save('generated_data_last.npy', generated_data)
    print('Finish Synthetic Data Generation')


    ## Performance metrics
    # Output initialization
    metric_results = dict()
    '''
    # 1. Discriminative Score-
    discriminative_score = list()
    for _ in range(args.metric_iteration):
        temp_disc = discriminative_score_metrics(ori_data[1:], generated_data[:-1])
        discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)
    '''
    # 2. Predictive score
    
    predictive_score = list()
    for tt in range(args.metric_iteration):
        temp_pred = predictive_score_metrics(ori_data[1:], generated_data[:-1])
        predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)
    
    # 3. Visualization (PCA and tSNE)
    visualization(ori_data[:-1], generated_data[1:], 'pca')
    visualization(ori_data[:-1], generated_data[1:], 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)

    return ori_data, generated_data



if __name__ == '__main__':
    # Inputs for the main function
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_name',
        choices=['sine', 'stock', 'energy'],
        default='stock',
        type=str)
    parser.add_argument(
        '--seq_len',
        help='sequence length',
        default=25,
        type=int)
    parser.add_argument(
        '--module',
        choices=['gru', 'lstm', 'lstmLN'],
        default='lstm',
        type=str)
    parser.add_argument(
        '--hidden_dim',
        help='hidden state dimensions (should be optimized)',
        default=25,
        type=int)
    parser.add_argument(
        '--num_layer',
        help='number of layers (should be optimized)',
        default=3,
        type=int)
    parser.add_argument(
        '--iteration',
        help='Training iterations (should be optimized)',
        default=60000,
        type=int)
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch (should be optimized)',
        default=128,
        type=int)
    parser.add_argument(
        '--metric_iteration',
        help='iterations of the metric computation',
        default=10,
        type=int)

    args = parser.parse_args()

    # Calls main function
    ori_data, generated_data = main(args)


