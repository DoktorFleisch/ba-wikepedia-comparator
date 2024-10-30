import torch
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def get_weighted_similarity_matrix(similarity_matrix, weight_matrix):
  return torch.stack([row * weight_matrix[index] for (index, row) in enumerate(similarity_matrix)])

def get_weighted_median(similarity_matrix, weight_matrix):
  return torch.mean(torch.tensor([torch.median(torch.sort(x)[0]) for x in get_weighted_similarity_matrix(similarity_matrix, weight_matrix)])*1.0)

def get_weighted_max_average(similarity_matrix, weight_matrix):
  return torch.mean(torch.tensor([(torch.max(x)) for x in get_weighted_similarity_matrix(similarity_matrix, weight_matrix)])*1.0)

def get_weighted_min_average(similarity_matrix, weight_matrix):
  return torch.mean(torch.tensor([(torch.min(x)) for x in get_weighted_similarity_matrix(similarity_matrix, weight_matrix)])*1.0)

def get_max_averaged(similarity_matrix):
  return torch.mean(torch.tensor([torch.max(x) for x in similarity_matrix]))

def get_min_averaged(similarity_matrix):
  return torch.mean(torch.tensor([torch.min(x) for x in similarity_matrix]))

def get_median_averaged(similarity_matrix):
  return torch.mean(torch.tensor([torch.median(torch.sort(x)[0]) for x in similarity_matrix])*1.0)


def plot_CC_Vectors(corr_a, corr_b, A, B, s):
  mean_A = torch.mean(A * 1.0, dim=1)
  mean_B = torch.mean(B * 1.0, dim=1)
  A_meaned = A - mean_A
  B_meaned = B - mean_B
  for i in range(corr_a):
    a_i = corr_a[:, i]
    b_i = corr_b[:, i]
    eta = torch.sum(torch.stack([a_i[k] * A_meaned[:, k] for k in range(len(a_i))], dim=1), dim=1)
    psi = torch.sum(torch.stack([b_i[k] * B_meaned[:, k] for k in range(len(b_i))], dim=1), dim=1)
    plt.figure(figsize=(16, 9))
    plt.plot(eta, psi, marker='o')
    #plt.title(f'Plot for {key}')
    plt.legend('correlation coefficient: {:.2f}'.format(s[i]))
    plt.xlabel('eta')
    plt.ylabel('psi')
    plt.grid(True)

    # Save the plot
    plt.savefig(f"{i}_plot.png")


def plot_similarity_hist(similarity_matrix_simple_first, weight_matrix_first,
                         similarity_matrix_simple_second, weight_matrix_second,
                         similarity_scores_pca_first, similarity_scores_pca_second,
                         first_article, second_article):
  plt.figure(figsize=(16, 9))

  plt.subplot(2, 2, 1)  # 1 row, 2 columns, subplot 1
  flattened_similarity__simple_first = similarity_matrix_simple_first.reshape(-1).numpy()
  plt.hist(get_weighted_similarity_matrix(similarity_matrix_simple_first, weight_matrix_first).reshape(-1).numpy(), bins=50, color='skyblue', edgecolor='black')
  plt.title(first_article + ' to ' + second_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')
  plt.legend(['average: {:.2f}'.format(np.mean(flattened_similarity__simple_first)) + '\n' +
              'median: {:.2f}'.format(np.median(np.sort(flattened_similarity__simple_first))) + '\n' +
              'averaged maximum similarity: {:.2f}'.format(get_max_averaged(similarity_matrix_simple_first)) + '\n' +
              'averaged minimum similarity: {:.2f}'.format(get_min_averaged(similarity_matrix_simple_first)) + '\n' +
              'averaged median similarity: {:.2f}'.format(get_median_averaged(similarity_matrix_simple_first)) + '\n' +

              'averaged median-weighted similarity: {:.2f}'.format(get_weighted_median(similarity_matrix_simple_first, weight_matrix_first)) + '\n' +

              'averaged max-weighted similarity: {:.2f}'.format(get_weighted_max_average(similarity_matrix_simple_first, weight_matrix_first)) + '\n' +

              'averaged min-weighted similarity: {:.2f}'.format(get_weighted_min_average(similarity_matrix_simple_first, weight_matrix_first))], loc='upper right')

  plt.subplot(2, 2, 3)  # 1 row, 2 columns, subplot 2
  flattened_similarity__simple_second = similarity_matrix_simple_second.reshape(-1).numpy()
  plt.hist(get_weighted_similarity_matrix(similarity_matrix_simple_second, weight_matrix_second).reshape(-1).numpy(), bins=50, color='skyblue', edgecolor='black')
  plt.title(second_article + ' to ' + first_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')
  plt.legend(['average: {:.2f}'.format(np.mean(flattened_similarity__simple_second)) + '\n' +
              'median: {:.2f}'.format(np.median(np.sort(flattened_similarity__simple_second))) + '\n' +
              'averaged maximum similarity: {:.2f}'.format(get_max_averaged(similarity_matrix_simple_second)) + '\n' +
              'averaged minimum similarity: {:.2f}'.format(get_min_averaged(similarity_matrix_simple_second)) + '\n' +
              'averaged median similarity: {:.2f}'.format(get_median_averaged(similarity_matrix_simple_second)) + '\n' +

              'averaged median-weighted similarity: {:.2f}'.format(get_weighted_median(similarity_matrix_simple_second, weight_matrix_second)) + '\n' +

              'averaged max-weighted similarity: {:.2f}'.format(get_weighted_max_average(similarity_matrix_simple_second, weight_matrix_second)) + '\n' +

              'averaged min-weighted similarity: {:.2f}'.format(get_weighted_min_average(similarity_matrix_simple_second, weight_matrix_second))], loc='upper right')


  plt.subplot(2, 2, 2)  # 1 row, 2 columns, subplot 1
  plt.plot(similarity_scores_pca_first, color='skyblue')
  plt.title(first_article + ' to ' + second_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')

  plt.subplot(2, 2, 4)   # 1 row, 2 columns, subplot 2
  plt.plot(similarity_scores_pca_second, color='skyblue')
  plt.title(second_article + ' to ' + first_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')

  plt.suptitle('Histogram of Similarity Scores ' + first_article + ' and ' + second_article)

  plt.savefig('plot' + first_article + ' and ' + second_article + '.jpg')

  plt.show()

def plot_similarity_heatmap_simple_with_pca_pairwise(
          similarity_matrix_simple_first,
          similarity_matrix_simple_second,
          similarity_matrix_pca_first,
          similarity_matrix_pca_second,
          first_article, second_article):

  subset_simple_first_A = sorted(set(subtitle[0] for subtitle in similarity_matrix_simple_first.keys()))
  subset_simple_first_B = sorted(set(subtitle[1] for subtitle in similarity_matrix_simple_first.keys()))

  subset_simple_second_A = sorted(set(subtitle[0] for subtitle in similarity_matrix_simple_second.keys()))
  subset_simple_second_B = sorted(set(subtitle[1] for subtitle in similarity_matrix_simple_second.keys()))

  subset_pca_first_A = sorted(set(subtitle[0] for subtitle in similarity_matrix_pca_first.keys()))
  subset_pca_first_B = sorted(set(subtitle[1] for subtitle in similarity_matrix_pca_first.keys()))

  subset_pca_second_A = sorted(set(subtitle[0] for subtitle in similarity_matrix_pca_second.keys()))
  subset_pca_second_B = sorted(set(subtitle[1] for subtitle in similarity_matrix_pca_second.keys()))

  num_subset_simple_first_A = len(subset_simple_first_A)
  num_subset_simple_first_B = len(subset_simple_first_B)
  similarity_matrix_simple_first_np = np.zeros((num_subset_simple_first_A, num_subset_simple_first_B))

  for i, subtitle_A in enumerate(subset_simple_first_A):
    for j, subtitle_B in enumerate(subset_simple_first_B):
      similarity_matrix_simple_first_np[i, j] = similarity_matrix_simple_first.get((subtitle_A, subtitle_B), 0.0)

  num_subset_simple_second_A = len(subset_simple_second_A)
  num_subset_simple_second_B = len(subset_simple_second_B)
  similarity_matrix_simple_second_np = np.zeros((num_subset_simple_second_A, num_subset_simple_second_B))

  for i, subtitle_A in enumerate(subset_simple_second_A):
    for j, subtitle_B in enumerate(subset_simple_second_B):
      similarity_matrix_simple_second_np[i, j] = similarity_matrix_simple_second.get((subtitle_A, subtitle_B), 0.0)

  num_subset_pca_first_A = len(subset_pca_first_A)
  num_subset_pca_first_B = len(subset_pca_first_B)
  similarity_matrix_pca_first_np = np.zeros((num_subset_pca_first_A, num_subset_pca_first_B))

  for i, subtitle_A in enumerate(subset_pca_first_A):
    for j, subtitle_B in enumerate(subset_pca_first_B):
      similarity_matrix_pca_first_np[i, j] = similarity_matrix_pca_first.get((subtitle_A, subtitle_B), 0.0)

  num_subset_pca_second_A = len(subset_pca_second_A)
  num_subset_pca_second_B = len(subset_pca_second_B)
  similarity_matrix_pca_second_np = np.zeros((num_subset_pca_second_A, num_subset_pca_second_B))

  for i, subtitle_A in enumerate(subset_pca_second_A):
    for j, subtitle_B in enumerate(subset_pca_second_B):
      similarity_matrix_pca_second_np[i, j] = similarity_matrix_pca_second.get((subtitle_A, subtitle_B), 0.0)

  # Plotting the clustered heatmap
  plt.figure(figsize=(10, 8))

  fig, axs = plt.subplots(2, 2, figsize=(16, 10))
  sns.heatmap(similarity_matrix_simple_first_np, cmap='Blues', annot=True, xticklabels=subset_simple_first_A, yticklabels=subset_simple_first_B,
              linewidths=.5, ax=axs[0, 0])
  axs[0, 0].set_title('cosine - ', first_article, ' to ', second_article)
  axs[0, 0].set_xlabel('Subsets of ', first_article)
  axs[0, 0].set_ylabel('Subsets of ', second_article)

  # Plot second heatmap (Version B)
  sns.heatmap(similarity_matrix_pca_first_np, cmap='Blues', annot=True, xticklabels=subset_pca_first_A, yticklabels=subset_pca_first_B,
              linewidths=.5, ax=axs[0, 1])
  axs[0, 1].set_title('PCA - ', first_article, ' to ', second_article)
  axs[0, 1].set_xlabel('Subsets of ', first_article)
  axs[0, 1].set_ylabel('Subsets of ', second_article)

  sns.heatmap(similarity_matrix_simple_second_np, cmap='Blues', annot=True, xticklabels=subset_simple_second_A, yticklabels=subset_simple_second_B,
                linewidths=.5, ax=axs[1, 0])
  axs[1, 0].set_title('cosine - ', second_article, ' to ', first_article)
  axs[1, 0].set_xlabel('Subsets of ', second_article)
  axs[1, 0].set_ylabel('Subsets of ', first_article)

  # Plot second heatmap (Version B)
  sns.heatmap(similarity_matrix_pca_second_np, cmap='Blues', annot=True, xticklabels=subset_pca_second_A, yticklabels=subset_pca_second_B,
                linewidths=.5, ax=axs[1, 1])
  axs[1, 1].set_title('PCA - ', second_article, ' to ', first_article)
  axs[1, 1].set_xlabel('Subsets of ', second_article)
  axs[1, 1].set_ylabel('Subsets of ', first_article)
  plt.show()

def plot_similarity_hist_simple_with_pca_pairwise(
                         similarity_matrix_simple_first, weight_matrix_first,
                         similarity_matrix_simple_second, weight_matrix_second,
                         similarity_matrix_pca_first, weight_matrix_pca_first,
                         similarity_matrix_pca_second, weight_matrix_pca_second,
                         first_article, second_article):
  plt.figure(figsize=(16, 9))

  plt.subplot(2, 2, 1)  # 1 row, 2 columns, subplot 1
  flattened_similarity__simple_first = similarity_matrix_simple_first.reshape(-1).numpy()
  plt.hist(get_weighted_similarity_matrix(similarity_matrix_simple_first, weight_matrix_first).reshape(-1).numpy(), bins=50, color='skyblue', edgecolor='black')
  plt.title(first_article + ' to ' + second_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')
  plt.legend(['average: {:.2f}'.format(np.mean(flattened_similarity__simple_first)) + '\n' +
              'median: {:.2f}'.format(np.median(np.sort(flattened_similarity__simple_first))) + '\n' +
              'averaged maximum similarity: {:.2f}'.format(get_max_averaged(similarity_matrix_simple_first)) + '\n' +
              'averaged minimum similarity: {:.2f}'.format(get_min_averaged(similarity_matrix_simple_first)) + '\n' +
              'averaged median similarity: {:.2f}'.format(get_median_averaged(similarity_matrix_simple_first)) + '\n' +

              'averaged median-weighted similarity: {:.2f}'.format(get_weighted_median(similarity_matrix_simple_first, weight_matrix_first)) + '\n' +

              'averaged max-weighted similarity: {:.2f}'.format(get_weighted_max_average(similarity_matrix_simple_first, weight_matrix_first)) + '\n' +

              'averaged min-weighted similarity: {:.2f}'.format(get_weighted_min_average(similarity_matrix_simple_first, weight_matrix_first))], loc='upper right')

  plt.subplot(2, 2, 3)  # 1 row, 2 columns, subplot 2
  flattened_similarity__simple_second = similarity_matrix_simple_second.reshape(-1).numpy()
  plt.hist(get_weighted_similarity_matrix(similarity_matrix_simple_second, weight_matrix_second).reshape(-1).numpy(), bins=50, color='skyblue', edgecolor='black')
  plt.title(second_article + ' to ' + first_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')
  plt.legend(['average: {:.2f}'.format(np.mean(flattened_similarity__simple_second)) + '\n' +
              'median: {:.2f}'.format(np.median(np.sort(flattened_similarity__simple_second))) + '\n' +
              'averaged maximum similarity: {:.2f}'.format(get_max_averaged(similarity_matrix_simple_second)) + '\n' +
              'averaged minimum similarity: {:.2f}'.format(get_min_averaged(similarity_matrix_simple_second)) + '\n' +
              'averaged median similarity: {:.2f}'.format(get_median_averaged(similarity_matrix_simple_second)) + '\n' +

              'averaged median-weighted similarity: {:.2f}'.format(get_weighted_median(similarity_matrix_simple_second, weight_matrix_second)) + '\n' +

              'averaged max-weighted similarity: {:.2f}'.format(get_weighted_max_average(similarity_matrix_simple_second, weight_matrix_second)) + '\n' +

              'averaged min-weighted similarity: {:.2f}'.format(get_weighted_min_average(similarity_matrix_simple_second, weight_matrix_second))], loc='upper right')


  plt.subplot(2, 2, 2)  # 1 row, 2 columns, subplot 1
  flattened_similarity_matrix_pca_first = similarity_matrix_pca_first.reshape(-1).numpy()
  plt.hist(get_weighted_similarity_matrix(similarity_matrix_pca_first, weight_matrix_pca_first).reshape(-1).numpy(), bins=50, color='skyblue', edgecolor='black')
  plt.title(first_article + ' to ' + second_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')
  plt.legend(['average: {:.2f}'.format(np.mean(flattened_similarity_matrix_pca_first)) + '\n' +
              'median: {:.2f}'.format(np.median(np.sort(flattened_similarity_matrix_pca_first))) + '\n' +
              'averaged maximum similarity: {:.2f}'.format(get_max_averaged(similarity_matrix_pca_first)) + '\n' +
              'averaged minimum similarity: {:.2f}'.format(get_min_averaged(similarity_matrix_pca_first)) + '\n' +
              'averaged median similarity: {:.2f}'.format(get_median_averaged(similarity_matrix_pca_first)) + '\n' +

              'averaged median-weighted similarity: {:.2f}'.format(get_weighted_median(similarity_matrix_pca_first, weight_matrix_pca_first)) + '\n' +

              'averaged max-weighted similarity: {:.2f}'.format(get_weighted_max_average(similarity_matrix_pca_first, weight_matrix_pca_first)) + '\n' +

              'averaged min-weighted similarity: {:.2f}'.format(get_weighted_min_average(similarity_matrix_pca_first, weight_matrix_pca_first))], loc='upper right')

  plt.subplot(2, 2, 4)   # 1 row, 2 columns, subplot 2
  flattened_similarity_matrix_pca_second = similarity_matrix_pca_second.reshape(-1).numpy()
  plt.hist(get_weighted_similarity_matrix(similarity_matrix_pca_second, weight_matrix_pca_second).reshape(-1).numpy(), bins=50, color='skyblue', edgecolor='black')
  plt.title(second_article + ' to ' + first_article)
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')
  plt.legend(['average: {:.2f}'.format(np.mean(flattened_similarity_matrix_pca_second)) + '\n' +
              'median: {:.2f}'.format(np.median(np.sort(flattened_similarity_matrix_pca_second))) + '\n' +
              'averaged maximum similarity: {:.2f}'.format(get_max_averaged(similarity_matrix_pca_second)) + '\n' +
              'averaged minimum similarity: {:.2f}'.format(get_min_averaged(similarity_matrix_pca_second)) + '\n' +
              'averaged median similarity: {:.2f}'.format(get_median_averaged(similarity_matrix_pca_second)) + '\n' +

              'averaged median-weighted similarity: {:.2f}'.format(get_weighted_median(similarity_matrix_pca_second, weight_matrix_pca_second)) + '\n' +

              'averaged max-weighted similarity: {:.2f}'.format(get_weighted_max_average(similarity_matrix_pca_second, weight_matrix_pca_second)) + '\n' +

              'averaged min-weighted similarity: {:.2f}'.format(get_weighted_min_average(similarity_matrix_pca_second, weight_matrix_pca_second))], loc='upper right')

  plt.suptitle('Histogram of Similarity Scores ' + first_article + ' and ' + second_article)

  plt.savefig('plot-' + first_article + ' and ' + second_article + '_PCA' + '.jpg')


  plt.show()


def plot_similarity_heatmap(similarity_matrix, x_labels, y_labels, approach):
  plt.figure(figsize=(22, 12))
  sns.heatmap(similarity_matrix.numpy(), annot=True, xticklabels=x_labels, yticklabels=y_labels, cmap='viridis')
  plt.xlabel('first_article')
  plt.ylabel('second_article')
  plt.title('Similarity Heatmap')

  plt.xticks(rotation=45, ha='right', fontsize=10)

  plt.tight_layout()
  # Save the plot as an image file
  plt.savefig("heatmap-" + approach + '.png', bbox_inches='tight')
  plt.show()

def plot_histogram(similarity_vector, approach, title="Similarity Scores Histogram"):
  plt.figure(figsize=(10, 6))
  plt.hist(similarity_vector.numpy(), bins=30, color='skyblue', edgecolor='black')
  plt.xlabel('Similarity Score')
  plt.ylabel('Frequency')
  plt.title(title)
  plt.savefig("hist" + "-" + approach + ".png")
  plt.show()