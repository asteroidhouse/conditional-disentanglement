"""Plotting utilities.
"""
import sklearn.metrics
import matplotlib.pyplot as plt


def plot_confusion_matrix(preds, gt_labels, title=None, fname=None):
  confusion_mat = sklearn.metrics.confusion_matrix(
      preds, gt_labels, normalize='all'
  )

  yticks = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']
  xticks = ['(0,0)', '(0,1)', '(1,0)', '(1,1)']

  fig = plt.figure()
  im = plt.imshow(confusion_mat)

  plt.xticks([0, 1, 2, 3], yticks, fontsize=18)
  plt.yticks([0, 1, 2, 3], xticks, fontsize=18)

  # Loop over data dimensions and create text annotations.
  for i in range(len(xticks)):
    for j in range(len(yticks)):
      text = plt.text(j, i, round(confusion_mat[i, j], 2),
                      ha='center', va='center', color='g', fontsize=16)

  if title:
    plt.title(title, fontsize=20)

  plt.tight_layout()
  plt.savefig(fname, bbox_inches='tight', pad_inches=0)
  plt.close(fig)
