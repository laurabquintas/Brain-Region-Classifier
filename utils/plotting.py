"""Matplotlib theme configuration and plotting utilities for classifier evaluation."""

from itertools import product
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from numpy import arange, ndarray, newaxis, set_printoptions
from matplotlib.font_manager import FontProperties
from sklearn.metrics import confusion_matrix
from cycler import cycler

# ── Color palette & theme ──────────────────────────────────────────────────────

plt.style.use('mlblabs')

PALETTE = {
    'yellow': '#ECD474', 'pale orange': '#E9AE4E', 'salmon': '#E2A36B',
    'orange': '#F79522', 'dark orange': '#D7725E',
    'acqua': '#64B29E', 'green': '#10A48A', 'olive': '#99C244',
    'pale blue': '#BDDDE0', 'blue2': '#199ED5', 'blue3': '#1DAFE5',
    'dark blue': '#0C70B2',
    'pale pink': '#D077AC', 'lavender': '#E09FD5', 'purple': '#923E97',
    'white': '#FFFFFF', 'light grey': '#D2D3D4', 'grey': '#939598',
}

LINE_COLOR = PALETTE['dark blue']
FILL_COLOR = PALETTE['pale blue']
ACTIVE_COLORS = [
    PALETTE['dark blue'], PALETTE['yellow'], PALETTE['pale orange'],
    PALETTE['acqua'], PALETTE['pale pink'], PALETTE['lavender'],
]

blues = [PALETTE['pale blue'], PALETTE['blue2'], PALETTE['blue3'], PALETTE['dark blue']]
cmap_blues = clrs.LinearSegmentedColormap.from_list("myCMPBlues", blues)

plt.rcParams['axes.prop_cycle'] = cycler('color', ACTIVE_COLORS)
plt.rcParams['text.color'] = LINE_COLOR
plt.rcParams['patch.edgecolor'] = LINE_COLOR
plt.rcParams['patch.facecolor'] = FILL_COLOR
plt.rcParams['axes.facecolor'] = PALETTE['white']
plt.rcParams['axes.edgecolor'] = PALETTE['grey']
plt.rcParams['axes.labelcolor'] = PALETTE['grey']
plt.rcParams['xtick.color'] = PALETTE['grey']
plt.rcParams['ytick.color'] = PALETTE['grey']
plt.rcParams['grid.color'] = PALETTE['light grey']

# ── Constants ──────────────────────────────────────────────────────────────────

HEIGHT: int = 4


# ── Chart functions ────────────────────────────────────────────────────────────

def _set_elements(ax, title='', xlabel='', ylabel='', percentage=False):
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if percentage:
        ax.set_ylim(0.0, 1.0)
    return ax


def multiple_bar_chart(xvalues, yvalues, ax=None, title='', xlabel='', ylabel='',
                       percentage=False, tm=0.05, ft=6, location='best'):
    """Grouped bar chart for comparing multiple metrics across categories."""
    ax = _set_elements(ax=ax, title=title, xlabel=xlabel, ylabel=ylabel, percentage=percentage)
    ngroups = len(xvalues)
    nseries = len(yvalues)
    pos_group = arange(ngroups)
    width = 0.8 / nseries
    pos_center = pos_group + (nseries - 1) * width / 2
    ax.set_xticks(pos_center)
    ax.set_xticklabels(xvalues)

    legend = []
    for i, metric in enumerate(yvalues):
        ax.bar(pos_group, yvalues[metric], width=width,
               edgecolor=LINE_COLOR, color=ACTIVE_COLORS[i])
        legend.append(metric)
        for k, v in enumerate(yvalues[metric]):
            ax.text(pos_group[k], v + tm, f'{v:.2f}',
                    ha='center', fontproperties=FontProperties(size=ft))
        pos_group = pos_group + width
    ax.legend(legend, fontsize='x-small', title_fontsize='small', loc=location)


def plot_confusion_matrix(cnf_matrix, classes_names, ax=None, normalize=False, title=''):
    """Heatmap confusion matrix with value annotations."""
    if ax is None:
        ax = plt.gca()
    if normalize:
        cm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, newaxis]
        title += " - normalized"
    else:
        cm = cnf_matrix
    title += ' - confusion matrix'
    set_printoptions(precision=2)

    _set_elements(ax=ax, title=title, xlabel='Predicted label', ylabel='True label')
    tick_marks = arange(len(classes_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes_names)
    ax.set_yticklabels(classes_names)
    ax.imshow(cm, interpolation='nearest', cmap=cmap_blues)

    fmt = '.2f' if normalize else 'd'
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), color='y', horizontalalignment="center")
