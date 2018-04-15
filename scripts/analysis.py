import csv
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from copy import deepcopy

rcParams['savefig.directory'] = '../report/figs/'

DATA_DIR = '/media/robin/DATA/Research/Bioinformatics/data/BINF-F401/'

def r_to_p(rs, N):
    rs = np.abs(rs)
    ts = rs*np.sqrt((N-2)/(1-rs*rs))
    ps = 1 - (2*(stats.t(df=N-2).cdf(ts)) - 1)
    return ps

def plot_p_vs_r(r_max: float=.5, n_points: int=5000, N: int=250, _ps: tuple=(.05, .01, .001, 1e-8)):
    rs = r_max * np.arange(n_points+1) / n_points
    ps = r_to_p(rs, N)
    fig = plt.figure(figsize=[7.75]*2)
    ax = fig.add_subplot(111)
    ax.plot(rs, ps, lw=2)
    xticks = list(plt.xticks()[0])
    quantiles = list()
    for p in _ps:
        ax.plot([0, r_max], [p, p], 'r-.', lw=2)
        quantile = rs[np.where(ps <= p)][0]
        ax.plot([quantile, quantile], [ps[-1], ps[0]], 'r-.', lw=2)
        xticks.append(quantile)
        quantiles.append(quantile)
    xticks.sort()
    ax.set_xticks(xticks)
    ax.set_xticklabels(['{:<.4}'.format(tick) for tick in xticks])
    for x, tick in zip(xticks, ax.get_xticklabels()):
        if x in quantiles:
            tick.set_rotation(37.5)
    ax.set_yscale('log')
    ax.set_xlabel('Pearson\'s r')
    ax.set_ylabel('p-value')
    plt.title('p-value vs Pearson\'s r with {} samples'.format(N))
    plt.grid()
    plt.show()

def _float(s: str) -> np.float:
    return float(s) if s != 'NA' else np.nan

def _extract_list(s: str) -> list:
    return tuple(s.split(',')) if s != 'NA' else tuple()

class Sample:
    def __init__(self, row_items: list):
        self.name = row_items[0]
        self.absolute_purity = _float(row_items[1])
        self.ihc_purity = _float(row_items[2])
        self.mutations = {
            'CDH1': _extract_list(row_items[3]),
            'GATA3': _extract_list(row_items[4]),
            'MAP3K1': _extract_list(row_items[5]),
            'PIK3CA': _extract_list(row_items[6]),
            'TP53': _extract_list(row_items[7])
        }

    def has_nan(self) -> bool:
        return np.isnan([self.absolute_purity, self.ihc_purity]).any()

    def nb_mutations(self, gene: str='') -> int:
        if gene == '':
            return np.sum([len(self.mutations[gene]) for gene in self.mutations])
        else:
            return len(self.mutations[gene])

    def __str__(self):
        return ('name:             {}\n' + \
                'ABSOLUTE:         {}\n' + \
                'IHC:              {}\n' + \
                'CDH1 mutations:   {}\n' + \
                'GATA3 mutations:  {}\n' + \
                'MAP3K1 mutations: {}\n' + \
                'PIK3CA mutations: {}\n' + \
                'TP53:             {}') \
               .format(self.name, self.absolute_purity, self.ihc_purity, self.mutations['CDH1'], self.mutations['GATA3'],
                       self.mutations['MAP3K1'], self.mutations['PIK3CA'], self.mutations['TP53'])

    @staticmethod
    def mutations():
        return ('CDH1', 'GATA3', 'MAP3K1', 'PIK3CA', 'TP53')

def load_sample_data(path: str):
    return [Sample(row) for row in load_tsv(path)]

def load_tsv(path: str) -> list:
    ret = list()
    with open(DATA_DIR + path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) == 0:
                continue
            if not ''.join(row).strip().startswith('#'):
                ret.append(row)
    return ret

def plot_abs_vs_ihc(sample_data: list):
    BIN_WIDTH = .05
    fig = plt.figure(figsize=(8, 12))
    ax = fig.add_subplot(211)
    x, y = map(np.asarray, zip(*[(r.absolute_purity, r.ihc_purity) for r in sample_data if not r.has_nan()]))
    pearson_r = np.corrcoef(x, y)[0, 1]
    spearman_r = stats.spearmanr(x, y)[0]
    ax.plot(x, y, 'ro', label='samples')
    ax.plot((0, 1), (0, 1), 'k-.', label='ABSOLUTE = IHC')
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.set_xlabel('ABSOLUTE purity')
    ax.set_ylabel('IHC purity')
    ax.set_title('Pearson\'s r: {:.3f}\nSpearman\'s r: {:.3f}'.format(pearson_r, spearman_r))
    plt.legend(loc='lower right')
    ax = fig.add_subplot(212)
    ax.hist(y-x, bins=np.arange(2 / BIN_WIDTH + 1) * BIN_WIDTH - 1, weights=100*np.ones(len(y)) / len(y), label='pdf')
    ylim = ax.get_ylim()
    plt.plot((0, 0), (0, ylim[1]), 'k-.', label='ABSOLUTE = IHC')
    ax.set_xlim((-1, +1))
    ax.set_ylim(ylim)  # be sure that ylim hasn't changed
    ax.set_xlabel('IHC - ABSOLUTE')
    ax.set_ylabel('% of samples')
    ax.set_xticks(np.arange(9) / 4 - 1)
    ax.set_title('IHC > ABSOLUTE for {:d} samples out of {:d}'.format((x<y).sum(), x.shape[0]))
    plt.legend(loc='best')
    plt.show()

def plot_mutations_distribution(sample_data: list):
    bins = np.arange(5) - .5
    fig = plt.figure(figsize=(9, 10.5))
    for idx, gene in enumerate(list(Sample.mutations()) + ['']):
        ax = fig.add_subplot(3, 2, 1+idx)
        nb_mutations = [r.nb_mutations(gene) for r in sample_data]
        ns = ax.hist(nb_mutations, bins=bins, color='salmon', alpha=.8,
                     weights=100*np.ones(len(nb_mutations)) / len(nb_mutations),
                     label='pdf')[0]
        ax.plot(np.arange(len(ns)), np.cumsum(ns), 'k-', label='cdf')
        ax.set_title('{}'.format(gene) if gene != '' else 'Cumulated')
        if idx > 3:
            ax.set_xlabel('Number of mutations')
        if idx % 2 == 0:
            ax.set_ylabel('% of samples')
        ax.set_xticks(np.arange(np.max(nb_mutations)+1))
        ax.set_xlim((bins[0], bins[-1]))
        ax.set_yticks(np.arange(11)*10)
        ax.set_ylim((0, 100))
        plt.grid(True)
        plt.legend(loc='best')
    plt.show()

def get_p_value(abs_purities: np.ndarray, nb_mutations: np.ndarray, n_classes: int=20) -> np.ndarray:
    table = np.zeros((n_classes, len(np.unique(nb_mutations))), dtype=np.int)
    x_min, x_max = np.min(abs_purities), 1
    for p, n in zip(abs_purities, nb_mutations):
        idx = int((p - x_min)*n_classes/(x_max-x_min))
        table[idx, n] += 1
    table = table[table.sum(axis=1) > 0,:]
    return stats.chi2_contingency(table.T, correction=False)[1]

def print_upper_triangle(array: np.ndarray, diagonal: bool=False):
    for i, row in enumerate(array):
        if not diagonal:
            i += 1
        j = 0
        while j < i:
            print(' ' * 6, end='\t')
            j += 1
        while j < array.shape[1]:
            print('{:+1.3f}'.format(row[j]), end='\t')
            j += 1
        print('')

def print_correlation_between_nb_mutations(sample_data: list):
    all_genes = Sample.mutations()
    nb_mutations = list()
    mutations = list()
    for i in range(len(all_genes)):
        nb_mutations.append([r.nb_mutations(all_genes[i]) for r in sample_data])
        mutations.append([1 if r.nb_mutations(all_genes[i]) > 0 else 0 for r in sample_data])
    print('Correlations (nb of mutations):')
    rs = np.corrcoef(nb_mutations)
    print_upper_triangle(rs)
    print('p-values:')
    print_upper_triangle(r_to_p(rs, len(sample_data)))
    print('Correlations (presence of mutations):')
    rs = np.corrcoef(mutations)
    print_upper_triangle(rs)
    print('p-values:')
    print_upper_triangle(r_to_p(rs, len(sample_data)))

def mutations_analysis(sample_data: list, purity: str='absolute', binarize: bool=False):
    if purity.lower() == 'absolute':
        get_purity = lambda r: r.absolute_purity
    elif purity.lower() == 'ihc':
        get_purity = lambda r: r.ihc_purity
    else:
        raise ValueError('Unknown purity: ' + purity)
    xlim = (-.5, 1.5 if binarize else 3.5)
    fig = plt.figure(figsize=(9, 10.5))
    fig.suptitle('Relationship between{} gene mutations and {} purity'.format('' if binarize else ' amount of', purity.upper()), fontsize=16)
    for idx, gene in enumerate(list(Sample.mutations()) + ['']):
        purities, nb_mutations = zip(*[(get_purity(r), r.nb_mutations(gene)) for r in sample_data if not np.isnan(get_purity(r))])
        if binarize:
            nb_mutations = (np.asarray(nb_mutations) > 0).astype(np.int)
        p_value = get_p_value(purities, nb_mutations)
        ax = fig.add_subplot(3, 2, 1+idx)
        ax.plot(nb_mutations, purities, 'ro', mec='orange')
        ax.set_title((gene if gene != '' else 'Cumulated') + '  (p = {:.3f})'.format(p_value))
        if idx > 3:
            if binarize:
                ax.set_xlabel('Mutated')
            else:
                ax.set_xlabel('Number of mutations')
        if idx % 2 == 0:
            ax.set_ylabel('{} purity'.format(purity.upper()))
        ax.set_xlim(xlim)
        ax.set_xticks(np.arange(np.max(nb_mutations)+1))
        if binarize:
            ax.set_xticklabels(('No', 'Yes'))
        ax.set_ylim((0, 1))
    plt.show()

class ClusteredSample:
    def __init__(self, row):
        self.name = row[0]
        self.cluster = int(row[1])
        self.silhouette_width = float(row[2])

    def __str__(self):
        return '{}\t{}\t{}'.format(self.name, self.cluster, self.silhouette_width)

    def __lt__(self, other):
        return (self.cluster < other.cluster) or (self.cluster == other.cluster and self.silhouette_width < other.silhouette_width)

def get_clusters(sample_data: list, keep_other_samples: bool=False) -> list:
    sample_names = set([sample.name for sample in sample_data])
    clustering = load_tsv('BRCA_clustering.tsv')
    if keep_other_samples:
        clustering = list(map(ClusteredSample, clustering))
    else:
        clustering = [ClusteredSample(c) for c in clustering if c[0] in sample_names]
    clustering.sort()
    clusters = list()
    for c in clustering:
        if c.cluster > len(clusters):
            clusters.append(list())
        clusters[-1].append(c)
    return clusters

def plot_silhouette(sample_data: list, keep_other_samples: bool=False):
    clusters = get_clusters(sample_data, keep_other_samples)
    nb_elems = 0
    for idx, cluster in reversed(list(enumerate(clusters))):
        color = cm.spectral(idx / len(clusters))
        plt.fill_betweenx(np.arange(nb_elems, nb_elems+len(cluster)), [c.silhouette_width for c in cluster],
                          facecolor=color, edgecolor='k', label='Cluster {}'.format(idx+1))
        nb_elems += len(cluster)
    plt.legend(loc='lower right')
    plt.yticks([])
    plt.grid()
    plt.ylim((-5, np.sum([len(cluster) for cluster in clusters])+5))
    plt.xlim((-.25, .75))
    NB_XTICKS = 20
    plt.xticks(np.arange(NB_XTICKS+1) / NB_XTICKS - .25, rotation=45)
    plt.xlabel('Silhouette Width')
    plt.show()

def cluster_to_abs(cluster: list, sample_data: list) -> list:
    purities = list()
    for clustered_sample in cluster:
        for sample in sample_data:
            if clustered_sample.name == sample.name:
                purities.append(sample.absolute_purity)
                break
    return purities

def plot_abs_per_cluster(sample_data: list):
    clusters = get_clusters(sample_data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_xs = list()
    for idx, cluster in enumerate(clusters):
        color = cm.spectral(idx / len(clusters))
        all_xs.append(list())
        for clustered_sample in cluster:
            for sample in sample_data:
                if sample.name == clustered_sample.name:
                    all_xs[-1].append(sample.absolute_purity)
                    break
    plt.boxplot(all_xs, showmeans=True)
    plt.ylabel('ABSOLUTE purity')
    plt.xlabel('Cluster')
    plt.xlim(0, len(clusters)+1)
    plt.xticks(range(1, 8))
    plt.grid(axis='y')
    plt.legend()
    plt.show()

def print_silhouette_details(sample_data: list):
    clusters = get_clusters(sample_data)
    for idx, cluster in enumerate(clusters):
        print('Cluster {}'.format(idx+1))
        silhouette_widths = np.asarray([c.silhouette_width for c in cluster])
        nb_negative_samples = (silhouette_widths < 0).sum()
        print('Nb negative s_w: {} out of {}  ({:.3f}%)' \
              .format(nb_negative_samples, len(cluster),
                      100 * nb_negative_samples / len(cluster)
              )
        )

def plot_silhouette_distribution_per_cluster(sample_data: list):
    clusters = get_clusters(sample_data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_xs = list()
    for idx, cluster in enumerate(clusters):
        color = cm.spectral(idx / len(clusters))
        all_xs.append(list())
        for clustered_sample in cluster:
            all_xs[-1].append(clustered_sample.silhouette_width)
    plt.boxplot(all_xs, showmeans=True)
    plt.ylabel('Silhouette Width')
    plt.xlabel('Cluster')
    plt.xlim(0, len(clusters)+1)
    plt.xticks(range(1, 8))
    plt.grid(axis='y')
    plt.legend()
    plt.show()

def compare_clustered_abs_means(sample_data):
    clusters = get_clusters(sample_data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    all_xs = list()
    for idx, cluster in enumerate(clusters):
        color = cm.spectral(idx / len(clusters))
        all_xs.append(list())
        for clustered_sample in cluster:
            for sample in sample_data:
                if sample.name == clustered_sample.name:
                    all_xs[-1].append(sample.absolute_purity)
                    break
    ps = np.zeros([len(clusters)]*2, dtype=np.float)
    non_significant = list()
    for i in range(len(all_xs)):
        for j in range(i+1, len(all_xs)):
            p = stats.ttest_ind(all_xs[i], all_xs[j], equal_var=False)[-1]
            ps[j,i] = ps[i,j] = p
            if p > .05:
                non_significant.append((i, j))
    fig = plt.figure(figsize=(8, 8))
    modified_cmap = deepcopy(cm.get_cmap('inferno'))
    modified_cmap.set_bad((0,0,0))
    ax = fig.add_axes([0.1, 0.1, 0.77, 0.77])
    ax_color = fig.add_axes([0.875, 0.1, 0.03, 0.77])
    img = ax.matshow(ps, cmap=modified_cmap, norm=LogNorm(vmin=1e-12, vmax=1))
    ax.set_xticklabels(map(str, range(len(clusters)+1)))
    ax.set_yticklabels(map(str, range(len(clusters)+1)))
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Cluster')
    for i, j in non_significant:
        ax.plot(i, j, 'kx', markersize=25, markeredgewidth=10)
        ax.plot(j, i, 'kx', markersize=25, markeredgewidth=10)
    ax.set_xlim(-.5, len(clusters)-.5)
    ax.set_ylim(-.5, len(clusters)-.5)
    ax.xaxis.tick_bottom()
    ax.set_title('Significance of the difference between\nmeans of each cluster pair')
    fig.colorbar(img, ticks=np.power(10., -np.arange(13)), cax=ax_color, format='%1.0e')
    fig.show()
    plt.show()

def plot_clusters_basic_stats(sample_data):
    clusters = get_clusters(sample_data)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(len(clusters))+1, [len(cluster) for cluster in clusters],
            'b-.', marker='o', markersize=8, lw=1, label='Cluster size')
    ax.plot(np.arange(len(clusters))+1, [(np.array([sample.silhouette_width for sample in cluster]) < 0).sum() for cluster in clusters],
            'm-.', marker='o', markersize=8, lw=1, label='silhouette < 0')
    ax_right = ax.twinx()
    ax_right.plot(np.arange(len(clusters))+1, [np.max(cluster_to_abs(cluster, sample_data)) for cluster in clusters],
                  'g-.', marker='*', markersize=15, lw=1, label='max purity')
    ax_right.plot(np.arange(len(clusters))+1, [np.mean(cluster_to_abs(cluster, sample_data)) for cluster in clusters],
                  'c-.', marker='*', markersize=15, lw=1, label='avg purity')
    ax_right.plot(np.arange(len(clusters))+1, [np.min(cluster_to_abs(cluster, sample_data)) for cluster in clusters],
                  'r-.', marker='*', markersize=15, lw=1, label='min purity')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of samples')
    ax_right.set_ylabel('ABSOLUTE purity')
    ax.set_xlim(.5, len(clusters)+.5)
    ax.set_yticks(np.arange(11) / 10 * ax.get_ylim()[1])
    ax_right.set_yticks(np.arange(11)/10)
    ax.set_title('Clusters size and ABSOLUTE purity statistics per cluster')
    ax.grid(axis='y')
    ax.legend(loc='center left', bbox_to_anchor=(0., .35))
    ax_right.legend(loc='upper right')
    plt.show()

def main():
    #plot_p_vs_r()
    sample_data = load_sample_data('BRCA-3.tsv')
    #plot_abs_vs_ihc(sample_data)
    #plot_mutations_distribution(sample_data)
    #print_correlation_between_nb_mutations(sample_data)
    #mutations_analysis(sample_data)
    #mutations_analysis(sample_data, 'ihc')
    #mutations_analysis(sample_data, binarize=True)
    #mutations_analysis(sample_data, 'ihc', binarize=True)
    #plot_silhouette(sample_data)
    #plot_abs_per_cluster(sample_data)
    #print_silhouette_details(sample_data)
    #plot_silhouette_distribution_per_cluster(sample_data)
    #compare_clustered_abs_means(sample_data)
    plot_clusters_basic_stats(sample_data)

if __name__ == '__main__':
    main()
