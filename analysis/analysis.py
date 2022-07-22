
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import bootstrap as retarded_bootstrap
from scipy.stats import ttest_ind
import krippendorff

from cattrs import structure

from evaluation_data_definitions import Project


with open('data.json') as f:
    json_data = json.load(f)
    data = structure(json_data, Project)


class sym:

    category = 'category'
    label = 'label'
    bot = 'bot'
    bot_cmp = 'bot comp'
    item = 'item'
    stat = 'stat'


    def __call__(self):
        return [
            v for k, v in self.__class__.__dict__.items()
            if not k.startswith('__')
        ]

    def __iter__(self):
        return iter(self())

    def __contains__(self, item):
        return item in self()


class behavior(sym):
    antisocial = 'antisocial'
    common_contra = 'commonsense contradiction'
    partner_contra = 'partner contradiction'
    self_contra = 'self contradiction'
    ignore = 'ignore'
    incorrect_fact = 'incorrect fact'
    correct_fact = 'correct fact'
    irrelevant = 'irrelevant'
    redundant = 'redundant'
    lack_empathy = 'lack of empathy'
    uninterpretable = 'uninterpretable'
    empathetic = 'empathetic'
    follow_up = 'follow up'
    topic_switch = 'topic switch'
    life_info = 'life info'
    preference_info = 'preference info'
behavior = behavior()

class scale(sym):
    consistent = 'consistent'
    engaging = 'engaging'
    emotional = 'emotional'
    grammatical = 'grammatical'
    informative = 'informative'
    proactive = 'proactive'
    quality = 'quality'
    relevant = 'relevant'
scale = scale()

class category(sym):
    likert_dialogue = 'likert dialogue'
    likert_turn = 'likert turn'
    comparative = 'comparative'
    behavior = 'behavior'

class bot(sym):
    blender2 = 'blender2_3B'
    emora = 'emora'
    bart_fid_rag = 'bart_fid_rag_bcb'
    raranked_blender = 'rerank_blender'
    reranked_blender2 = 'rerank_blender2'
    cem = 'cem'
    dukenet = 'dukenet'
bot = bot()

class stat(sym):
    fleiss_kappa = "Fleiss' kappa"
    kripp_alpha = "Krippendorff's alpha"
    kend_tau = "Kendall's tau"
    mcfad_r2 = "McFadden's pseudo-R-squared"
    ci_low = "CI low"
    ci_high = "CI high"
    proportion = 'proportion'
    mean = 'mean'
    n = 'n'
    likert_dialogue_quality = 'likert dialogue quality'
    likert_turn_quality = 'likert turn quality'

class stage:
    annotation_pilots = 'annotation_pilots'
    annotation_pilots_onboarding = 'annotation_pilots_onboarding'
    bot_pilots = 'bot_pilots'
    extra_unused = 'extra_unused'
    dialogue_collection = 'dialogue_collection'
    student_evaluation = 'student_evaluation'
    student_onboarding = 'student_onboarding'
    student_gold_units = 'student_gold_units'
    mturk_evaluation = 'mturk_evaluation'
    mturk_onboarding = 'mturk_onboarding'
    mturk_gold_units = 'mturk_gold_units'
    surge_evaluation = 'surge_evaluation'
    surge_onboarding = 'surge_onboarding'
    surge_gold_units = 'surge_gold_units'
    expert_evaluation = 'expert_evaluation'



def bootstrap_ci(data, statistic_fn, n_resamples=10**3):
    wrapped_data = [dict(point=d) for d in data]
    statistic_fn_wrapper = lambda ds: statistic_fn([d['point'] for d in ds])
    result = retarded_bootstrap((wrapped_data,), statistic_fn_wrapper, vectorized=False, n_resamples=n_resamples)
    return result.confidence_interval

def fleiss_kappa(df, ci=False):
    """
    :param df: pandas dataframe: items x labeler: label
    :return: pandas series of kappa, CI low, CI high
    """
    def _fleiss_kappa(M):
          """
          See `Fleiss' Kappa <https://en.wikipedia.org/wiki/Fleiss%27_kappa>`_.
          :param M: a matrix of shape (:attr:`N`, :attr:`k`) where `N` is the number of subjects and `k` is the number of categories into which assignments are made. `M[i, j]` represent the number of raters who assigned the `i`th subject to the `j`th category.
          :type M: numpy matrix
          """
          N, k = M.shape  # N is # of items, k is # of categories
          n_annotators = float(np.sum(M[0, :]))  # # of annotators
          p = np.sum(M, axis=0) / (N * n_annotators)
          P = (np.sum(M * M, axis=1) - n_annotators) / (n_annotators * (n_annotators - 1))
          Pbar = np.sum(P) / N
          PbarE = np.sum(p * p)
          if (1 - PbarE) == 0:
            kappa = np.nan
          else:
            kappa = (Pbar - PbarE) / (1 - PbarE)
          return kappa
    counts = df.stack().groupby(level=df.index.names).value_counts().unstack(fill_value=0)
    cnp = counts.to_numpy().astype(int)
    kappa = _fleiss_kappa(cnp)
    if ci:
        low, high = bootstrap_ci(cnp, lambda ds: _fleiss_kappa(np.array(ds)))
        n = len(df)
        result = {
            stat.fleiss_kappa: kappa,
            stat.ci_low: low, stat.ci_high: high,
            stat.n: len(df)
        }
    else:
        result = {
            stat.fleiss_kappa: kappa,
            stat.n: len(df)
        }
    return pd.Series(result.values(), result)


def krippendorfs_alpha(df, ci=True):
    """
    :param df: pandas dataframe: items x labeler: label
    :return:
    """
    ratings = df.to_numpy().astype(int)
    ka = lambda x: krippendorff.alpha(x.T, level_of_measurement='ordinal')
    try:
        alpha = ka(ratings)
    except AssertionError:
        alpha = None
    if ci:
        try:
            low, high = bootstrap_ci(ratings, lambda x: ka(np.array(x)))
        except AssertionError:
            low, high = None, None
        result = {
            stat.kripp_alpha: alpha,
            stat.ci_low: low, stat.ci_high: high,
            stat.n: len(df)
        }
    else:
        result = {
            stat.kripp_alpha: alpha,
            stat.n: len(df)
        }
    return pd.Series(result.values(), result)


def mean_and_ci(df: pd.DataFrame):
    vals = df.to_numpy()
    mean = vals.mean()
    def t_conf_int(data, alpha=0.05):
        mean = sum(data) / len(data)
        stderr = stats.sem(data)
        return stats.t.interval(
            alpha=(1 - alpha),
            df=len(data) - 1,
            loc=mean,
            scale=stderr
        )
    (low,), (high,) = t_conf_int(vals)
    result = {stat.mean: mean, stat.ci_low: low, stat.ci_high: high, stat.n: len(df)}
    return pd.Series(result.values(), result)


def prop_and_ci(df: pd.DataFrame):
    vals = df.to_numpy()
    positives = vals.sum()
    total = len(vals)
    prop = positives / total
    low, high = sm.stats.proportion_confint(positives, total, method='wilson')
    if not isinstance(low, float):
        low, = low
    if not isinstance(high, float):
        high, = high
    result = {stat.proportion: prop, stat.ci_low: low, stat.ci_high: high, stat.n: len(df)}
    return pd.Series(result.values(), result)


def sign_tests(df: pd.DataFrame):
    """
    :param df: (bot, data point) x label -> -1/0/1
    :return: p values of test on each bot pair (pd.Series)
    """

def t_tests(df: pd.DataFrame):
    """
    :param df: (bot, data point) x label -> score
    :return: p values of test on each bot pair (pd.Series)
    """


def prop_tests(df: pd.DataFrame):
    """
    :param df: (bot, data point) x label -> 0/1
    :return: p values of test on each bot pair (pd.Series)
    """


def regressions(df):
    """
    :param df: dialogue x (*features, quality) -> value
    :return: *(coef, low, high), mcfadden r^2
    """


__all__ = [

    # the data
    'data',

    # symbols
    'sym',
    'scale',
    'behavior',
    'category',
    'stage',
    'bot',
    'stat',

    # stats
    'fleiss_kappa',
    'krippendorfs_alpha',
    'mean_and_ci',
    'prop_and_ci'

]


if __name__ == '__main__':

    td = {
        ('emora', 1):   {'x': 2, 'y': 9, 'q': 1},
        ('emora', 2):   {'x': 3, 'y': 5, 'q': 2},
        ('emora', 3):   {'x': 5, 'y': 1, 'q': 3},
        ('blender', 1): {'x': 5, 'y': 2, 'q': 1},
        ('blender', 2): {'x': 4, 'y': 3, 'q': 2},
        ('blender', 3): {'x': 5, 'y': 4, 'q': 2},
        ('blender', 4): {'x': 4, 'y': 5, 'q': 3}
    }
    tdf = pd.DataFrame(td.values(), td)

    result = t_tests(tdf)
    print(result)


