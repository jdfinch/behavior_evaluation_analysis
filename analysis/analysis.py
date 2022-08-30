
import nltk
import random
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from scipy.stats import bootstrap as retarded_bootstrap
from scipy.stats import ttest_ind
import krippendorff
import evaluation_data_definitions as edd

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


def to_file(f):
    def fn_to_file(*args, load=None, reload=None, **kwargs):
        if load:
            return pd.read_pickle(load)
        result = f(*args, **kwargs)
        if reload:
            result.to_pickle(reload)
            return pd.read_pickle(reload)
        return result
    return fn_to_file


def prettify(df, float_prec=None, col_types=None, sort_by=None, to_csv=None, index=True, header=True):
    if col_types:
        for col, type in col_types.items():
            df[col] = df[col].astype(type)
    if sort_by:
        df.sort_values(sort_by, ascending=False, inplace=True)
    if float_prec:
        df = df.round(float_prec)
    if to_csv:
        df.to_csv(to_csv, float_format=f"%.{float_prec}f", header=header, index=index)
    return df


@to_file
def across_evaluations(annotations, evaluation_fn):
    """
    :param annotations: iterable of annotations df to apply evaluation_fn to
    :param evaluation_fn: function (input is annotations df, output is results df)
    :return: results dataframe where first index level codes which evaluation (integer id)
    """
    results = [evaluation_fn(annotation) for annotation in annotations]
    all_results = pd.concat(results, keys=range(len(results)))
    all_results.index.set_names('round', level=0, inplace=True)
    return all_results


def get_example(
        evaluation,
        category,
        label,
        mark,
        bot=None,
        context=0,
        seed=123,
        annotations: pd.DataFrame = None
):
    if annotations is None:
        annotations = evaluation.annotation_dataframe()
    labels = annotations.xs((category, label), level=(1, 2)).reset_index()
    options = labels[labels[0] == mark]
    if bot:
        options = options[options[sym.bot] == bot]
    try:
        example = options.sample(1, random_state=seed)
    except ValueError:
        return f'No samples for {category} {label} {mark} {bot}\n'
    eid = example[sym.item].item()
    if isinstance(eid, tuple):
        did, tid = eid
        turns = evaluation.dialogues[did].turns[max(0, tid-context):tid+1]
        botstring = '' if not bot else f'{bot}~~~\n'
        contextstring = ''.join((
            (
                f'User:  {turn.user_turn}\n'
                f'Sys:   {turn.bot_turn}\n'
            )
            for turn in turns[:-1]
        ))
        turn = turns[-1]
        turnstring = (
            f'User:  {turn.user_turn}\n'
            f'Sys:   {turn.bot_turn}\n'
            f'Label: {label} = {mark}\n'
        )
        return botstring + contextstring + turnstring
    else:
        dialogue = evaluation.dialogues[eid]
        turns = [
            turn
            for turn_pair in dialogue.turns
            for turn in (turn_pair.user_turn, turn_pair.bot_turn)
        ]
        return '\n'.join([f'{dialogue.bot}~~~', *turns, f'Label: {label} = {mark}\n'])


@to_file
def interactor_summary_stats(evaluation: edd.Evaluation):
    num_dialogues = len(evaluation.dialogues)
    mean_turns = (
        sum((
            2*len(d.turns)
            for d in evaluation.dialogues.values()
        ))
        / num_dialogues
    )
    user_turn_len = (
        sum((
            len(nltk.word_tokenize(t.user_turn))
            for d in evaluation.dialogues.values()
            for t in d.turns
        ))
        / sum((
            len(d.turns)
            for d in evaluation.dialogues.values()
        ))
    )
    num_interactors = len({
        unit.worker_id
        for unit in evaluation.work_units.values()
    })
    summary = {
        'dialogues': num_dialogues,
        'mean turns': mean_turns,
        'user turn length': user_turn_len,
        'interactors': num_interactors,
    }
    return pd.DataFrame(summary.values(), summary)

@to_file
def screening_rates_by_label(evaluation: edd.OnboardingEvaluation):
    perfs = {}
    workers_passed = {}
    workers_attempted = {}
    for did, dialogue in evaluation.dialogues.items():
        for attempt in dialogue.attempts:
            work_unit = evaluation.work_units[attempt.work_unit_id]
            round = int(did.split('_')[-1])
            task = work_unit.task
            labels = work_unit.labels
            num_mistakes = len(attempt.mistakes)
            worker = work_unit.worker_id
            accuracy = attempt.performance
            perfs.setdefault(task, []).append((num_mistakes, accuracy))
            workers_attempted.setdefault(task, set()).add(worker)
            if attempt.passed:
                workers_passed.setdefault(task, set()).add(worker)
    screening = {}
    for task, ls in perfs.items():
        mistakes, accuracies = zip(*ls)
        avg_m = sum(mistakes) / len(mistakes)
        avg_a = (
            sum(accuracies) / len(accuracies)
            if all((a is not None for a in accuracies)) else None
        )
        n = len(mistakes)
        attempted = len(workers_attempted.get(task, ()))
        passed = len(workers_passed.get(task, ()))
        screening[task] = {
            'attempted': attempted, 'passed': passed,
            'mistakes': avg_m, 'accuracy': avg_a, 'n': n
        }
    return pd.DataFrame(screening.values(), screening)


@to_file
def agreement_dataframe(annotations, ci=True):
    doubly_annotated = annotations.iloc[:,:2].dropna().astype(int)
    label_groups = doubly_annotated.groupby(level=[sym.category, sym.label])
    kappas = label_groups.apply(fleiss_kappa, ci=ci)
    alphas = label_groups.apply(krippendorfs_alpha, ci=ci)
    agreements = pd.concat((alphas, kappas), axis=1)
    return agreements


def get_singly_annotated(df: pd.DataFrame, seed=None):
    if len(df.columns) == 1:
        return df.astype(int)
    previous_state = random.getstate()
    random.seed(seed)
    df = df.iloc[:,:2]
    mask = df[1].isna()
    singly_annotated = df.iloc[:,0][mask]
    doubly_annotated = df[~mask]
    selection = [random.randint(0, 1) for _ in range(len(doubly_annotated))]
    indices = list(range(len(doubly_annotated)))
    select_annotated = doubly_annotated.values[indices, selection]
    select_annotated = pd.DataFrame(select_annotated, index=doubly_annotated.index)
    annotations = pd.concat((singly_annotated, select_annotated))
    random.setstate(previous_state)
    return annotations.astype(int)


@to_file
def evaluate_comparisons(annotations):
    single_annotated = get_singly_annotated(annotations)
    prop_dfs = []
    for cmp, cmp_label in {-1: 'lose', 0: 'tie', 1: 'win'}.items():
        annotated = single_annotated == cmp
        annotated = annotated.astype(int)
        groups = annotated.groupby(level=[sym.bot, sym.bot_cmp, sym.label])
        props = groups.apply(prop_and_ci)
        props.rename(columns={stat.proportion: cmp_label}, inplace=True)
        prop_dfs.append(props)
    result = pd.concat(prop_dfs, axis=1)
    prop_dfs = []
    for cmp, cmp_label in {-1: 'lose', 0: 'tie', 1: 'win'}.items():
        annotated = single_annotated == cmp
        annotated = annotated.astype(int)
        groups = annotated.groupby(level=[sym.bot, sym.label])
        props = groups.apply(prop_and_ci)
        props.rename(columns={stat.proportion: cmp_label}, inplace=True)
        prop_dfs.append(props)
    result_vs_all = pd.concat(prop_dfs, axis=1)
    others_idx = {sym.bot_cmp: 'others'}
    result_vs_all = result_vs_all.assign(**others_idx)
    levels = [sym.bot, sym.bot_cmp, sym.label]
    result_vs_all = result_vs_all.set_index(sym.bot_cmp, append=True)
    result_vs_all = result_vs_all.reset_index().set_index(levels)
    result = pd.concat((result_vs_all, result))
    return result


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
    'prop_and_ci',

    # utils
    'to_file',
    'prettify',
    'across_evaluations',
    'get_example',
    'interactor_summary_stats',
    'screening_rates_by_label',
    'agreement_dataframe',
    'get_singly_annotated',
    'evaluate_comparisons'

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


