# Adapted from World Bank GitHub
# worldbank/ML-classification-algorithms-poverty
# https://github.com/worldbank/ML-classification-algorithms-poverty

# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from IPython.display import display

# import itertools
from IPython.display import display

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model as KerasLoadModel


from sklearn.metrics import (
    confusion_matrix,
    log_loss,
    roc_auc_score,
    accuracy_score,
    precision_score
)

from sklearn.metrics import (
    recall_score,
    f1_score,
    cohen_kappa_score,
    roc_curve,
    auc
)


def calculate_metrics(y_test, y_pred, y_prob=None, sample_weights=None):
    """Cacluate model performance metrics"""

    # Dictionary of metrics to calculate
    metrics = {}
    metrics['confusion_matrix']  = confusion_matrix(y_test, y_pred, sample_weight=sample_weights)
    metrics['roc_auc']           = None
    metrics['accuracy']          = accuracy_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['precision']         = precision_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['recall']            = recall_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['f1']                = f1_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['cohen_kappa']       = cohen_kappa_score(y_test, y_pred)
    metrics['cross_entropy']     = None
    metrics['fpr']               = None
    metrics['tpr']               = None
    metrics['auc']               = None

    # Populate metrics that require y_prob
    if y_prob is not None:
        clip_yprob(y_prob)
        metrics['cross_entropy']     = log_loss(y_test,
                                                clip_yprob(y_prob), 
                                                sample_weight=sample_weights)
        metrics['roc_auc']           = roc_auc_score(y_test,
                                                     y_prob, 
                                                     sample_weight=sample_weights)

        fpr, tpr, _ = roc_curve(y_test,
                                y_prob, 
                                sample_weight=sample_weights)
        metrics['fpr']               = fpr
        metrics['tpr']               = tpr
      # metrics['auc']               = auc(fpr, tpr, reorder=True)
        metrics['auc']               = auc(fpr, tpr)
    
    return metrics



# Evaluate model performance. Options to display results
metrics = calculate_metrics(y_test, y_pred, y_prob)

# Provide an output name if none given:
if model_name is None:
    model_name = 'score'
if prefix is not None:
    model_name = prefix + "_" + model_name

metrics['name'] = model_name


display_model_comparison(comp_models, show_roc=(y_prob is not None))

# Derived from https://pandas.pydata.org/pandas-docs/stable/style.html
def highlight_abs_min(data, color='steelblue'):
    '''
    highlight the maximum in a Series or DataFrame
    '''
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_abs_min = (data.abs() == data.abs().min())
        return [attr if v else '' for v in is_abs_min]
    else:  # from .apply(axis=None)
        is_abs_min = data.abs() == data.abs().min().min()
        return pd.DataFrame(np.where(is_abs_min, attr, ''),
                            index=data.index, columns=data.columns)


def plot_roc(models, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, model in enumerate(models):
        if model['fpr'] is not None:
            if i==0: 
                lw = 3
            else:
                lw = 1
            ax.plot(model['fpr'], model['tpr'], lw=lw,
                    label='{} AUC = {:0.2f}'.format(model['name'], model['roc_auc']))
    ax.set_title('Receiver Operating Characteristic')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlim([-0.01,1])
    ax.set_ylim([0,1.01])
    ax.legend(loc='lower right')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return


# Derived from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, 
                          reverse=True,
                          ax=None, fig=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, 
                          colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if reverse is True:
        cm = cm[::-1,::-1]
        classes = classes[::-1]
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.7)

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = "{:.0f}".format(cm[i,j])
        if normalize:
            txt = txt + "\n{:0.1%}".format(cm_norm[i,j])
        
        ax.text(j, i, txt, fontsize=14, fontweight='bold',
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.grid('off')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return

def get_rank_order(x):
    # maximize most metrics
    # except those we want to minimize
    ascending = x.name in ('cross_entropy', 'pov_error_rate')
    
    return x.rank(ascending=ascending)


def display_model_comparison(comp_models, 
                             show_roc=False, 
                             show_cm=True, 
                             show_pov_rate_error=False, 
                             highlight_best=True, 
                             transpose=False, 
                             rank_order=False):
    # Don't highlight if only showing one model
    if len(comp_models) == 1:
        highlight_best = False

    # Plot ROC and confusion matrix in a single row
    n_axes = sum([show_roc, show_cm])
    if n_axes > 0:
        fig, axes = plt.subplots(1,n_axes, figsize=(5*n_axes,4))
        i = 0
        
        if n_axes == 1:
            axes = [axes]

        # Plot ROC curve
        if show_roc:
            ax = axes[i]
            plot_roc(comp_models, ax)
            i += 1

        # Plot confusion matrices
        if show_cm:
            ax = axes[i]
            conf_mat = comp_models[0]['confusion_matrix']
            plot_confusion_matrix(conf_mat, 
                                  classes=['Non-poor', 'Poor'], 
                                  reverse=True,
                                  normalize=True,
                                  ax=ax, 
                                  fig=fig)

        fig.tight_layout()
        plt.show()

    # Display score table
    disp_metrics = ['accuracy', 'recall', 'precision', 'f1', 'cross_entropy', 'roc_auc', 'cohen_kappa']
    disp_types = {'max': ['accuracy', 'recall', 'precision', 'f1', 'roc_auc', 'cohen_kappa'], 
                  'min': ['cross_entropy']}
    if show_pov_rate_error == True:
        disp_metrics.append('pov_rate_error')
        disp_types['abs_min'] = ['pov_rate_error']
    
    met_df = []
    for m in comp_models:
        m_disp = {x[0]: x[1] for x in m.items() if x[0] in disp_metrics}
        met_df.append(pd.DataFrame.from_dict(m_disp, orient='index')
                      .rename(columns={0:m['name']})
                      .loc[disp_metrics])
    met_df = pd.concat(met_df, axis=1)

    if transpose:
        met_df = met_df.T
        axis = 0
        highlight_index = {'max': pd.IndexSlice[:, disp_types['max']], 
                           'min': pd.IndexSlice[:, disp_types['min']]}
        if show_pov_rate_error == True:
            highlight_index['abs_min'] = pd.IndexSlice[:, disp_types['abs_min']]
        
        if rank_order:
            met_df['mean_rank'] = met_df.apply(get_rank_order).mean(axis=1)
            met_df = met_df.sort_values('mean_rank')
            
    else:
        axis = 1
        highlight_index = {'max': pd.IndexSlice[disp_types['max'], :], 
                           'min': pd.IndexSlice[disp_types['min'], :]}
        if show_pov_rate_error == True:
            highlight_index['abs_min'] = pd.IndexSlice[disp_types['abs_min'], :]


    scores = met_df.style
    if highlight_best:
        scores = scores.highlight_max(subset=highlight_index['max'], 
                                      color='steelblue',
                                      axis=axis)
        scores = scores.highlight_min(subset=highlight_index['min'], 
                                      color='steelblue',
                                      axis=axis)
        if show_pov_rate_error == True:
            scores = scores.apply(highlight_abs_min, 
                                  subset=highlight_index['abs_min'], 
                                  color='steelblue',
                                  axis=axis)

    display(scores.set_caption("Model Scores"))
    return met_df

def display_precision_recall(results):
    if len(results > 10):
        fig, ax = plt.subplots(figsize=(6,len(results)*0.3))
    else:
        fig, ax = plt.subplots()
    (results.sort_values('mean_rank', ascending=False)
     [['recall', 'precision']]
     .plot.barh(title='Precision and Recall', ax=ax))
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles[::-1], labels=labels[::-1], bbox_to_anchor=(1.3, 1))
    plt.show()

def display_feat_ranks(feats):
    feat_rankings = []
    for key, value in sorted(feats.items()):
        if type(value) == pd.DataFrame:
            if 'abs' in value.columns:
                ranks = (pd.DataFrame(value['abs'].rank(ascending=False).rename(key)))
            if 'importance' in value.columns:
                ranks = (pd.DataFrame(value['importance'].rank(ascending=False).rename(key)))
            feat_rankings.append(ranks)
    feat_rankings = pd.concat(feat_rankings, axis=1)
    mean_rank = feat_rankings.mean(axis=1)
    counts = feat_rankings.count(axis=1)
    feat_rankings['mean_rank'] = mean_rank
    feat_rankings['count'] = counts
    feat_rankings = feat_rankings.sort_values('mean_rank', ascending=True)

    display(feat_rankings.style.set_caption("Feature Ranking"))
    return feat_rankings


# Clip y used to prevent y=0

def clip_yprob(y_prob):
    """Clip yprob to avoid 0 or 1 values. Fixes bug in log_loss calculation
    that results in returning nan."""
    eps = 1e-15
    y_prob = np.array([x if x <= 1-eps else 1-eps for x in y_prob])
    y_prob = np.array([x if x >= eps else eps for x in y_prob])
    return y_prob

def calculate_metrics(y_test, y_pred, y_prob=None):
    """Cacluate model performance metrics"""

    # Dictionary of metrics to calculate
    metrics = {}
    metrics['confusion_matrix']  = confusion_matrix(y_test, y_pred, sample_weight=sample_weights)
    metrics['roc_auc']           = None
    metrics['accuracy']          = accuracy_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['precision']         = precision_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['recall']            = recall_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['f1']                = f1_score(y_test, y_pred, sample_weight=sample_weights)
    metrics['cohen_kappa']       = cohen_kappa_score(y_test, y_pred)
    metrics['cross_entropy']     = None
    metrics['fpr']               = None
    metrics['tpr']               = None
    metrics['auc']               = None

    # Populate metrics that require y_prob
    if y_prob is not None:
        clip_yprob(y_prob)
        metrics['cross_entropy']     = log_loss(y_test,
                                                clip_yprob(y_prob))
        metrics['roc_auc']           = roc_auc_score(y_test,
                                                     y_prob)
                                                     
        fpr, tpr, _ = roc_curve(y_test,y_prob) 
                                
        metrics['fpr']               = fpr
        metrics['tpr']               = tpr
        metrics['auc']               = auc(fpr, tpr)
    
    return metrics

def evaluate_model(y_test,
                   y_pred,
                   y_prob=None,
                   sample_weights=None,
                   show=True,
                   compare_models=None,
                   store_model=False,
                   model_name=None,
                   prefix=None,
                   country=None,
                   model=None,
                   features=None,
                   predict_pov_rate=True):
    """Evaluate model performance. Options to display results and store model"""

   
