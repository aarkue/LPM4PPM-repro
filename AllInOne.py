# fmt: off
import os
# Disables `this does not indicate an error and you can ignore this message` tensorflow log
# Note, that this should be executed before tensorflow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import ast
from sklearn.model_selection import train_test_split
from pm4py.algo.conformance.alignments.petri_net.algorithm import apply_log as align_log
from pm4py.algo.conformance.alignments.petri_net.algorithm import Parameters as AlignmentParameters
import pm4py
import pandas as pd
import numpy as np
from typing import Any, Dict, FrozenSet, List, Literal, Optional, Set, Tuple, Union
import glob
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import random
import math
import itertools
import json
import datetime
from datetime import time
from copy import deepcopy
import keras_tuner
from tensorflow.keras.layers import Dense, Flatten, Embedding, LSTM, Concatenate, Input
from tensorflow.keras.models import Sequential, Model
import tensorflow as tf
# fmt: on


# Constants
CASE_ID_COL = pm4py.util.constants.CASE_CONCEPT_NAME
ACTIVITY_COL = pm4py.util.xes_constants.DEFAULT_NAME_KEY
TIME_COL = pm4py.util.xes_constants.DEFAULT_TIMESTAMP_KEY
BASE_CASE_ID_COL = 'base:'+CASE_ID_COL
CASE_LEN_COL = 'case:length'
LABEL_COL = 'label'
COLS_TO_KEEP = [CASE_ID_COL, ACTIVITY_COL,
                CASE_LEN_COL, BASE_CASE_ID_COL, LABEL_COL, TIME_COL]
INPUT_MODE = Literal['OH_ACT', 'WOH_BIN',
                     'WOH_FREQ', 'EMB_ACT', 'EMB_ACT+LPM', 'EMB_LPM']
INPUT_MODES: List[INPUT_MODE] = [
    'OH_ACT',
    'WOH_BIN',
    'WOH_FREQ',
    'EMB_ACT',
    'EMB_ACT+LPM',
    'EMB_LPM',
]
RAND_SEED = 2023


def convert_csv_log_to_xes(log_path: str, out_xes_path: str):
    COLS_TO_KEEP_FOR_XES = [CASE_ID_COL, ACTIVITY_COL, LABEL_COL, TIME_COL]

    log_df = pd.read_csv(log_path, sep=";")
    activity_col_name = "Activity" if "Activity" in log_df.columns else "Activity code"
    case_id_col_name = "Case ID"
    label_col_name = "label"
    timestamp_col_name = "time:timestamp" if "time:timestamp" in log_df.columns else "Complete Timestamp"
    all_expected_cols_present = activity_col_name in log_df.columns and case_id_col_name in log_df.columns and label_col_name in log_df.columns and timestamp_col_name in log_df.columns
    if not all_expected_cols_present:
        print(log_df.columns)
    log_df[ACTIVITY_COL] = log_df[activity_col_name].astype('string')
    log_df[CASE_ID_COL] = log_df[case_id_col_name].astype('string')
    log_df[TIME_COL] = pd.to_datetime(log_df[timestamp_col_name])
    log_df[LABEL_COL] = log_df[label_col_name].astype('string')
    log_df = log_df.drop(
        [c for c in log_df.columns if c not in COLS_TO_KEEP_FOR_XES], axis=1)
    pm4py.write_xes(log_df, out_xes_path)


def discover_lpms_inductive(xes_log_path: str, lpm_path: str, num_lpms: int = 100, noise_thresh=1.0):
    log = pm4py.read_xes(xes_log_path)
    from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
    acts = log[ACTIVITY_COL].unique()
    models = dict()
    for a1, a2, a3, a4 in itertools.combinations(acts, 4):
        # Skip many models early on
        if random.random() > 1/math.pow(len(acts), 2) or frozenset([a1, a2, a3, a4]) in models:
            continue
        log_copy = log[log[ACTIVITY_COL].isin([a1, a2, a3, a4])]
        log_copy = pm4py.filter_variants_top_k(log_copy,2)
        pn = pm4py.discover_petri_net_inductive(log_copy, False, noise_thresh)
        models[frozenset([a1, a2, a3, a4])] = pn
    lpms = list(models.values())
    random.shuffle(lpms)
    for i, pn in enumerate(lpms[0:num_lpms]):
        pm4py.write_pnml(*pn, lpm_path+f"LPM_IM_{i}.pnml")


def preprocess(log_path: str, lpm_path: str, out_csv_path: str, alignment_mode: Literal['default', 'lax'] = 'default'):

    log = pm4py.read_xes(log_path)
    log.sort_values(by=[CASE_ID_COL, TIME_COL], inplace=True)

    case_lengths_per_label: Dict[str, List[int]] = {}
    for case_id, case in log.groupby([CASE_ID_COL]):
        label = case[LABEL_COL].unique()[0]
        ls = case_lengths_per_label.get(label, [])
        ls.append(case.shape[0])  # Append case length
        case_lengths_per_label[label] = ls

    labels = list(case_lengths_per_label.keys())
    labels.sort(key=lambda l: len(case_lengths_per_label[l]))
    minority_label = labels[0]

    max_length = min(case_lengths_per_label[minority_label])
    # Cover 90% of all cases with minority label completely
    while len([l for l in case_lengths_per_label[minority_label] if l <= max_length])/len(case_lengths_per_label[minority_label]) < 0.9:
        max_length += 1

    max_prefix_size = min(max_length, 40)
    MIN_PREFIX_SIZE = 2
    print(f"Using prefix sizes of {MIN_PREFIX_SIZE} - {max_prefix_size}")

    def generate_prefix_data(log, min_length, max_length):
        print("Generating Trace Prefixes...")
        dt_prefixes = pd.DataFrame()
        for nr_events in range(min_length, max_length + 1):
            tmp = log.groupby(CASE_ID_COL).filter(
                lambda x: x.shape[0] >= nr_events).groupby(CASE_ID_COL).head(nr_events)
            tmp[BASE_CASE_ID_COL] = tmp[CASE_ID_COL]
            tmp[CASE_ID_COL] = tmp[CASE_ID_COL].apply(
                lambda x: "%s_%s" % (x, nr_events))
            dt_prefixes = pd.concat([dt_prefixes, tmp], axis=0)

        dt_prefixes[CASE_LEN_COL] = dt_prefixes.groupby(
            CASE_ID_COL)[ACTIVITY_COL].transform(len)
        dt_prefixes = dt_prefixes.reset_index(drop=True)

        print("Finished Generating Trace Prefixes.")
        return dt_prefixes.drop([c for c in dt_prefixes.columns if c not in COLS_TO_KEEP], axis=1)
    prefix_log = generate_prefix_data(log, MIN_PREFIX_SIZE, max_prefix_size)

    def custom_alignment(pn: pm4py.objects.petri_net.obj.PetriNet, log):
        model_cost_function = dict()
        sync_cost_function = dict()
        for t in pn[0].transitions:
            if t.label is not None:
                # Visible transition
                model_cost_function[t] = 100000000000000
                sync_cost_function[t] = 0
            else:
                # Silent transition
                model_cost_function[t] = 1
        parameters = {
            AlignmentParameters.SHOW_PROGRESS_BAR: False,
            AlignmentParameters.PARAM_MODEL_COST_FUNCTION: None if alignment_mode == 'lax' else model_cost_function,
            AlignmentParameters.PARAM_SYNC_COST_FUNCTION:  None if alignment_mode == 'lax' else sync_cost_function,
            AlignmentParameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True}
        aligned_traces = align_log(log, *pn, parameters=parameters)
        return aligned_traces

    all_lpm_cols = []
    print("LPM PATH:",lpm_path)
    lpm_files = glob.glob(lpm_path + "/*.pnml")
    no_lpms_found = True
    for lpm_file in lpm_files:
        no_lpms_found = False
        lpm_id = str(id(lpm_file.split("/")[-1]))
        # try:
        #     lpm_id = lpm_file#.split(".")[0].split("_")[-1]
        # except Exception as e:
        #     print("Could not find LPM ID in filename")
        #     print(e)
        print(f"Found lpm: {lpm_file.split('/')[-1]} with id {lpm_id}")
        lpm_col = f"LPM_{lpm_id}"
        all_lpm_cols.append(lpm_col)
        prefix_log[lpm_col] = False
        pn = pm4py.read_pnml(lpm_file)

        lpm_acts = []
        for t in pn[0].transitions:
            if t.label is not None and t.label not in lpm_acts:
                lpm_acts.append(t.label)
        lpm_acts.sort()

        # Only keep activities that are inside the LPM
        prefix_log_filtered = prefix_log[prefix_log[ACTIVITY_COL].isin(
            lpm_acts)]
        # and only keep cases which really contain all the activities of the LPM
        prefix_log_filtered = prefix_log_filtered.groupby(CASE_ID_COL).filter(
            lambda x: len(x[ACTIVITY_COL].unique()) >= len(lpm_acts))

        # Compute alignments
        prefix_log_filtered[lpm_col] = False

        case_ids = prefix_log_filtered[CASE_ID_COL].unique()
        for case_id, case in prefix_log_filtered.groupby([CASE_ID_COL]):
            # case = prefix_log_filtered[prefix_log_filtered[CASE_ID_COL] == case_id]
            case_event_index = case.index
            c_align = custom_alignment(pn, case)
            assert len(c_align) == 1
            if c_align[0] is None or 'alignment' not in c_align[0]:
                print("????")
                print(c_align[0])
                print("?!?!?!")
                continue
            aligned_trace = c_align[0]['alignment']
            aligned_acts = set()
            for al in aligned_trace:
                if ">>" in al[1]:
                    continue
                else:
                    aligned_acts.add(al[1][0])
            event_index = 0
            # Each activity of the LPM was aligned (at least once)
            if alignment_mode == 'default':
                print(len(lpm_acts), len(aligned_acts))
                if len(aligned_acts) >= len(lpm_acts):
                    for al in aligned_trace:
                        if ">>" in al[1]:
                            if None in al[1]:
                                continue
                            else:
                                event_index += 1
                        else:
                            prefix_log.at[case_event_index[event_index],
                                          lpm_col] = True
                            event_index += 1
            elif alignment_mode == 'lax':
                for al in aligned_trace:
                    if al[0][0] == ">>":
                        # Log Skip
                        continue
                    elif al[0][1] == ">>":
                        # Model Skip
                        event_index += 1
                        # if None in al[1]:
                        #     continue
                        # else:
                        #     event_index += 1
                    elif None in al[0]:
                        continue
                    else:
                        prefix_log.at[case_event_index[event_index],
                                      lpm_col] = True
                        event_index += 1

    if no_lpms_found:
        raise Exception("No LPMs found!")
        
    def add_compact_LPM_feature(x):
        return [c for c in all_lpm_cols if x[c]]
    prefix_log['LPMs_list'] = prefix_log.apply(
        lambda x: add_compact_LPM_feature(x), axis=1)
    prefix_log['LPMs_frequency'] = prefix_log.apply(
        lambda x: len(x['LPMs_list']), axis=1)
    prefix_log['LPMs_binary'] = prefix_log.apply(
        lambda x: 1 if x['LPMs_frequency'] >= 1 else 0, axis=1)
    out = prefix_log.drop(all_lpm_cols, axis=1)
    out.to_csv(out_csv_path, index=False, sep=";")
    return max_prefix_size


def split_data(read: pd.DataFrame, input_mode: INPUT_MODE, max_prefix_size: int, train_ratio: float = 0.8, save_ids_to_folder: str = None):
    # Seed for deterministic sklearn train/test split
    np.random.seed(RAND_SEED)

    outcomes = []
    outcomes_dict = {}
    case_ids = read[BASE_CASE_ID_COL].unique()
    tmp_read = read.drop_duplicates([BASE_CASE_ID_COL])
    tmp_read.set_index([BASE_CASE_ID_COL], inplace=True)
    for case_id in case_ids:
        outcome = tmp_read.at[case_id, LABEL_COL]
        outcomes.append(outcome)
        outcomes_dict[case_id] = outcome

    # ID/Indexing of Activities/LPMs
    all_acts = read[ACTIVITY_COL].unique()
    act_ids = {act: i for i, act in enumerate(all_acts)}
    outcome_ids = {outcome: i for i, outcome in enumerate(set(outcomes))}
    all_lpm_lists: Set[FrozenSet[str]] = set()
    for lpm_list in read['LPMs_list']:
        all_lpm_lists.add(frozenset(lpm_list))
    lpm_lists_ids = {lpm: i for i, lpm in enumerate(
        sorted(all_lpm_lists, key=lambda x: len(x)))}
    if save_ids_to_folder is not None:
        timestamp = datetime.datetime.now().isoformat()
        with open(results_path + f"{timestamp}_" + input_mode+"_ids.json", "w") as f:
            json.dump({"timestamp": timestamp,
                      "act_ids": act_ids, "outcome_ids": outcome_ids,"lpm_lists_ids": {str(list(k)): v for k,v in lpm_lists_ids.items()}}, f)
    train, test = train_test_split(
        case_ids, train_size=train_ratio, stratify=outcomes)
    train_prefixes = read[read[BASE_CASE_ID_COL].isin(train.tolist())]
    test_prefixes = read[read[BASE_CASE_ID_COL].isin(test.tolist())]

    # Preprocess token for training/test
    def preprocess_tokens(prefixDF: pd.DataFrame):
        x_tokens = []
        y_tokens = []
        lpm_tokens = []
        for case_id, case in prefixDF.groupby([CASE_ID_COL]):
            base_case_id = case[BASE_CASE_ID_COL].unique()[0]
            trace = case[ACTIVITY_COL].to_list()
            encoded_trace: List[Union[int, List[int]]] = []
            if input_mode.startswith('EMB_'):
                encoded_trace = [act_ids[a] for a in trace]
                x_tokens.append(encoded_trace)
                y_tokens.append(outcome_ids[outcomes_dict[base_case_id]])
                if input_mode == 'EMB_LPM' or input_mode == 'EMB_ACT+LPM':
                    lpm_sequence = [frozenset(l)
                                    for l in case['LPMs_list'].to_list()]
                    encoded_lpm_sequence = [lpm_lists_ids[l]
                                            for l in lpm_sequence]
                    lpm_tokens.append(encoded_lpm_sequence)
            elif input_mode == 'OH_ACT' or input_mode.startswith("WOH"):
                lpm_sequence = [frozenset(l)
                                for l in case['LPMs_list'].to_list()]
                encoded_trace = []
                for i in range(len(trace)):
                    s = 1
                    if input_mode == 'WOH_BIN':
                        if len(lpm_sequence[i]) > 0:
                            s = 2
                    elif input_mode == 'WOH_FREQ':
                        s = len(lpm_sequence[i]) + 1
                    encoded_ev = [s if j == act_ids[trace[i]]
                                  else 0 for j in range(len(act_ids))]
                    encoded_trace.append(encoded_ev)
                x_tokens.append(encoded_trace)
                y_tokens.append(outcome_ids[outcomes_dict[base_case_id]])
            else:
                print(f"Unknown input_mode: {input_mode}")
        print(f"Finished going through all cases in dataset.")
        x_tokens = tf.keras.preprocessing.sequence.pad_sequences(
            x_tokens, maxlen=max_prefix_size)
        lpm_tokens = tf.keras.preprocessing.sequence.pad_sequences(
            lpm_tokens, maxlen=max_prefix_size)

        dtype = np.float32 if input_mode.startswith("EMB_") else np.int64
        x = np.array(x_tokens, dtype=dtype)
        y = np.array(y_tokens, dtype=dtype)
        lpm = np.array(lpm_tokens, dtype=dtype)

        return (x, y, lpm)

    # (test_x, test_y, test_lpm)
    test_data = preprocess_tokens(test_prefixes)
    # (train_x, train_y, train_lpm)
    train_data = preprocess_tokens(train_prefixes)
    return (train_data, test_data)


def train_model(input_mode: INPUT_MODE, model: tf.keras.Model, train_data: Tuple[Any, Any, Any], num_epochs: int, batch_size: int):
    (train_x, train_y, train_lpm) = train_data
    train_input = None
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="auto")
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0,
                                                      mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

    if input_mode == 'OH_ACT' or input_mode == 'WOH_BIN' or input_mode == 'WOH_FREQ':
        train_input = train_x
    elif input_mode == 'EMB_ACT+LPM':
        train_input = [train_x, train_lpm]
    elif input_mode == 'EMB_ACT':
        train_input = train_x
    elif input_mode == 'EMB_LPM':
        train_input = train_lpm
    train_history = model.fit(train_input, train_y, batch_size=batch_size, epochs=num_epochs,
                              validation_split=0.2, callbacks=[lr_reducer, early_stopping])
    return train_history


def test_model(input_mode: INPUT_MODE, model: tf.keras.Model, test_data: Tuple[Any, Any, Any], batch_size: int):
    # Evaluate model on test dataset
    test_input = None
    (test_x, test_y, test_lpm) = test_data
    if input_mode == 'OH_ACT' or input_mode == 'WOH_BIN' or input_mode == 'WOH_FREQ':
        test_input = test_x
        # Just to double check...
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_x, [int(np.round(a)) for a in test_y]))
        test_dataset = test_dataset.batch(64)
        result = model.evaluate(test_dataset)
        print(dict(zip(model.metrics_names, result)))
    elif input_mode == 'EMB_ACT+LPM':
        test_input = [test_x, test_lpm]
    elif input_mode == 'EMB_ACT':
        test_input = test_x
    elif input_mode == 'EMB_LPM':
        test_input = test_lpm
    predicted_y_probs = model.predict(test_input, verbose=0)
    predicted_y_bin = np.array([int(np.round(a)) for a in predicted_y_probs])

    print(f"Accuracy: {accuracy_score(test_y, predicted_y_bin)}")
    w_f1 = f1_score(test_y, predicted_y_bin, average='weighted')
    print(
        f"Weighted F1-Score: {w_f1}")
    w_precision = precision_score(test_y, predicted_y_bin, average='weighted')
    print(
        f"Weighted Precision: {w_precision}")
    w_recall = recall_score(test_y, predicted_y_bin, average='weighted')
    print(
        f"Weighted Recall: {w_recall}")

    print("---")

    predicted_y_probs = predicted_y_probs[:, 0]
    # kappa
    kappa = cohen_kappa_score(test_y, predicted_y_bin)
    print('Cohens kappa: %f' % kappa)
    # ROC AUC
    auc = roc_auc_score(test_y, predicted_y_probs)
    # confusion matrix
    cm = confusion_matrix(test_y, predicted_y_bin)
    # print(cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]

    print('ROC AUC: %f' % auc)
    accuracy = (TP + TN) / (TP + FP + FN + TN)
    print('Accuracy:', accuracy)
    f1score = (TP / (TP + 0.5 * (FP + FN)))
    print('F1-score: ', f1score)
    return {'roc-auc': auc, "accuracy": accuracy,
            "f1": f1score, "kappa": kappa, "w_f1": w_f1, "w_precision": w_precision, "w_recall": w_recall}


def train_and_test_lstm(processed_csv_path: str, input_mode: INPUT_MODE, max_prefix_size: int, num_epochs: int, learning_rate: float, lstm_units: List[int], lstm_dropout_rate: List[float], batch_size: int, act_embedding_size: Optional[int], lpm_embedding_size: Optional[int], results_path: str,  train_ratio=0.8):
    saved_parameters = deepcopy(locals())
    timestamp = datetime.datetime.now().isoformat()

    read = pd.read_csv(processed_csv_path, sep=";")
    read['LPMs_list'] = read['LPMs_list'].apply(lambda x: ast.literal_eval(x))

    # Count activities + lpms
    all_acts = read[ACTIVITY_COL].unique()
    all_lpm_lists: Set[FrozenSet[str]] = set()
    for lpm_list in read['LPMs_list']:
        all_lpm_lists.add(frozenset(lpm_list))

    ((train_x, train_y, train_lpm), (test_x, test_y, test_lpm)) = split_data(read=read,
                                                                             input_mode=input_mode, max_prefix_size=max_prefix_size, train_ratio=train_ratio)

    # Build Model
    model: tf.keras.Model = None

    tf.keras.utils.set_random_seed(RAND_SEED)
    if input_mode.startswith("EMB_"):
        last_input_layer = None
        inputs = None
        if input_mode == 'EMB_ACT+LPM':
            act_input = Input(shape=max_prefix_size, name="act_input")
            act_embedding = Embedding(
                len(all_acts), act_embedding_size, input_length=max_prefix_size, name="act_emb")(act_input)
            lpm_input = Input(shape=max_prefix_size, name="lpm_input")
            lpm_embedding = Embedding(
                len(all_lpm_lists), lpm_embedding_size, input_length=max_prefix_size, name="lmb_emb")(lpm_input)
            concat_layer = Concatenate(name="concat_layer", axis=2)([
                act_embedding, lpm_embedding])
            last_input_layer = concat_layer
            inputs = [act_input, lpm_input]
        elif input_mode == 'EMB_ACT':
            act_input = Input(shape=max_prefix_size, name="act_input")
            act_embedding = Embedding(
                len(all_acts), act_embedding_size, input_length=max_prefix_size, name="act_emb")(act_input)
            last_input_layer = act_embedding
            inputs = act_input
        elif input_mode == 'EMB_LPM':
            lpm_input = Input(shape=max_prefix_size, name="lpm_input")
            lpm_embedding = Embedding(
                len(all_lpm_lists), lpm_embedding_size, input_length=max_prefix_size, name="lmb_emb")(lpm_input)
            last_input_layer = lpm_embedding
            inputs = lpm_input
        last_layer = last_input_layer
        for i in range(len(lstm_units)-1):
            last_layer = LSTM(
                lstm_units[i], dropout=lstm_dropout_rate[i], return_sequences=True)(last_layer)
        lstm = LSTM(lstm_units[-1], dropout=lstm_dropout_rate[-1],
                    return_sequences=False)(last_layer)
        flatten_layer = Flatten()(lstm)
        dense_layer = Dense(1, activation="sigmoid")(flatten_layer)
        model = Model(inputs=inputs, outputs=[dense_layer])
    elif input_mode == 'OH_ACT' or input_mode == 'WOH_BIN' or input_mode == 'WOH_FREQ':
        model = Sequential()
        print("!!!! ", train_x.shape[1], train_x.shape[2])
        lstm = LSTM(lstm_units[0], input_shape=(
            train_x.shape[1], train_x.shape[2]), dropout=lstm_dropout_rate[0], return_sequences=len(lstm_units) > 1)
        model.add(lstm)
        for i in range(1, len(lstm_units)-1):
            lstm = LSTM(
                lstm_units[i], dropout=lstm_dropout_rate[i], return_sequences=True)
            model.add(lstm)
        if len(lstm_units) > 1:
            lstm = LSTM(lstm_units[-1], dropout=lstm_dropout_rate[-1])
        model.add(lstm)
        model.add(Dense(1, activation='sigmoid'))

    print(model.summary())

    # Training
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=[
                  tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), 'acc', tf.keras.metrics.AUC()])

    train_history = train_model(input_mode=input_mode, model=model, train_data=(
        train_x, train_y, train_lpm), num_epochs=num_epochs, batch_size=batch_size)

    # Plot model history
    from matplotlib import pyplot as plt

    def plot_model_output():
        n = len(train_history.history['loss'])
        fig, ax = plt.subplots()
        plt.plot(
            range(n, ), train_history.history['loss'], label='training_loss')
        plt.plot(
            range(n, ), train_history.history['val_loss'], label='validation_loss')
        plt.legend()
        plt.plot(
            range(n, ), train_history.history['acc'], label='training_accuracy')
        plt.plot(
            range(n, ), train_history.history['val_acc'], label='validation_accuracy')
        plt.legend()
        ax.set_ylim(bottom=0.0, top=1.0)
        plt.savefig(results_path + f"{timestamp}_" +
                    input_mode + "_training_plot.svg")
        plt.show()
    plot_model_output()

    test_res = test_model(input_mode=input_mode, model=model, test_data=(
        test_x, test_y, test_lpm), batch_size=batch_size)

    with open(results_path + f"{timestamp}_" + input_mode+"_info.json", "w") as f:
        json.dump({"timestamp": timestamp,
                  "parameters": saved_parameters, "test_res": test_res}, f)
    model.save(results_path + f"{timestamp}_" + input_mode + "_model.keras")


def build_model(input_mode: str, num_acts: int, num_lpms: int, max_prefix_size: int, act_embedding_size: int, lpm_embedding_size: int, lstm_units: list[int], lstm_dropout_rates: list[float], activation_fun: str, learning_rate: float, train_x=None):
    print(f"#Layers: {len(lstm_units)}")
    assert len(lstm_dropout_rates) == len(lstm_units)
    # Build Model
    model: tf.keras.Model = None
    tf.keras.utils.set_random_seed(RAND_SEED)
    if input_mode.startswith("EMB_"):
        last_input_layer = None
        inputs = None
        if input_mode == 'EMB_ACT+LPM':
            act_input = Input(shape=max_prefix_size, name="act_input")
            act_embedding = Embedding(
                num_acts, act_embedding_size, input_length=max_prefix_size, name="act_emb")(act_input)
            lpm_input = Input(shape=max_prefix_size, name="lpm_input")
            lpm_embedding = Embedding(
                num_lpms, lpm_embedding_size, input_length=max_prefix_size, name="lmb_emb")(lpm_input)
            concat_layer = Concatenate(name="concat_layer", axis=2)([
                act_embedding, lpm_embedding])
            last_input_layer = concat_layer
            inputs = [act_input, lpm_input]
        elif input_mode == 'EMB_ACT':
            act_input = Input(shape=max_prefix_size, name="act_input")
            act_embedding = Embedding(
                num_acts, act_embedding_size, input_length=max_prefix_size, name="act_emb")(act_input)
            last_input_layer = act_embedding
            inputs = act_input
        elif input_mode == 'EMB_LPM':
            lpm_input = Input(shape=max_prefix_size, name="lpm_input")
            lpm_embedding = Embedding(
                num_lpms, lpm_embedding_size, input_length=max_prefix_size, name="lmb_emb")(lpm_input)
            last_input_layer = lpm_embedding
            inputs = lpm_input
        last_layer = last_input_layer
        for i in range(len(lstm_units)-1):
            last_layer = LSTM(
                lstm_units[i], dropout=lstm_dropout_rates[i], return_sequences=True)(last_layer)
        lstm = LSTM(lstm_units[-1], dropout=lstm_dropout_rates[-1],
                    return_sequences=False)(last_layer)
        flatten_layer = Flatten()(lstm)
        dense_layer = Dense(1, activation="sigmoid")(flatten_layer)
        model = Model(inputs=inputs, outputs=[dense_layer])
    elif input_mode == 'OH_ACT' or input_mode == 'WOH_BIN' or input_mode == 'WOH_FREQ':
        print(lstm_dropout_rates, lstm_units, num_acts, max_prefix_size)
        model = Sequential()
        lstm = LSTM(lstm_units[0], input_shape=(
            max_prefix_size, num_acts), dropout=lstm_dropout_rates[0], return_sequences=len(lstm_units) > 1)
        model.add(lstm)
        for i in range(1, len(lstm_units)-1):
            lstm = LSTM(
                lstm_units[i], dropout=lstm_dropout_rates[i], return_sequences=True)
            model.add(lstm)
        if len(lstm_units) > 1:
            lstm = LSTM(lstm_units[-1], dropout=lstm_dropout_rates[-1])
            model.add(lstm)
        model.add(Dense(1, activation=activation_fun))
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def hp_search(processed_csv_path: str, input_mode: INPUT_MODE, max_prefix_size: int, results_path: str, max_trials: int, train_ratio=0.8):
    epochs = 100
    batch_size = 32
    read = pd.read_csv(processed_csv_path, sep=";")
    read['LPMs_list'] = read['LPMs_list'].apply(lambda x: ast.literal_eval(x))

    # Count activities + lpms
    all_acts = read[ACTIVITY_COL].unique()
    all_lpm_lists: Set[FrozenSet[str]] = set()
    for lpm_list in read['LPMs_list']:
        all_lpm_lists.add(frozenset(lpm_list))

    ((train_x, train_y, train_lpm), (test_x, test_y, test_lpm)) = split_data(read=read,
                                                                             input_mode=input_mode, max_prefix_size=max_prefix_size, train_ratio=train_ratio,save_ids_to_folder=results_path)

    def build_hp_model(hp: keras_tuner.HyperParameters):
        units = []
        dropout_rates = []
        for i in range(hp.Int("layers", 1, 4, default=1)):
            units.append(hp.Int(f"units_{i}", 5, 200, step=10))
            dropout_rates.append(
                hp.Float(f"dropoutrate_{i}", 0.0, 0.8, step=0.1))
        activation_fun = hp.Choice(
            "activation", ["relu", "tanh", "sigmoid"], default="sigmoid")
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-5, max_value=5e-2, sampling="log")
        act_embedding_size = None
        lpm_embedding_size = None
        if input_mode.startswith("EMB_") and "ACT" in input_mode:
            act_embedding_size = hp.Int(
                "act_embedding_size", min_value=8, max_value=128, step=12)
        if input_mode.startswith("EMB_") and "LPM" in input_mode:
            lpm_embedding_size = hp.Int(
                "lpm_embedding_size", min_value=8, max_value=128, step=12)
        model = build_model(
            input_mode=input_mode,
            num_acts=len(all_acts),
            num_lpms=len(all_lpm_lists),
            max_prefix_size=max_prefix_size,
            act_embedding_size=act_embedding_size,
            lpm_embedding_size=lpm_embedding_size,
            lstm_units=units,
            lstm_dropout_rates=dropout_rates,
            activation_fun=activation_fun,
            learning_rate=learning_rate,
        )
        return model
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="auto")
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0,
                                                      mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
    
    timestamp = datetime.datetime.now().isoformat()
    tuner = keras_tuner.BayesianOptimization(hypermodel=build_hp_model, max_trials=max_trials, objective="val_loss",
                                             overwrite=True,
                                             directory="tuner"+timestamp,
                                             project_name="tuner_proj"+timestamp, seed=RAND_SEED)
    print(tuner.search_space_summary())
    train_data = []
    if input_mode == 'OH_ACT' or input_mode == 'WOH_BIN' or input_mode == 'WOH_FREQ':
        train_data = train_x
    elif input_mode == "EMB_ACT":
        train_data = train_x
    elif input_mode == "EMB_LPM":
        train_data = train_lpm
    elif input_mode == "EMB_ACT+LPM":
        train_data = [train_x, train_lpm]

    tuner.search(x=train_data, y=train_y, validation_split=0.2, batch_size=batch_size,
                 callbacks=[lr_reducer, early_stopping], epochs=epochs)
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters:")
    print(best_hyperparameters.values)
    best_model = tuner.get_best_models()[0]

    train_history = train_model(input_mode=input_mode, model=best_model, train_data=(
        train_x, train_y, train_lpm), num_epochs=epochs, batch_size=batch_size)
    test_res = test_model(input_mode=input_mode, model=best_model, test_data=(
        test_x, test_y, test_lpm), batch_size=batch_size)

    create_dir_if_needed(results_path)
    timestamp = datetime.datetime.now().isoformat()
    results_path
    with open(results_path + f"{timestamp}_" + input_mode+"_info.json", "w") as f:
        json.dump({"timestamp": timestamp,
                  "parameters": best_hyperparameters.values, "test_res": test_res, "RAND_SEED": RAND_SEED}, f)
    best_model.save(
        results_path + f"{timestamp}_" + input_mode + "_model.keras")


def create_dir_if_needed(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                    prog='AllInOne')
    parser.add_argument('--log', default=None) 
    parser.add_argument('--folder')
    parser.add_argument('--repeats', default="0,1,2,3,4")
    parser.add_argument('--discoverLPMs',action='store_true',default=False)
    parser.add_argument('--skipWithMaxPrefix',default=None)
    parser.add_argument('--skipPreprocessing',action='store_true',default=False)
    parser.add_argument('--maxTrials', default=50)
    args = parser.parse_args()
    discover_lpms = args.discoverLPMs
    repeats = [int(i.strip()) for i in args.repeats.split(",")]
    print(f"Executing runs {repeats} for log {args.log} with max. trials of {int(args.maxTrials)}")
    
    abs_path = "."
    CSV_LOGS_DIR = f"{abs_path}/data/main/logs/"
    XES_LOG_DIR = f"{abs_path}/data/main/xes_logs/"
    LPMS_DIR = f"{abs_path}/{args.folder}/data/lpms/"
    CSV_PROCESSED_DIR = f"{abs_path}/{args.folder}/data/processed/"
    RES_DIR = f"{abs_path}/{args.folder}/data/results/"
    for p in [LPMS_DIR, CSV_PROCESSED_DIR, XES_LOG_DIR, RES_DIR]:
        create_dir_if_needed(p)
        if args.log is None:
            LOG_NAMES = [
                # "Production",
                # "BPIC11_f1",
                "BPIC11_f2",
                # "BPIC11_f3",
                # "BPIC11_f4",
                # "traffic_fines_1".
                # # "BPIC15_1_f2",
                # # "BPIC15_3_f2",
                # # "BPIC15_2_f2",
                # # "BPIC15_5_f2",
                # # "BPIC15_4_f2",
                # # "hospital_billing_3",
                # # "hospital_billing_2",
                # "bpic2012_O_CANCELLED-COMPLETE",
                # "bpic2012_O_ACCEPTED-COMPLETE",
                # "bpic2012_O_DECLINED-COMPLETE",
                # # "BPIC17_O_Refused",
                # # "BPIC17_O_Accepted",
                # # "BPIC17_O_Cancelled",
                # # "sepsis_cases_2",
                # # "sepsis_cases_1",
                # # "sepsis_cases_4",
            ]
        else:
            LOG_NAMES = [args.log]
    for log_name in LOG_NAMES:
        xes_log_path = XES_LOG_DIR + f"{log_name}.xes"
        lpms_path = LPMS_DIR + log_name + "/"
        create_dir_if_needed(lpms_path)

        processed_csv_path = CSV_PROCESSED_DIR+log_name+"/processed.csv"
        create_dir_if_needed(CSV_PROCESSED_DIR+log_name)

        if args.skipWithMaxPrefix is None and not args.skipPreprocessing:
            convert_csv_log_to_xes(CSV_LOGS_DIR + log_name +
                               ".csv", out_xes_path=xes_log_path)
        if discover_lpms:
            if args.skipPreprocessing:
                raise Exception("Combining discoverLPMs with skipping Preprocessing does not make sense!")
            discover_lpms_inductive(xes_log_path=xes_log_path,
                                    lpm_path=lpms_path, num_lpms=100)

        results_path = RES_DIR + log_name + "/"
        create_dir_if_needed(results_path)
        if args.skipWithMaxPrefix is None and not args.skipPreprocessing:
            max_prefix_size = preprocess(
                log_path=xes_log_path, lpm_path=LPMS_DIR+log_name, out_csv_path=processed_csv_path)
        elif args.skipWithMaxPrefix is not None:   
            max_prefix_size = int(args.skipWithMaxPrefix)
        elif args.skipPreprocessing:
            # Infer max prefix size based on processed csv
            read = pd.read_csv(processed_csv_path, sep=";")
            max_prefix_size = int(read[CASE_ID_COL].value_counts().max())
            print(f"Inferred max prefix size of {max_prefix_size}")
        print(f"Max prefix size of {max_prefix_size} selected. {type(max_prefix_size)}")

        for i in repeats:#range(5): # 0,1,2,3,4
            RAND_SEED = 2023 * i
            for input_mode in INPUT_MODES:
                print(f"{i+1}/5 run for input mode {input_mode}")
                hp_search(processed_csv_path=processed_csv_path, input_mode=input_mode,
                      max_prefix_size=max_prefix_size, results_path=results_path, max_trials=int(args.maxTrials))

            # train_and_test_lstm(processed_csv_path=processed_csv_path, input_mode=input_mode,
            #                     max_prefix_size=max_prefix_size, num_epochs=100, learning_rate=0.00008,
            #                     lstm_units=[50, 25], lstm_dropout_rate=[0.15, 0.1], batch_size=16,
            #                     act_embedding_size=24, lpm_embedding_size=16, train_ratio=0.8, results_path=results_path)
