import json
from Knapsack_Environment_Class import problem_size, dataset, max_value, n_problems, min_capacity, max_capacity, path
import numpy as np
import pandas as pd
import random
import pprint
import math


def select_action(q, current_state, allowed_actions, epsilon):
    r = random.random()
    if r < epsilon:
        a = allowed_actions[random.randint(0, len(allowed_actions) - 1)]
    else:
        tmp = q.loc[current_state, :]
        tmp = list(tmp)
        tmp = np.argmax(tmp)
        a = allowed_actions[tmp]
    return a


num_threshold_list = [1, 2, 3, 4, 5]


def get_allowed_actions():
    max_thresholds = 5
    #allowed_actions = ['original']
    allowed_actions = []
    for i in range(max_thresholds):
        allowed_actions.append('threshold_' + str(i + 1))
    return allowed_actions


def unique_values(features):
    num_of_uniques = features.nunique()
    return num_of_uniques


def state_space_size(s, a, num_of_uniques):
    if a == 'original':
        return num_of_uniques[s]
    b = a.split('_')
    return int(b[1]) + 1


def get_original_reward(row):
    return row['normalized_' + str(row['action'])]


def ranges_of_thresholds(features):
    features = pd.DataFrame(features)
    threshold_list = dict()
    for col in features.columns:
        sorted_f = features[col].dropna()
        sorted_f = sorted_f.sort_values()
        threshold_list[col] = {}
        threshold_list[col]['total_not_nan_values'] = len(sorted_f)
        threshold_list[col]['ranges'] = dict()
        for t in num_threshold_list:
            prev = 0
            if t not in threshold_list[col]:
                threshold_list[col][str(t)] = list()
                threshold_list[col]['ranges'][str(t)] = list()
            for i in range(t):
                # in/(i+1). for example for two thresholds the indices are 1/3 and 2/3
                index = (i + 1) * len(sorted_f) / (t + 1)
                index = int(index)
                threshold_list[col][str(t)].append({'threshold_index': index, 'threshold': float(sorted_f.values[index])})
                threshold_list[col]['ranges'][str(t)].append(float(sorted_f.values[index]) - prev)
                prev = float(sorted_f.values[index])
            threshold_list[col]['ranges'][str(t)].append(float(sorted_f.values[len(sorted_f) - 1]) - prev)
    json.dump(threshold_list, open(path + 'threshold_list_' + str(problem_size), 'w'), indent=4)
    return threshold_list


def commons_in_two_ranges(features):
    features = pd.DataFrame(features)
    commons_in = dict()
    for col in features.columns:
        sorted_f = features[col].dropna()
        sorted_f = sorted_f.sort_values()
        commons_in[col] = {}
        for t in num_threshold_list:
            if t not in commons_in[col]:
                commons_in[col][str(t)] = dict()
            for i in range(t):
                # in/(i+1). for example for two thresholds the indices are 1/3 and 2/3
                index = (i + 1) * len(sorted_f) / (t + 1)
                index = int(index)
                tmp = index + 1
                while tmp < len(sorted_f.values) and sorted_f.values[tmp] == sorted_f.values[index]:
                    tmp += 1
                common = tmp - index - 1
                commons_in[col][str(t)]['common_' + str(i) + '_' + str(i+1)] = common
    json.dump(commons_in, open(path + 'commons_in_two_ranges_' + str(problem_size), 'w'), indent=4)
    return commons_in


def normalize_data(feature_data, max_d):
    output = feature_data / max_d
    output = 2 * output - 1
    return output


def reward_function(state, action, value_decoder, threshold_list, commons_in_, num_of_uniques, size_so_far):
    tmp_a = action.split('_')
    ranges_list = threshold_list[state]['ranges'][tmp_a[1]]
    ranges_mult = np.prod(ranges_list)
    common_list = commons_in_[state][tmp_a[1]]
    commons_sum = np.sum(list(common_list.values()))
    size = state_space_size(state, action, num_of_uniques)
    if commons_sum == 0:
        commons_sum = 0.01
    r = ranges_mult / (size * commons_sum)
    return r, (size_so_far * size)


def check_convergence(q, previous_q, diff):
    sum_ = 0
    output = True
    for i in q.index:
        for j in q.columns:
            sum_ = sum_ + abs(q.loc[i, j] - previous_q.loc[i, j])
            if abs(q.loc[i, j] - previous_q.loc[i, j]) > diff:
                output = False
    print(sum_)
    return output


def q_learning(features, allowed_actions, uniques, value_decoder, threshold_list, commons_in):
    epsilon = 1
    decay = 0.9
    alfa = 0.9
    gamma = 0.3
    diff = 0.0001
    states = list(features.columns)
    q = pd.DataFrame(index=states, columns=allowed_actions)
    q = q.fillna(0)
    converge = False
    while not converge:
        i = 0
        s = states[i]
        end_of_episode = False
        previous_q = q.copy()
        size = 1
        while not end_of_episode:
            a = select_action(q, s, allowed_actions, epsilon)
            r, size = reward_function(s, a, value_decoder, threshold_list, commons_in, uniques, size)
            if i + 1 < len(states):
                s_prime = states[i + 1]
                # tmp = np.max(q.loc[s_prime, :])
                # tmp = tmp * gamma
                # tmp = tmp + r
                # tmp = tmp - q.loc[s, a]
                # tmp = tmp * alfa
                # q.loc[s, a] = q.loc[s, a] + tmp
                q.loc[s, a] = q.loc[s, a] + alfa * (r + (gamma * np.max(q.loc[s_prime, :])) - q.loc[s, a])
                s = s_prime
            else:
                q.loc[s, a] = q.loc[s, a] + alfa * (r - q.loc[s, a])
                end_of_episode = True
            i = i + 1
        q.to_csv(path + 'tmp_q_1st')
        converge = check_convergence(q, previous_q, diff)
        alfa = alfa * decay
        epsilon = epsilon * decay
    return q


def best_decision(q, features, threshold_list):
    decision = dict()
    for i in q.index:
        decision[i] = q.columns[np.argmax(np.array(q.loc[i, :]))]
    states = dict()
    for k, v in decision.items():
        states[k] = v
        tmp = v.split('_')
        co = 1
        for thr in threshold_list[k][tmp[1]]:
            states[k + '_threshold_' + str(co)] = thr['threshold']
            co += 1

        ############
        if str(k).startswith('w_ratio'):
            states[k] = 'threshold_2'
            states[k + '_threshold_' + str(1)] = 0.5
            states[k + '_threshold_' + str(2)] = 1
        ############
    json.dump(states, open(path + 'knapsack_' + dataset + 'states' + str(problem_size), 'w'), indent=4)
    return decision


def get_second_max(ar):
    max1 = 0
    max2 = 0
    max2_i = 0
    for i in range(len(ar.index)):
        if ar[ar.index[i]] > max1:
            max1 = ar[ar.index[i]]
        elif ar[ar.index[i]] > max2:
            max2 = ar[ar.index[i]]
            max2_i = i
    return max2_i


def second_best_decision(q, features):
    decision2 = dict()
    for i in q.index:
        decision1_index = np.argmax(np.array(q.loc[i, :]))
        second_max_index = get_second_max(q.loc[i, :])
        decision2_index = decision1_index
        rn = random.random()
        if rn > 0.5:
            decision2_index = second_max_index
        decision2[i] = q.columns[decision2_index]
    states = dict()
    for k, v in decision2.items():
        states[k] = v
        if v == 'binary':
            median = np.median(features[k])
            states[k + '_threshold'] = median
        if v == 'ternary':
            tmp = np.sort(features[k])
            threshold1 = tmp[int(len(features[k]) / 3)]
            threshold2 = tmp[int(2 * len(features[k]) / 3)]
            states[k + '_threshold1'] = str(threshold1)
            states[k + '_threshold2'] = str(threshold2)
    json.dump(states, open(path + 'Decisions\\RTB_states_second_best.json', 'w'), indent=4)
    return decision2


def random_decision(q, features, allowed_actions):
    decision3 = dict()
    for i in q.index:
        if np.sum(q.loc[i, :]) == 0:
            decision3[i] = 0
        else:
            random_decision_index = random.randint(0, len(allowed_actions[i]) - 1)
            while q.loc[i, q.columns[random_decision_index]] == 0:
                random_decision_index = random.randint(0, len(allowed_actions[i]) - 1)
            decision3[i] = q.columns[random_decision_index]
    states = dict()
    for k, v in decision3.items():
        states[k] = v
        if v == 'binary':
            median = np.median(features[k])
            states[k + '_threshold'] = median
        if v == 'ternary':
            tmp = np.sort(features[k])
            threshold1 = tmp[int(len(features[k]) / 3)]
            threshold2 = tmp[int(2 * len(features[k]) / 3)]
            states[k + '_threshold1'] = str(threshold1)
            states[k + '_threshold2'] = str(threshold2)
    json.dump(states, open(path + 'Decisions\\RTB_states_random_decision.json', 'w'), indent=4)
    return decision3


def driver(feature_file):
    features = pd.read_csv(path + feature_file)
    # features = pd.read_csv(path + feature_file, nrows=100000)
    allowed_actions = []
    for d in num_threshold_list:
        allowed_actions.append('threshold_' + str(d))

    value_decoder = json.load(open(path + 'value_decoder' + str(problem_size)))
    num_of_uniques = unique_values(features)
    # threshold_list = json.load(open(path + 'threshold_list_' + str(problem_size), 'r'))
    threshold_list = ranges_of_thresholds(features)
    commons_in_two_ranges(features)
    commons_in_ = json.load(open(path + 'commons_in_two_ranges_' + str(problem_size), 'r'))
    q = q_learning(features, allowed_actions, num_of_uniques, value_decoder, threshold_list, commons_in_)
    print(q)
    decision1 = best_decision(q, features, threshold_list)
    pprint.pprint(decision1)
    # decision2 = second_best_decision(q, features)
    # pprint.pprint(decision2)
    # decision3 = random_decision(q, features, allowed_actions)
    # pprint.pprint(decision3)


def test():
    # features = pd.read_csv(path + 'episodes_raw_problems' + str(problem_size) + '_df')
    features = pd.read_csv(path + 'episodes_raw_problems' + str(problem_size) + '_df')
    # ranges_of_thresholds(features)
    # commons_in_two_ranges(features)
    actions = ['ignore', 'original', 'binary', 'ternary']
    # tmp = 'user_id'
    # print(features[tmp].nunique())
    # num_of_uniques = unique_values(features)
    # print(features.nunique())


# feature_importance_calculator('one_hot_version_2018.csv')
# test()
driver('episodes_' + dataset + 'problem_instances_' + str(problem_size) + '_' + str(n_problems) + '_df')
# feature_importance_calculator()
# convert_json_to_csv_features(path + 'new_balanced_aggregated_cleaned_26Nov.json')

