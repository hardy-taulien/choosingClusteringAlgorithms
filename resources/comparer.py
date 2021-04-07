################### comparer algorithm ###################
# takes data and calculates meta information to help de- #
# cide on which clustering algorithm to use.             #
##########################################################
# imports
import pandas as pd
import math
import numpy as np

############## default values for algorithms #############
# define default values and conditions for clustering
# algorithms to be chosen when met
# default values are marked with "_default"
# accepted values may lie within "_acceptance" reach of
# the default value
ca_properties = {
    "fuzz": {
        "dist_default": 0.6,
        "dist_acceptance": 0.39,
        "dev_default": 0.,
        "dev_acceptance": 0.2,
        "rep_default": 0.5,
        "rep_acceptance": 0.499,
        "equi_default": 0.5,
        "equi_acceptance": 0.45
    },
    "dbscan": {
        "dist_default": 0.5,
        "dist_acceptance": 0.45,
        "dev_default": 0.,
        "dev_acceptance": 0.75,
        "rep_default": 0.5,
        "rep_acceptance": 0.499,
        "equi_default": 0.,
        "equi_acceptance": 0.9
    },
    "aggl": {
        "dist_default": 0.5,
        "dist_acceptance": 0.5,
        "dev_default": 0.,
        "dev_acceptance": 1.,
        "rep_default": 0.5,
        "rep_acceptance": 0.49,
        "equi_default": 0.,
        "equi_acceptance": 1.
    },
    "gmm": {
        "dist_default": 0.9,
        "dist_acceptance": 0.09,
        "dev_default": 0.,
        "dev_acceptance": 1.,
        "rep_default": 0.5,
        "rep_acceptance": 0.49,
        "equi_default": 0.,
        "equi_acceptance": 1.
    }
}


################### function definition ##################
# identify sparse spots in the distribution
# by checking the deviation vicinity of elements and checking for holes in the value range
# pass along confidentiality of how accurate it may be
# @returns fd=final min distance used to find holes, n=number of distinct regions, p=confidentiality in percent
def sparseDetection(df, x, y, _n=1, _m=1):
    # initialize variables
    c_x = c_y = 0  # number of sparse spots for x and y, iteration counter
    # multipliers for deviation ranges
    n = _n
    m = _m
    # standard deviations
    xdev = math.sqrt(df[x].var())
    ydev = math.sqrt(df[y].var())
    # average covered area of deviation
    x_coverage = xdev / (df[x].max() - df[x].min())
    y_coverage = ydev / (df[y].max() - df[y].min())
    # average density as threshold for how many values need to be found
    h_x = len(df.index) * (1 + x_coverage) / (df[x].max() - df[x].min())
    h_y = len(df.index) * (1 + y_coverage) / (df[y].max() - df[y].min())

    # continue until any counter leaves threshold
    while -3 <= c_x <= 3 and -3 <= c_y <= 3:
        # lists to keep the values that fulfil the requirements
        l_x = []
        l_y = []
        # iterate over all elements
        for i in df.index:
            # determine the valid range
            min_xi = df[x][i] - n * xdev
            max_xi = df[x][i] + n * xdev
            min_yi = df[y][i] - m * ydev
            max_yi = df[y][i] + m * ydev
            range_x = [min_xi, max_xi]
            range_y = [min_yi, max_yi]
            # check the vicinity of the current x for enough neighbours
            if int(df.groupby(pd.cut(df[x], range_x)).count()[x]) >= h_x:
                # make addition to the ranges list
                l_x = sparseIteratorHelper(l_x, (min_xi, max_xi), df[x][i])
                l_x_control = l_x
                # check for overlap that might have occurred
                l_x_test = overlapHelper(l_x_control)
                # loop until the function returns the same list we hand it
                while l_x_test.sort() != l_x_control.sort():
                    l_x_control = l_x_test
                    l_x_test = overlapHelper(l_x_test)

            if int(df.groupby(pd.cut(df[y], range_y)).count()[y]) >= h_y:
                # send to iterator helper function
                l_y = sparseIteratorHelper(l_y, (min_yi, max_yi), df[y][i])
                l_y_control = l_y
                # check for overlapping ranges
                l_y_test = overlapHelper(l_y_control)
                # loop until no overlaps remain
                while l_y_test.sort() != l_y_control.sort():
                    l_y_control = l_y_test
                    l_y_test = overlapHelper(l_y_test)

            # before continuing, check if the range was too large and we already have the entire value range as l_x or l_y
            if len(l_x) >= 1 and len(l_y) >= 1:
                if l_x[0][0] <= df[x].min() and l_x[0][1] >= df[x].max() or l_y[0][0] <= df[y].min() and l_y[0][1] >= df[y].max():
                    break
        # after iteration: evaluate and prepare next iteration:
        # count holes = length of list with found ranges - 1
        s_x = len(l_x) - 1
        s_y = len(l_y) - 1

        # too wide range, reduce n
        if s_x < 2:
            n /= (2 + abs(c_x))
            c_x -= 1
        # too narrow range, increase n
        elif s_x > 5:
            n *= (2 + abs(c_x))
            c_x += 1
        # terminating condition
        else:
            return s_x, s_y, l_x, l_y, n, m

        # too wide range, reduce m
        if s_y < 2:
            m /= (2 + abs(c_y))
            c_y -= 1
        # too narrow range, increase m
        elif s_y > 5:
            m *= (2 + abs(c_y))
            c_y += 1
        # terminating condition
        else:
            return s_x, s_y, l_x, l_y, n, m
    s_x = s_y = 0
    l_x = [(df[x].min(), df[x].max())]
    l_y = [(df[y].min(), df[y].max())]
    # default return when c threshold is exceeded
    return s_x, s_y, l_x, l_y, n, m

# function to help with the iteration
def sparseIteratorHelper(old_list, pair, ref):
    new_list = old_list
    if pair in old_list:
        return old_list
    # add current pair to list if empty
    if not new_list:
        new_list.append(pair)
    for old_pair in old_list:
        # pair is within known range
        if old_pair[0] < pair[0] < ref < pair[1] < old_pair[1]:
            continue
        # pair has a wider range to the left
        elif pair[0] < old_pair[0] < ref < pair[1] < old_pair[1]:
            new_list[old_list.index(old_pair)] = (pair[0], old_pair[1])
        # pair has a wider range to the right
        elif old_pair[0] < old_pair[1] < ref < old_pair[1] < pair[1]:
            new_list[old_list.index(old_pair)] = (old_pair[0], pair[1])
        # pair is unknown to the list
        elif pair[1] < old_pair[0] or pair[0] > old_pair[1]:
            new_list.append(pair)
    # return, but remove repeating tuples as they do not reliably get filtered out yet
    return list(set(new_list))

# helper to find overlapping intervals
# returns the list with first instance of overlapping values merged or no changes if there is no overlap
def overlapHelper(tpl_list):
    # get a list of pairs that have overlapping intervals
    lows = [e[0] for e in tpl_list]
    highs = [e[1] for e in tpl_list]

    for i in range(len(lows)):
        for j in range(len(highs)):
            # do not compare an interval with itself -> skip if i == j
            if i == j:
                continue
            if highs[i] > highs[j] > lows[i] > lows[j]:
                new_pair = (lows[j], highs[i])
                tpl_list.remove((lows[j], highs[j]))  # old overlapping pair
                tpl_list.remove((lows[i], highs[i]))  # old overlapping pair #2
                tpl_list.append(new_pair)  # new pair, merge of the overlap
                # not safe to modify anything anymore since we removed and added elements, return
                # break would work as well but because of the nested loop it is easier to implement it like this
                return tpl_list
            elif highs[i] >= lows[j] >= lows[i] and lows[i] <= highs[j] <= highs[i]:
                # remove in betweens (sometimes an interval that lies within another is overlooked)
                # j range lies within i range
                tpl_list.remove((lows[j], highs[j]))
                # again, it's not safe to continue with a modified list, so return
                return tpl_list
    # if nothing has been modified, return the same list that was given as parameter to signal that there is no overlap
    return tpl_list


# calculate a score for the values that lay within sigma range of the mean of each parameter
# This gives an insight on how evenly spread the data is
# @returns percentages of values left and right of the mean
def distributionScore(df, x, y, b_x=1, b_y=1):
    # calculate deviations
    xdev = math.sqrt(df[x].var())
    ydev = math.sqrt(df[y].var())
    xmed = df[x].median()
    ymed = df[y].median()
    # count elements in sigma-range, left and right
    xsl = 0
    xsr = 0
    ysl = 0
    ysr = 0
    for i in df.index:
        if df[x][i] > (xmed - b_x * xdev):
            xsl += 1
        elif df[x][i] < (xmed + b_x * xdev):
            xsr += 1

        if df[y][i] > (ymed - b_x * ydev):
            ysl += 1
        elif df[y][i] < (ymed + b_y * ydev):
            ysr += 1
    # percentage of values in each section
    idx_len = len(df.index)
    xr_perc = xsr / idx_len
    xl_perc = xsl / idx_len
    yr_perc = ysr / idx_len
    yl_perc = ysl / idx_len

    return xr_perc, xl_perc, yr_perc, yl_perc


# calculate a score for the value ranges under consideration
# of standard deviations
# This gives an insight into how dense/sparse the data is distributed along the axes
def deviationScore(df, x, y):
    xvalRange = df[x].max() - df[x].min()
    yvalRange = df[y].max() - df[y].min()
    # standard deviation
    xdev = math.sqrt(df[x].var())
    ydev = math.sqrt(df[y].var())

    # distribution indicator - how spread is the data across the axis
    xdevi = xdev / xvalRange
    ydevi = ydev / yvalRange

    # get a representative value out of the partial values
    # positive value: high scatter of y, little/no scatter of x and vice versa
    devi = 0 + xdevi - ydevi

    # The closer the value for devi is to 0, the fewer we know about the scattering. We return the partial values
    # as well as the final deviation indicator to be able to draw conclusions
    return xdevi, ydevi, devi


# calculate a score for the position of the medians and the means in the
# range of values for the parameters
# @returns normalized coordinates of medians (-1, 1)
def equilibriumScore(df, x, y):
    xmed = df[x].median()
    xminVal = df[x].min()
    xmaxVal = df[x].max()
    ymed = df[y].median()
    yminVal = df[y].min()
    ymaxVal = df[y].max()

    # position of median scaled to (-1,1)
    xpos = (xmed - xminVal) / (xmaxVal - xminVal)
    xpos = xpos * 2 - 1
    ypos = (ymed - yminVal) / (ymaxVal - yminVal)
    ypos = ypos * 2 - 1

    return xpos, ypos


# a score to determine the density of narrow areas
# evaluates how often values are repeated
def repetitionScore(df, x, y):
    # define repetition counters for both axes
    x_rep = 0
    y_rep = 0

    # lists to keep track of visited values
    x_visited = []
    y_visited = []

    # iterate over the data set and count repetitions
    for row in df.itertuples():
        if getattr(row, x) not in x_visited:
            x_visited.append(getattr(row, x))
        else:
            x_rep += 1

        if getattr(row, y) not in y_visited:
            y_visited.append(getattr(row, y))
        else:
            y_rep += 1

    # calculate scores for x and y
    s_xrep = x_rep / len(df[x])
    s_yrep = y_rep / len(df[y])

    return s_xrep, s_yrep


################### choosing an algorithm ##################
# checks scores and gives points based on how well the distribution fits the default values
def chooseAlgorithm(df, x, y, n):
    # each algorithm gets a rating
    fuzz = aggl = gmm = dbscan = 0

    # get scores that don't need further input
    rep_scores = repetitionScore(df, x, y)
    equi_scores = equilibriumScore(df, x, y)
    dev_scores = deviationScore(df, x, y)
    # because of complexity, give a subset to sparseDetection
    sparse_results = sparseDetection(df, x, y)

    # each hit on the overall vicinity gives a point to the rating.
    # each hit on the close vicinity (1/4th acceptance level) around the expected value gives another point to the rating

    # get the scores
    rep_ratings = scoreEval(rep_scores, "rep", [0, 1])
    equi_ratings = scoreEval(equi_scores, "equi", [0, 1])
    dev_ratings = scoreEval(dev_scores, "dev", [2])

    # choose b based on deviation score, run distribution score and sparse spot detection using these results
    _b_x = dev_scores[0]
    _b_y = dev_scores[1]
    dist_results = distributionScore(df, x, y, b_x=_b_x, b_y=_b_y)

    # ratings for distribution score
    dist_scores_combined = [dist_results[0] + dist_results[1], dist_results[2] + dist_results[3]]
    dist_ratings = scoreEval(dist_scores_combined, "dist", [0, 1])

    # check if number of clusters given matches found holes on any axis first
    # since we have the number of holes, we assume number of dense areas =  s + 1
    # acceptance level: 1
    n_hit = 1 if -1 <= sparse_results[0] - (n + 1) <= 1 or -1 <= sparse_results[1] - (n + 1) <= 1 else 0
    n_hit += 1 if sparse_results[0] - (n + 1) == 0 or sparse_results[1] - (n + 1) == 0 else 0

    # first round of rating check to see if dbscan is an option
    dbscan += rep_ratings["dbscan"] + equi_ratings["dbscan"] + dev_ratings["dbscan"] + dist_ratings["dbscan"]
    aggl += rep_ratings["aggl"] + equi_ratings["aggl"] + dev_ratings["aggl"] + dist_ratings["aggl"]
    fuzz += rep_ratings["fuzz"] + equi_ratings["fuzz"] + dev_ratings["fuzz"] + dist_ratings["fuzz"]
    gmm += rep_ratings["gmm"] + equi_ratings["gmm"] + dev_ratings["gmm"] + dist_ratings["gmm"]
    # algorithms that don't use n profit from n not being accurate and vice versa
    if n_hit == 1:
        aggl += 1
        gmm += 1
        fuzz += 1
    # give an additional points to all except aggl if direct hit
    elif n_hit == 2:
        gmm += 1
        fuzz += 1
    else:
        dbscan += 3

    # if length of intervals found in sparse detection is not too varied, fuzzy c-means profits from the score
    # since the other algorithms do not assume globular clusters, they are not affected
    covered_x = []
    covered_y = []
    for pair in sparse_results[2]:
        covered_x.append(abs(pair[1] - pair[0]))
    for pair in sparse_results[3]:
        covered_y.append(abs(pair[1] - pair[0]))
    l_x_dev = np.std(covered_x)
    l_y_dev = np.std(covered_y)
    # we will consider the deviation too large if it is higher than the length of any interval
    interval_control = True
    for ivl in covered_x:
        if l_x_dev > ivl:
            interval_control = False
    for ivl in covered_y:
        if l_y_dev > ivl:
            interval_control = False
    if interval_control:
        fuzz += 1


    # find a candidate based on the ratings
    # list of ratings
    ratings = [fuzz, dbscan, aggl, gmm]
    high_rating = [i for i, x in enumerate(ratings) if x == max(ratings)]
    # for dbscan, will be set later. otherwise 0 because irrelevant
    eps = min_pts = 0
    if len(high_rating) == 1:
        if high_rating[0] == 1:
            # reuse h from sparse detection
            xdev = math.sqrt(df[x].var())
            ydev = math.sqrt(df[y].var())
            x_coverage = xdev / (df[x].max() - df[x].min())
            y_coverage = ydev / (df[y].max() - df[y].min())
            # average density as threshold for how many values need to be found - used as min_pts if dbscan is used
            min_p_x = len(df.index) * x_coverage / (df[x].max() - df[x].min())
            min_p_y = len(df.index) * y_coverage / (df[y].max() - df[y].min())
            # set epsilon for possible use of dbscan if we have found a reasonable amount of sparse areas
            # mean of x and y distance used
            eps = abs((2 * sparse_results[4] * xdev + 2 * sparse_results[5] * ydev) / 2)
            # avg of the two parameters
            min_pts = (min_p_x + min_p_y) / 2
        choice = list(ca_properties.keys())[high_rating[0]]
    # tie between multiple candidates - complexity decides:
    # fuzz > gmm > dbscan > aggl
    else:
        if 0 in high_rating:
            choice = 'fuzz'
        elif 3 in high_rating:
            choice = 'gmm'
        elif 1 in high_rating:
            # reuse h from sparse detection
            xdev = math.sqrt(df[x].var())
            ydev = math.sqrt(df[y].var())
            x_coverage = xdev / (df[x].max() - df[x].min())
            y_coverage = ydev / (df[y].max() - df[y].min())
            # average density as threshold for how many values need to be found - used as min_pts if dbscan is used
            min_p_x = len(df.index) * x_coverage / (df[x].max() - df[x].min())
            min_p_y = len(df.index) * y_coverage / (df[y].max() - df[y].min())
            # set epsilon for possible use of dbscan if we have found a reasonable amount of sparse areas
            # mean of x and y distance used
            eps = abs((2 * sparse_results[4] * xdev + 2 * sparse_results[5] * ydev) / 2)
            # avg of the two parameters
            min_pts = (min_p_x + min_p_y) / 2
            choice = 'dbscan'
        else:
            choice = 'aggl'

    return choice, eps, min_pts

# evaluate the reached score by comparing to the defaults and acceptance levels
# n contains the index of the scores that need to be checked
def scoreEval(score, abb, n):
    result = {}
    for ca_name in ca_properties:
        result[ca_name] = 0
        for i in n:
            if result[ca_name] < 2 \
                    and ca_properties[ca_name][abb + "_default"] - ca_properties[ca_name][abb + "_acceptance"] / 4 \
                    <= score[i] <= \
                    ca_properties[ca_name][abb + "_default"] + ca_properties[ca_name][abb + "_acceptance"] / 4:
                result[ca_name] = 2
            elif result[ca_name] < 1 \
                    and ca_properties[ca_name][abb + "_default"] - ca_properties[ca_name][abb + "_acceptance"] \
                    <= score[i] <= \
                    ca_properties[ca_name][abb + "_default"] + ca_properties[ca_name][abb + "_acceptance"]:
                result[ca_name] = 1
    return result
