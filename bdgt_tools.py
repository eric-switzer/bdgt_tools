"""
Utilities for managing a budget
"""
import copy
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def rebin(dict_in, total, precision=1.):
    """Take a dictionary of values and convert them to sum to a new value with
    given precision.

    Use: converting billable hours into more round units with fixed total time.
    """
    (keys, values) = zip(*dict_in.iteritems())
    values = np.array(values)
    values *= total / np.sum(values) / precision
    array_sum = np.sum(values)

    lower = 0
    result = np.zeros_like(values)
    diff = np.zeros_like(values)
    result = np.zeros_like(values)
    for ind, val in enumerate(values):
        result[ind] = np.floor(val)
        diff[ind] = val - result[ind]
        lower += result[ind]

    difference = int(array_sum - lower)
    inc_ind = np.argpartition(diff, -difference)[-difference:]
    result[inc_ind] += 1

    result *= precision

    dict_out = dict(zip(keys, result))
    return dict_out


def spread(time_in, name, spread_to):
    """Take a dictionary of values and spread it to other entries.
    Use: distributing a task (e.g. mgmt) over all grants
    """
    time_out = copy.copy(time_in)
    nreal = len(spread_to)

    time_spread = time_in[name] / float(nreal)
    for entry in spread_to:
        time_out[entry] += time_spread

    del time_out[name]
    return time_out


def print_dict(dict_in, fmt="%d"):
    (keys, values) = zip(*dict_in.iteritems())
    for key in sorted(keys):
        fmt_str = "%s: %s" % (key, fmt)
        print fmt_str % dict_in[key]


def first_transaction(table, amount_col="Amount"):
    """Determine the first transaction
    """
    first_trans = table[amount_col][0]

    # if there is no tranaction entry set to zero
    if pd.isnull(first_trans):
        first_trans = 0

    return first_trans


def starting_balance(table, amount_col="Amount", balance_col="Balance"):
    """Recover the starting balance for an account
    Assumes that the starting balance entry already includes the first
    transaction
    """
    try:
        first_trans = first_transaction(table, amount_col=amount_col)
        bal_after_trans = table[balance_col][0]

        starting = bal_after_trans - first_trans

        # if there was no balance, set starting to zero
        if pd.isnull(bal_after_trans):
            starting = 0

    except (IndexError, KeyError) as err:
        starting = 0

    return starting


def rebalance(table, amount_col="Amount", balance_col="Balance"):
    """Re-calculate the balance column

    If balance_col is given, use this to determine the starting value
    """
    output = table.copy()
    starting = starting_balance(table, amount_col=amount_col,
                                balance_col=balance_col)

    #print "Rebalance starting with ", starting

    output[balance_col] = table[amount_col].cumsum() + starting

    return output


def combine(table_list, amount_col="Amount", balance_col="Balance",
            account_col="Account", append_name=None,
            uncategorized="Uncategorized", append_skip="transfer"):
    """Combine budgets
    Use the .name attribute to assign a column to the original accounts
    This data is kept in the "account_col"
    """
    # first go through and get names and total balance
    keys = []
    unnamed = 0
    total_balance = 0

    for tab in table_list:
        try:
            name = tab.name
            keys.append(name)
        except AttributeError:
            name = "Not_specified_%d" % unnamed
            unnamed += 1
            keys.append(name)

        starting = starting_balance(tab, amount_col=amount_col,
                                    balance_col=balance_col)

        print "Account %s starting balance: %s" % (name, starting)

        total_balance += starting

    print "Accounts: ", keys
    print "Total starting balance: ", total_balance

    combined = pd.concat(table_list, keys=keys)

    # move the accounts level to a column
    combined.reset_index(level=0, inplace=True)
    combined.rename(index=str, columns={"level_0": account_col}, inplace=True)

    if append_name is not None:
        combined[append_name] = combined[append_name].fillna(uncategorized)
        combined[append_name] = combined[append_name].astype(str) + \
                                combined[account_col].astype(str)

    # sort on date
    combined.sort_index(inplace=True)

    # redo the starting balance as sum of starting from all accounts
    first_trans = combined[amount_col][0]
    combined[balance_col][0] = total_balance + first_trans

    # rebalance the combined accounts
    combined = rebalance(combined, amount_col=amount_col,
                         balance_col=balance_col)

    # for some reasons the date is turned into a string
    combined.index = combined.index.to_datetime()

    return combined


def slice_categories(table, balance_col="Balance", amount_col="Amount",
                     category_col="Category", uncategorized="Uncategorized"):
    """re-starts all the balances; return initial balance
    handle missing categories
    collapse transfers
    categorize groups as either income or loss
    """
    # fill any categories with Uncategorized
    df = table.copy()
    df[category_col] = table[category_col].fillna(uncategorized)

    if balance_col not in df.columns:
        print "missing %s" % balance_col
        df[balance_col] = 0.

    start = starting_balance(df, amount_col=amount_col,
                             balance_col=balance_col)

    cat_list = df[category_col].tolist()
    cat_list = list(set(cat_list))
    cat_list = [entry.strip() for entry in cat_list]

    #print cat_list
    split_cats = []
    for cat in cat_list:
        #new_df = df[df[category_col].str.match(cat)]
        new_df = df.loc[df[category_col] == cat]
        # get rid of the category column and name the table
        new_df = new_df.drop(category_col, 1)

        new_df = new_df.groupby(new_df.index).sum()
        #new_df = new_df.resample("D", how='sum').dropna()

        new_df[balance_col][0] = np.nan
        new_df = rebalance(new_df, amount_col=amount_col,
                            balance_col=balance_col)

        new_df.name = cat

        split_cats.append(new_df.copy())

    return cat_list, split_cats


def sort_dict_on_value(dict_in):
    dsort = sorted(dict_in.items(),
                   key=operator.itemgetter(1))

    dsort = [item[0] for item in dsort]

    return dsort


def plot_wedges(cat_list, split_cats, idx=None, title="Budget wedges", alpha=0.3,
                cumulative=True, amount_field="Amount", balance_field="Balance"):
    """Show cumulative input and output broken down by category
    idx is an optional datetimeindex vector over which to plot the data
    cumulative picks between cumulative and instantaneous spending
    """
    # First distinguish between income and spending
    income = {}
    spending = {}
    income_final = {}
    spending_final = {}
    for cat_name, cat in zip(cat_list, split_cats):
        final_balance = cat[balance_field][-1]
        #print cat_name, final_balance
        cat.name = cat_name
        if final_balance > 0:
            income[cat_name] = cat
            income_final[cat_name] = -final_balance

        if final_balance < 0:
            spending[cat_name] = cat
            spending_final[cat_name] = final_balance

    sorted_income = sort_dict_on_value(income_final)
    sorted_spending = sort_dict_on_value(spending_final)
    print "sorted income: ", sorted_income
    print "sorted spending: ", sorted_spending

    # if no index is given, determine the time base from the input tables
    if idx is None:
        for cat_name, cat in zip(cat_list, split_cats):
            if idx is None:
                idx = cat.copy().index
            else:
                idx = idx.union(cat.index)

    fig, ax = plt.subplots(figsize=(10, 7))

    n_spend = len(spending)
    color = cm.rainbow(np.linspace(0, 1, n_spend))

    if cumulative:
        plot_field = balance_field
    else:
        plot_field = amount_field

    start = np.zeros(shape=idx.shape)
    for entry, cname in enumerate(sorted_spending):
        spend = spending[cname]
        name = spend.name
        name += "=%.2g" % -(spending_final[cname])

        #print entry, cname, name
        if cumulative:
            data = spend.reindex(idx, method="pad")
        else:
            data = spend.reindex(idx, method=None)

        data[plot_field].fillna(0, inplace=True)

        cost = -data[plot_field]
        lwr = start
        upr = start + cost
        plt.fill_between(data.index, lwr, upr, linewidth=0,
                         facecolor=color[entry], alpha=alpha, label=name)

        lwr += cost

    if len(sorted_spending) > 0:
        ax.plot_date(data.index, upr, "-", linewidth=2, color="red",
                     label="total spending")

    n_income = len(income)
    color = cm.rainbow(np.linspace(0, 1, n_income))

    start = np.zeros(shape=idx.shape)
    for entry, cname in enumerate(sorted_income):
        inc = income[cname]
        name = inc.name
        name += "=%.2g" % (income_final[cname])

        if cumulative:
            data = inc.reindex(idx, method="pad")
        else:
            data = inc.reindex(idx, method=None)

        data[plot_field].fillna(0, inplace=True)

        cost = data[plot_field]
        lwr = start
        upr = start + cost

        ax.plot_date(data.index, upr, ".", linewidth=2, label=name)

        lwr += cost

    if len(sorted_income) > 0:
        ax.plot_date(data.index, upr, "-", linewidth=2, color="black",
                     label="total income")

    fig.autofmt_xdate()
    ax.yaxis.tick_right()
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_right()

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper center',
              bbox_to_anchor=(0.5, -0.2), ncol=3)

    plt.title(title)

    plt.show()
