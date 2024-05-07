
import numpy as np
import pandas as pd
import warnings

YE_lb = 0.01
def_num_std = 3
absent_ids =[ \
    'cpd90003', # starch
    'cpd90021', # xylan
    'cpd90022', # xyloligo
    'cpd00028', # heme
    'cpd01262', # amylotriose
    'cpd01132', # puromycin
    'cpd01399', # Maltotetraose
    'cpd00010', # CoA
    'cpd15494', # maltoheptaose
    'cpd11976', # maltodextrin
    'cpd00006', # NADP
    'cpd11451', # mql7
    'cpd11606', # menaquinone
    'cpd17026', # methymenaquinol 7
    'cpd17027', # methylmenaquinone 7
    'cpd90005', # Maltooctaose
    'cpd90006', # Maltoundecaose
    'cpd90007', # Maltohexaose
    'cpd90008', # Maltotridecaose
    'cpd90020', # cellulose
    'cpd00003', # NAD
    'cpd00038', # GTP
    'cpd00109', # Cytochrome c3+
    'cpd00110', # Cytochrome c2+
    'cpd15560', # ubiquinone-8
    'cpd15561', # ubiquinol-8
    'cpd00393', # folate
    'cpd00526', # cholate
    'cpd90003', # starch n=27
    'cpd90004', # starch n=19
    'cpd23430', # isocholate
    'cpd23431', # 3-dehydrocholate
    'cpd01318', # Glycocholate
    'cpd03247', # Glycochenodeoxycholate
    ]


def set_medium(model, sample, media_names, media_table):
    """Set medium constraints.

    Parameters
    ----------
    model : micom Community instance
        Community model.
    medium_name : str
        Medium ID.
    media_table : str
        Table with all media composition.

    Returns
    -------
    micom Community instance
        Community model with new exchange constraints.

    """
    
    excRxns = [r.id for r in model.exchanges]
    idx1 = media_table['medium'].isin([media_names[sample]])
    idx2 = media_table['excRxn'].isin(excRxns)
    BA_excRxns = media_table['excRxn'][np.logical_and(idx1, idx2)]
    BA_lbs = media_table['maxFlux'][np.logical_and(idx1, idx2)]
    d = dict(zip(BA_excRxns, BA_lbs))
    # set yeast extract and other compounds
    YE_excRxns = pd.Series(excRxns)[~pd.Series(excRxns).isin(BA_excRxns)]
    d.update(dict(zip(YE_excRxns, [YE_lb]*len(YE_excRxns))))
    # forbid uptake of heavy compounds
    for met in absent_ids:
        if 'EX_' + met + '_m' in excRxns: d['EX_' + met + '_m'] = 0
    d['EX_cpd00007_m'] = 0 # no oxygen
    d['EX_cpd01024_m'] = 0 # no methane

    model.medium = d

    return model


def set_biochemical_constraints(model, sample, biochemistry_table, community_abundance=1.0, num_std=def_num_std):
    """Set biochemical measurement constraints.

    Parameters
    ----------
    model : micom Community instance
        Community model.
    sample : str
        Sample name.
    biochemistry_table : pandas DataFrame
        Table with all the biochemical data.
    community_abundance : float
        Scaling factor to account for incomplete consumption/production

    Returns
    -------
    micom Community instance
        Community model with new exchange constraints.

    """
    
    rxns = [r.id for r in model.reactions]
    # split table by type of constraint (only lower bound or both)
    lb_table = biochemistry_table.loc[biochemistry_table['type'] == 'lb', :].reset_index(drop=True)
    mean_table = biochemistry_table.loc[biochemistry_table['type'] == 'mean', :].reset_index(drop=True)
    std_table = biochemistry_table.loc[biochemistry_table['type'] == 'std', :].reset_index(drop=True)
    
    # set lower bounds
    if not lb_table.empty:
        for i in range(lb_table.shape[0]):
            if lb_table['excRxn'][i] in rxns:
                model.reactions.get_by_id(lb_table['excRxn'][i]).lower_bound = lb_table[sample][i]

    # set bounds for measurements with mean +- std
    if not mean_table.empty:
        for i in range(mean_table.shape[0]):
            if mean_table['excRxn'][i] in rxns:
                # if std is available
                if mean_table['met'][i] in list(std_table['met']):
                    std_idx = std_table.index[std_table['met'] == mean_table['met'][i]]
                    # there should be a single std for each mean provided
                    if len(std_idx) > 1:
                        warnings.warn('Redundant rows detected for ' + mean_table['met'][i] + '. Taking the first one.')
                    std_idx = std_idx[0]
                    model.reactions.get_by_id(mean_table['excRxn'][i]).lower_bound = community_abundance*mean_table[sample][i] - num_std*std_table[sample][std_idx]
                    model.reactions.get_by_id(mean_table['excRxn'][i]).upper_bound = community_abundance*mean_table[sample][i] + num_std*std_table[sample][std_idx]

                else:
                    model.reactions.get_by_id(mean_table['excRxn'][i]).lower_bound = community_abundance*mean_table[sample][i]
                    model.reactions.get_by_id(mean_table['excRxn'][i]).upper_bound = community_abundance*mean_table[sample][i]

    return model
