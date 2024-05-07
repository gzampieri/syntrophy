
import numpy as np
import pandas as pd
import warnings
from cobra.core.gene import parse_gpr
from ast import Name, And, Or, BoolOp, Expression

MIN_EXP_SCALE = 10

class CoCo():
    """Main class for COndition-specific COmmunity model creation."""


    def __init__(self, gene_expr, default_ub=1000., sparse_community=True):
        """Initialise data for CoCo model creation.

        Parameters
        ----------
        gene_expr : pandas DataFrame
            Table containing all the input transcriptomic profiles for T taxa over S samples.
        default_ub : float
            Default "infinity" flux upper bound of input models (usually 1000.).

        """

        self.gene_expr = gene_expr.copy()
        self.default_ub = default_ub
        self.sparse_community = sparse_community

        self.taxa = np.unique(self.gene_expr.index.get_level_values(0))
        self.samples = np.array(self.gene_expr.columns)
        self.fold_changes = pd.DataFrame(index=self.gene_expr.index, columns=self.samples)

        # Correct null expression values using 1/MIN_EXP_SCALE of minimum expression value for each gene
        min_gene_expr = pd.concat([gene_expr[gene_expr > 0].min(axis=1)]*len(self.samples), axis=1)
        min_gene_expr.columns = gene_expr.columns
        self.gene_expr[self.gene_expr == 0] = min_gene_expr[self.gene_expr == 0] / MIN_EXP_SCALE
        
        # other preliminary calculations
        self.__calculate_fold_changes(gene_expr)
        self.alpha_matrix = pd.DataFrame(index=self.taxa, columns=self.samples)
        self.count_log_matrix = pd.DataFrame(index=self.taxa, columns=self.samples)
        self.__set_params() 


    def __set_params(self):
        """Calculate private parameters.
        
        These include a TxS matrix for the logarithmic map relaxation 
        and another TxS matrix for setting taxon-specific base bounds."""

        for s in self.samples:
            # total count sum for each taxon in a sample
            count_sums = self.gene_expr[s].sum(axis=0, level=0)
            # maximum count sum over all taxa in the sample
            max_count = count_sums.max()
            # scale alphas based on the maximum count sum
            self.alpha_matrix.loc[:, s] = count_sums / max_count
            # set a logarithmic scale for taxon-specific base bounds
            self.count_log_matrix.loc[:, s] = np.log(count_sums + 1)
        return


    def get_alpha(self):
        return self.alpha_matrix


    def get_fold_changes(self):
        return self.fold_changes


    def get_count_logs(self):
        return self.count_log_matrix


    def get_param_range(self):
        """Estimate feasible parameter range."""
        
        max_factors = pd.DataFrame(index=self.taxa, columns=self.samples)
        for s in max_factors.columns:
            for t in max_factors.index:
                max_factors.loc[t, s] = np.log(self.fold_changes.loc[t, s].max())
        max_bounds = self.count_log_matrix.mul(max_factors) 
        return self.default_ub/max_bounds.max().max(), self.default_ub/self.count_log_matrix.max().max()


    def __calculate_fold_changes(self, gene_expr_orig=None):
        """Calculate gene expression fold change for each sample.

        Parameters
        ----------
        gene_expr : pandas DataFrame
            Table containing all the input transcriptomic profiles.

        Returns
        -------
        pandas DataFrame
            Table containing transcriptomic fold change profiles for all the samples.

        """

        num_samples = len(self.gene_expr.columns)
        if self.sparse_community == True:
            expr_mean = pd.DataFrame(index=self.gene_expr.index, columns=self.gene_expr.columns)
            for t in self.taxa:
                # samples where the total count sum for a taxon is greater than the number of genes (i.e. decent count distribution)
                sufficient_counts_idx = gene_expr_orig.loc[t, :].sum(axis=0) >= gene_expr_orig.loc[t, :].shape[0]
                # gene count mean over those samples
                t_mean = pd.concat([self.gene_expr.loc[t, sufficient_counts_idx].mean(axis=1)]*num_samples, axis=1)
                t_mean.columns = self.samples
                t_mean.index = pd.MultiIndex.from_product(iterables=[[t], t_mean.index])
                expr_mean.loc[t, :] = t_mean
                # count fold change over those samples
                self.fold_changes.loc[t, :] = self.gene_expr.div(expr_mean)
                # count fold change over the other samples (i.e. fc = 1)
                self.fold_changes.loc[t, ~sufficient_counts_idx] = 1.0
        else:
            # gene count mean over all samples for all taxa
            expr_mean = pd.concat([self.gene_expr.mean(axis=1)]*num_samples, axis=1)
            expr_mean.columns = self.gene_expr.columns
            # count fold change over all samples
            self.fold_changes = self.gene_expr.div(expr_mean)
        
        return
    

    def check_gene_coverage(self, model, verbose=True):
        """Main function for COndition-specific COmmunity model creation.

        Parameters
        ----------
        model : micom Community instance
            Community model.
        verbose : bool
            Whether or not to print additional information.

        Returns
        -------
        pandas Series
            A list of model genes absent from the gene expression dataframe.

        """

        genes = pd.Series([g.id for g in model.genes])
        missing_genes = genes.loc[~genes.isin(self.fold_changes.index.get_level_values(1))]

        if verbose:
            print(len(genes)-len(missing_genes), 'genes out of', len(genes), 'in the input model have expression data')
        
        if len(missing_genes) == len(genes):
            warnings.warn('\tWARNING: No gene expression data found for input model')
        elif verbose:
            # individually check the genes of each taxon
            for t in model.taxa:
                genes_taxon = genes.loc[genes.str.contains(t)]
                present_genes_taxon = genes_taxon.loc[genes_taxon.isin(self.fold_changes.index.get_level_values(1))]
                print('\t... of which', len(present_genes_taxon), 'in', t, \
                      '(' + str(np.round(100*len(present_genes_taxon)/len(genes_taxon), 1)) + '%)')

        return missing_genes


    def build(self, model, sample_name, delta=1.0, gamma=1.0, meta=True, verbose=False):
        """Main function for COndition-specific COmmunity model creation.

        Parameters
        ----------
        model : micom Community instance
            Base community model.
        sample_name : string
            Name of the sample (i.e. condition).
        delta : float
            Parameter that controls the base bounds for each taxon model.
        gamma : float
            Parameter that controls the impact of gene expression on reaction bounds.
        meta : bool
            Whether or not to use the taxon-specific parametrisation (true for metatrnscriptomics,
            false for standard transcriptomics METRADE-like).
        verbose : bool
            Whether or not to print additional information.

        Returns
        -------
        micom Community instance
            A condition-specific community model.

        """

        if gamma <= 0:
            raise('gamma should be strictly positive!')
        if delta <= 0:
            raise('delta should be strictly positive!')
        
        rxns = pd.Series([r.id for r in model.reactions])
        ubs = np.array([r.upper_bound for r in model.reactions])
        lbs = np.array([r.lower_bound for r in model.reactions])
        gprs = np.array([r.gene_reaction_rule for r in model.reactions])

        missing_genes = self.check_gene_coverage(model, verbose)

        # select gene expression values for the selected condition
        fold_change_dict = dict(zip(self.fold_changes.index.get_level_values(1), self.fold_changes[sample_name].values))
        
        # populate with 1s the fold changes for the genes in the model without available expression data
        fold_change_dict.update(dict(zip(missing_genes, [1.0]*len(missing_genes))))

        # calculate gene set expression (effective reaction expression)
        rxn_expr = np.array([1.0]*len(rxns))
        for i in range(len(rxn_expr)):
            rxn_expr[i] = self.__gene_set_expression(model.reactions[i].gene_reaction_rule, fold_change_dict)

        factors = np.array([1.0]*len(rxns))
        if meta:
            # calculate taxon-dependent bound factors (alpha and delta) for each reaction
            alphas = np.array([1.0]*len(rxns))
            deltas_up = ubs
            deltas_low = lbs
            idx1 = ubs == self.default_ub
            idx2 = lbs == -self.default_ub
            idx3 = gprs != ''
            for t in self.taxa:
                alphas[rxns.str.contains(t)] = self.alpha_matrix.loc[t, sample_name]
                deltas_up[np.logical_and.reduce([rxns.str.contains(t), idx1, idx3])] = delta * self.count_log_matrix.loc[t, sample_name]
                deltas_low[np.logical_and.reduce([rxns.str.contains(t), idx2, idx3])] = -delta * self.count_log_matrix.loc[t, sample_name]
            # calculate gene-dependent bound factors for each reaction
            idx1 = rxn_expr >= 1
            idx2 = rxn_expr < 1
            factors[idx1] = alphas[idx1] * (1 + gamma * alphas[idx1] * np.log(rxn_expr[idx1])) + (1.0 - alphas[idx1])
            factors[idx2] = alphas[idx2] / (1 + gamma * alphas[idx2] * np.abs(np.log(rxn_expr[idx2]))) + (1.0 - alphas[idx2])
            # apply new constraints
            for i in range(len(rxns)):
                model.reactions.get_by_id(rxns[i]).upper_bound = deltas_up[i] * factors[i]
                model.reactions.get_by_id(rxns[i]).lower_bound = deltas_low[i] * factors[i]
        else: # METRADE
            # calculate gene-dependent bound factors for each reaction
            idx1 = rxn_expr >= 1
            idx2 = rxn_expr < 1
            factors[idx1] = 1 + gamma * np.log(rxn_expr[idx1])
            factors[idx2] = 1 / (1 + gamma * np.abs(np.log(rxn_expr[idx2])))
            # apply new constraints
            for i in range(len(rxns)):
                model.reactions.get_by_id(rxns[i]).upper_bound = ubs[i] * factors[i]
                model.reactions.get_by_id(rxns[i]).lower_bound = lbs[i] * factors[i]
        if verbose:
            print('factors between ' + str(np.min(factors)) + ' and ' + str(np.max(factors)))
        
        ubs = np.array([r.upper_bound for r in model.reactions])
        lbs = np.array([r.lower_bound for r in model.reactions])
        if verbose:
            print('upper bounds between ' + str(np.min(ubs[ubs!=0])) + ' and ' + str(np.max(ubs)))
            print('lower bounds between ' + str(np.min(lbs)) + ' and ' + str(np.max(lbs[lbs!=0])))


    def __evaluate_gpr(self, expr, conf_genes): # taken from https://github.com/resendislab/corda/blob/master/corda/util.py
        """Internal Corda-style evaluation of a gene-protein-reaction rule."""
        if isinstance(expr, Expression):
            return self.__evaluate_gpr(expr.body, conf_genes)
        elif isinstance(expr, Name):
            if expr.id not in conf_genes:
                return 1
            return conf_genes[expr.id]
        elif isinstance(expr, BoolOp):
            op = expr.op
            if isinstance(op, Or):
                return max(self.__evaluate_gpr(i, conf_genes) for i in expr.values)
            elif isinstance(op, And):
                return min(self.__evaluate_gpr(i, conf_genes) for i in expr.values)
            else:
                raise TypeError("unsupported operation " + op.__class__.__name__)
        elif expr is None:
            return 1
        else:
            raise TypeError("unsupported operation  " + repr(expr))


    def __gene_set_expression(self, rule, conf_genes): # taken from https://github.com/resendislab/corda/blob/master/corda/util.py
        """Calculate effective reaction expression based on a gene-protein-reaction rule.

        Parameters:
        ----------
        rule : str
            A gene-protein-reaction rule. For instance "A and B" or "A or B".
        conf_genes : dict
            A str->float map denoting the mapping of gene IDs to expression values.
            
        Returns
        -------
        float
            Gene set expression.
        """

        ast_rule, _ = parse_gpr(rule)
        return self.__evaluate_gpr(ast_rule, conf_genes)


    def __str__(self):
        return "A condition-specific community genome-scale metabolic model (CoCo-GEM) builder."
