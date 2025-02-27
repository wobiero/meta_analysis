import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MetaRegression:
    def __init__(self, effect_sizes, variances, covariates):
        """
        Initialize the MetaRegression class.
        
        Args:
            effect_sizes (array-like): Array of effect sizes.
            variances (array-like): Array of variances for effect sizes.
            covariates (pd.DataFrame): DataFrame of study-level covariates.
        """
        self.effect_sizes = np.array(effect_sizes)
        self.variances = np.array(variances)
        
        # Check if covariates already has a constant
        if isinstance(covariates, pd.DataFrame) and 'const' not in covariates.columns:
            self.covariates = sm.add_constant(covariates)
        else:
            self.covariates = covariates
            
        self.validate_data()
        
    def validate_data(self):
        """Validate effect sizes, variances, and covariates."""
        if len(self.effect_sizes) != len(self.variances) or len(self.effect_sizes) != len(self.covariates):
            raise ValueError("Effect sizes, variances and covariates must be of the same length!")
            
        # Check for NaN values
        if np.isnan(self.effect_sizes).any() or np.isnan(self.variances).any():
            raise ValueError("Effect sizes and variances cannot contain NaN values")
            
    def fit(self):
        """Fit meta-regression model."""
        weights = 1 / self.variances
        self.model = sm.WLS(self.effect_sizes, self.covariates, weights=weights)
        self.results = self.model.fit()
        return self.results
    
    def plot(self, covariate_name):
        """Plot the relationship between a covariate and effect sizes."""
        if not hasattr(self, 'results'):
            raise ValueError("You must run fit() before plotting")
            
        covariate = self.covariates[covariate_name]
        plt.scatter(covariate, self.effect_sizes, label="Studies")
        plt.plot(covariate, self.results.fittedvalues, color="red", label="Meta-Regression")
        plt.ylabel("Effect size")
        plt.xlabel(covariate_name)
        plt.title(f"Meta-Regression: {covariate_name} vs Effect Size")
        plt.legend(loc="best")
        plt.show()
        
    def to_dataframe(self):
        """
        Export meta-regression results as a Pandas DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame containing coefficients, p-values, and other metrics.
        """
        if not hasattr(self, 'results'):
            raise ValueError("You must run fit() before exporting results")
            
        results_df = pd.DataFrame({
            'covariate': self.results.params.index,
            'coefficient': self.results.params,
            'std_error': self.results.bse,
            'p_value': self.results.pvalues,
            'conf_int_lower': self.results.conf_int()[0],
            'conf_int_upper': self.results.conf_int()[1]
        })
        return results_df
        
    def to_csv(self, filename):
        """
        Export meta-regression results to a CSV file.
        
        Args:
            filename (str): Name of the output CSV file.
        """
        df = self.to_dataframe()
        df.to_csv(filename, index=False)
        
    def to_excel(self, filename):
        """
        Export meta-regression results to an Excel file.
        
        Args:
            filename (str): Name of the output Excel file.
        """
        df = self.to_dataframe()
        df.to_excel(filename, index=False)