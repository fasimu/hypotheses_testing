"""
Heart Disease Research

This project will investigate some data from a sample patients
who were evaluated for heart disease at the Cleveland Clinic Foundation.
The data was downloaded from the UCI Machine Learning Repository
(https://archive.ics.uci.edu/ml/datasets/Heart+Disease) and cleaned for analysis.

It contains the following variables:
- age: age in years
- sex: sex assigned at birth; 'male' or 'female'
- trestbps: resting blood pressure in mm Hg
- chol: serum cholesterol in mg/dl
- cp: chest pain type ('typical angina', 'atypical angina', 'non-anginal pain', or 'asymptomatic')
- exang: whether the patient experiences exercise-induced angina (1: yes; 0: no)
- fbs: whether the patient’s fasting blood sugar is >120 mg/dl (1: yes; 0: no)
- thalach: maximum heart rate achieved in exercise test
- heart_disease: whether the patient is found to have heart disease ('presence': diagnosed with heart disease; 'absence': no heart disease)
"""

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2_contingency

# load data
heart = pd.read_csv('heart_disease.csv')
print(heart.head())

# Predictors of Heart Disease
""" Question: Is thalach associated with whether or not
a patient will ultimately be diagnosed with heart disease? 
Answer: Based on this plot, patients diagnosed with heart disease
generally had a lower maximum heart rate during their exercise test.
"""
# box plot of `thalach` based on heart disease
sns.boxplot(x=heart.thalach, y=heart.heart_disease)
plt.show()

# save `thalach` for hd patients and non-hd patients
thalach_hd = heart.thalach[heart.heart_disease == 'presence']
thalach_no_hd = heart.thalach[heart.heart_disease == 'absence']

# calculate mean and median difference
mean_diff = np.mean(thalach_hd) - np.mean(thalach_no_hd) # -19.11905597473242
median_diff = np.median(thalach_hd) - np.median(thalach_no_hd) # -19.0

""" Question: Is the average thalach of a heart disease patient is significantly
different from the average thalach for a person without heart disease?
Answer: p-value = 3.456964908430172e-14
Since pval is much lower than 0.05 means that the Null hypoteses is rejected.
Conclude that there is a significant difference in thalach for people with
heart disease compared to people without heart disease.
"""
# run two-sample t-test
tstat, pval = ttest_ind(thalach_hd, thalach_no_hd)
print('p-value for `thalach` two-sample t-test: ', pval)

# investigating other quantitative variables
# 'age', 'trestbps' (resting blood pressure), and 'chol' (cholesterol)
""" Question: Is any of those variables (age, trestbps, and chol) associated with
whether or not a patient will ultimately be diagnosed with heart disease?
Answer:
p-value of age is 8.955636917529706e-05
means that age is (high) significantly associated with heart disease.
p-value of trestbps is 0.008548268928594928
means that trestbps is significantly associated with heart disease.
p-value of chol is 0.13914167020436527
means that cholesterol is not significantly associated with heart disease.
"""
def show_boxplot(x, y):
  plt.clf()
  sns.boxplot(x, y)
  plt.show()

def ttest_hypotheses(x):
  _hd = x[heart.heart_disease == 'presence']
  _no_hd = x[heart.heart_disease == 'absence']
  tstat, pval = ttest_ind(_hd, _no_hd)
  return print('p-value for two-sample t-test: ', pval)

# age
show_boxplot(x=heart.age, y=heart.heart_disease)
ttest_hypotheses(heart.age)
# trestbps
show_boxplot(x=heart.trestbps, y=heart.heart_disease)
ttest_hypotheses(heart.trestbps)
# chol
show_boxplot(x=heart.chol, y=heart.heart_disease)
ttest_hypotheses(heart.chol)


# Chest Pain and Max Heart Rate
""" Investigate the relationship between thalach (maximum heart
rate achieved during exercise) and the type of heart pain
a person experiences.
"""
# box plot of `thalach` based on chest pain type
show_boxplot(heart.thalach, heart.cp)

# save `thalach` for chest pain type
thalach_typical = heart.thalach[heart.cp == 'typical angina']
thalach_asymptom = heart.thalach[heart.cp == 'asymptomatic']
thalach_nonangin = heart.thalach[heart.cp == 'non-anginal pain']
thalach_atypical = heart.thalach[heart.cp == 'atypical angina']

""" Question: Is there at least one pair of chest pain categories for which
people in those categories have significantly different thalach?
Answer: p-value for ANOVA is 1.9065505247705008e-10
means that there is at least one pair of chest pain types (cp) for which
people with those pain types have significantly different average max
heart rates during exercise (thalach).
"""
# run test ANOVA
fstat, pval = f_oneway(thalach_typical, thalach_asymptom, thalach_nonangin, thalach_atypical)
print('p-value for ANOVA: ', pval)

""" Question: Which of those pairs are significantly different?
Answer:
pair between asymptomatic ~ atypical angina, non-anginal pain, typical angina
is "Rejected" means that people with those chest pain types have significantly
different maximum heart rates during exercise.

Maybe surprisingly, people who are 'asymptomatic' seem to have a lower maximum
heart rate (associated with heart disease) than people who have other kinds
of chest pain.
"""
# run Tukey’s range test
results = pairwise_tukeyhsd(endog = heart.thalach, groups = heart.cp)
print(results)


# Heart Disease and Chest Pain
""" Investigate the relationship between the kind of chest pain a person
experiences and whether or not they have heart disease.
Answer:
p-value for chi-square test is 1.2517106007837527e-17.
This is less than 0.05, so we can conclude that there is a significant
association between these variables.
"""
# contingency table of heart disease vs cp
table = pd.crosstab(heart.cp, heart.heart_disease)

# run Chi-Square test.
chi2, pval, dof, exp = chi2_contingency(table)
print('p-value for chi-square test: ', pval)