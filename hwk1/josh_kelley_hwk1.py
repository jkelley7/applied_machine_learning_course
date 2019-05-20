############################################################################
#
# Joshua Kelley Spring '19'
# Applied Machine Learning BUAN 6341.003
# Assignment 1 
#
#############################################################################

import numpy as np
import pandas as pd
import os
import glob
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
pd.set_option('display.max_columns',54)

############################################################################
#
# Functions
#
#############################################################################
def cost_func(x,y,theta):
    """Compute cost"""
    xr,xc = x.shape
    yhat = np.array(np.dot(x,theta),dtype = np.float64)
    error = yhat-y
    tss = np.sum(np.square(error, dtype=np.float64))
    mss = (1/(2*xr))*tss
    return mss

def grad_desc(x,y,theta,itera,alpha,improvement, fit_intercept = True):
    """
    
    Params:
    x: x matrix of features
    y: target variable
    itera: number of iterations to perform
    alpha: step to take
    improvement: if it converges at our improvement parameter than stop
    fit_intercept: include intercept/bias variable or not
    
    Returns:
    theta_it: theta history
    cost_it: cost history
    
    """
    xrows,xcols = x.shape
    thetas = theta
    
    if fit_intercept == True:
        x_b = np.ones((xrows,1))
        x = np.hstack((x_b,x))
        xrows,xcols = x.shape
        thetas = np.random.rand(xcols,1)
    try:
        yrows,ycols = y.shape
    except:
        y = y[:,np.newaxis]
        yrows, ycols = y.shape    
    
    #itera = itera+1
    cost_ = []
    #beta rows and cols
    tn,tm = thetas.shape
    cost_it = np.zeros(itera)
    theta_it = np.zeros((itera,xcols))
    previous_improvement = improvement + 1
    curr_iter = 0
    while previous_improvement > improvement and curr_iter < itera:
    #for i in range(itera):
        pred = np.array(np.dot(x, thetas), dtype=np.float64)
        errors = pred-y
        grad = (x.T.dot(errors))*(1/xrows)
        thetas -= alpha*grad
        cost = cost_func(x,y,thetas)
        theta_it[curr_iter,:] = thetas.T
        cost_it[curr_iter] = cost

        if curr_iter == 0:
            pass
        else:
            previous_improvement = (cost_it[curr_iter-1] - cost_it[curr_iter])    
        curr_iter +=1
    
    best_betas = theta_it[np.argmin(cost_it[cost_it > 0])]
    #old return for best betas theta_it[np.argmin(cost_it)],
    return theta_it, cost_it[cost_it > 0], best_betas

def predict(X, y, theta, fit_intercept= True):
    """
    Takes the best params from grad descent and fits it
    
    Params:
    test_df: testing dataframe
    cols_to_fit: columns to pass into
    y: y values
    theta: best betas from our training
    fit_intercept: should you include and intercept term
    
    returns:
    error sum of squares"""
    
    xrows, xcols = X.shape
    
    if fit_intercept == True:
        X_int = np.ones((xrows,1))
        X = np.hstack((X_int,X))
    
    try:
        yrows, ycols = y.shape
    except:
        y = y[:,np.newaxis]
        yrows, ycols = y.shape
    
    try:
        trows, tcols = theta.shape
    except:
        theta = theta[:,np.newaxis]
        trows, tcols = theta.shape 
    
    y_pred = np.dot(X,theta)
    error = y_pred - y
    tss = np.sum(np.square(error))
    ess = tss / (2*xrows)
    
    return ess

def mov_avg (array, ma_window):
    """
    Returns a moving average.
    :params
    :array = numpy array
    :ma_window = how many n periods you want to use
    """
    weights = np.repeat(1.0, ma_window)/ma_window
    ma = np.convolve(array, weights, 'full')
    return ma

def standardscaler(x):
    """Takes and input and returns ther standardized version of that matrix.
    
    :Params
    :x numpy representation of the data"""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis = 0)
    
    scaledx = np.divide((x - mean), std)
    
    return scaledx, mean, std

def inversescaler(x, mean, std):
    """Takes a standardized matrix and inverse transforms it to its normal
    representation
    
    :Params
    :x standardized matrix
    :mean - mean of that matrix
    :std - standard deviation of that matrix
    
    :returns data to its normal representations"""
    
    inverse = (x * std) + mean
    
    return inverse

############################################################################
#
# Colnames
#
#############################################################################
colnames = [
    'page_pop','page_check','page_talk_bt','page_cat','tot_min','tot_max',
    'tot_avg','tot_med','tot_std','l24_min','l24_max','l24_avg','l24_med',
    'l24_std','l24_48_min','l24_48_max','l24_48_avg','l24_48_med','l24_48_std',
        'l48_72_min','l48_72_max','l48_72_avg','l48_72_med','l48_72_std','diff_24_48_min',
        'diff_24_48_max','diff_24_48_avg','diff_24_48_med','diff_24_48_std','num_comm_pre',
        'num_comm_l24','num_comm_pre_24_48','num_comm_fir_24','num_comm_diff_24_48','base_time',
        'pst_len','pst_shre_cnt','pst_prom_stat','h_hours','pub_sun','pub_mon','pub_tue',
        'pub_wed','pub_thu','pub_fri','pub_sat','base_sun','base_mon','base_tue','base_wed',
        'base_thu','base_fri','base_sat','comm_h_hours']


############################################################################
#
# Bringing in the data | You may need to change the filepath| Splitting the dataset
#
#############################################################################
train_df = pd.read_csv('.\data\Dataset\Training\Features_Variant_1.csv', header=None, nrows=30000)
test_df = pd.concat([pd.read_csv(f,header=None,nrows=1000)for f in glob.glob('.\data\Dataset\Testing\TestSet\Test_Case_*.csv')], ignore_index=True)

final_df = pd.concat([train_df,test_df])

new_cols = dict(zip(train_df.columns,colnames))
train_df.rename(new_cols, axis='columns',inplace=True)
test_df.rename(new_cols, axis='columns',inplace=True)
final_df.rename(new_cols, axis='columns',inplace=True)

np.random.seed(20190201)
train_set = .7
tot_train_rows = np.ceil(len(final_df)*train_set)
indices = np.random.choice(a=len(final_df),size=int(tot_train_rows))
train_split_df = final_df.iloc[indices,:]
test_split_df = final_df.iloc[~indices,:]

############################################################################
#
# Exploratory Analysis
#
#############################################################################

train_split_df.loc[:,['page_pop', 'page_talk_bt', 'tot_avg', 'l24_avg', 'l24_48_avg', 'l48_72_avg', 'pst_len', 'pst_shre_cnt', 'comm_h_hours']].corr(method='pearson')
corr = train_split_df.corr(method='pearson')
#corr['comm_h_hours'].sort_values(ascending=False)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(14, 14))
ax.set_title('Correlation of all variables against all variables', fontsize = 16)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.1, cbar_kws={"shrink": .8})


# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(20, 4))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(20, 10, as_cmap=True)

ax.set_title("Zoomed in Correlation Plot of Target Variable", fontsize = 16)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr[-1:], mask=mask[-1:], cmap=cmap, vmax=1, center=0,
            square=True, linewidths=.15, cbar_kws={"shrink": 1})


############################################################################
#
# Splitting the datasets and scaling them
#
#############################################################################

full_train_x = train_split_df.iloc[:,:-1].values
full_train_y = train_split_df.iloc[:,-1].values

full_test_x = test_split_df.iloc[:,:-1].values
full_test_y = test_split_df.iloc[:,-1].values

num_of_cols = 6
top_n = np.abs(corr['comm_h_hours']).sort_values(ascending=False)[1:num_of_cols].index

#these are the ones i'm selecting
top_n3 = ['num_comm_diff_24_48','num_comm_l24','h_hours','pst_shre_cnt','page_talk_bt']
print(top_n3)
top_n_x = train_split_df[top_n3].values 
top_n_y = train_split_df['comm_h_hours'].values

#our top 5 columns
top_n_x_test = test_split_df[top_n3].values

full_xscaler = StandardScaler().fit(full_train_x)
full_train_scaled_x = full_xscaler.transform(full_train_x)

#When scaling you are essentially creating a new variable
#we must use the same mean and standard deviation to ensure this other new variable
#is in the same units as our test case

#The assumption here is that they come from the same population distribution
full_test_scaled_x = full_xscaler.transform(full_test_x)

top_n_xscaler = StandardScaler().fit(top_n_x)
top_ntrain_xscaled = top_n_xscaler.transform(top_n_x)
top_ntest_xscaled = top_n_xscaler.transform(top_n_x_test)

############################################################################
#
# Testing out the function
#
#############################################################################

n_iter = 6000
lr = .001
improvement = 10**-3
np.random.seed(20190201)
theta = np.random.rand(full_train_scaled_x.shape[1],1)
theta_df = pd.DataFrame(theta,columns = ['Beta_values'])
theta_,cost_history, best_betas = grad_desc(full_train_scaled_x,full_train_y,theta,n_iter,lr,improvement, fit_intercept=True)

plt.figure(figsize =(12,7))
plt.plot(cost_history)
plt.title('Testing out Grad Descent', fontsize = 18)
plt.ylabel('error', fontsize = 14)
plt.xlabel('iterations', fontsize = 14)
plt.show()

############################################################################
#
# Trying to see which features really help decrease the errors
#
#############################################################################
lr = .01
n_iter = 2000
improvement = .15
train_exp_error = []
feature_num = []
n_samples, n_features = full_train_scaled_x.shape
for i in range(0,n_features):
    print(i)
    feature_num.append(i)
    trained_scalex = full_train_scaled_x[:,1:i+2]
    n_samples, n_features = trained_scalex.shape
    theta = np.random.rand(n_features,1)
    theta_,cost_history, best_betas = grad_desc(trained_scalex,full_train_y,theta,n_iter,lr,improvement, fit_intercept=True)
    train_exp_error.append(np.min(cost_history))

plt.figure(figsize =(12,7))
plt.title('Minimum Cost by number of features', fontsize=20)
plt.plot(train_exp_error, linestyle = '-',linewidth=2.5, label='cost by feature numbers')
plt.xlabel('Number of features', fontsize = 14)
plt.ylabel('minimum cost error', fontsize = 14)
plt.legend()
plt.tight_layout()
plt.show()

############################################################################
#
# # Experiment 1
#1. Experiment with various values of learning rate ∝ and report on your findings as how the error varies for train and test sets with varying ∝. 
#2. Plot the results. 
#3. Report your best ∝ and why you picked it. 
#
# This part is showing how larger alphas need more iterations to converge than smaller alphas
#############################################################################
np.random.seed(20190201)
alphas_to_test = np.linspace(1,.005,num=2)
n_iter = 1000
improvement = 10**-3
train_error =[]
test_error= []
n_samples, n_features = full_train_scaled_x.shape

#plt.figure(figsize= (12,6))
for alpha in alphas_to_test:
    theta = np.random.rand(n_features,1)
    theta_,cost_history, best_betas = grad_desc(full_train_scaled_x,full_train_y,theta,n_iter,alpha,improvement,fit_intercept=True)
    train_error.append(cost_history[np.argmin(cost_history)])
    test_error.append(predict(full_test_scaled_x, full_test_y, best_betas, fit_intercept=True))
    plt.figure(figsize=(12,6))
    plt.title('Cost vs iterations at alpha level %s' % alpha)
    plt.xlabel('iteration', fontsize = 14)
    plt.ylabel('cost', fontsize = 14)
    plt.plot(cost_history, label = '%s alpha' % alpha)
    #plt.plot(cost_history)
    #plt.ylim(0,10)
    plt.xlim(-1,50)
    plt.legend()
    plt.show()

np.random.seed(20190201)
alphas_to_test = np.linspace(.05,.0005,num=2)
n_iter = 10000
improvement = 10**-3
train_error =[]
test_error= []
n_samples, n_features = full_train_scaled_x.shape

plt.figure(figsize= (12,6))
for alpha in alphas_to_test:
    theta = np.random.rand(n_features,1)
    theta_,cost_history, best_betas = grad_desc(full_train_scaled_x,full_train_y,theta,n_iter,alpha,improvement,fit_intercept=True)
    train_error.append(cost_history[np.argmin(cost_history)])
    test_error.append(predict(full_test_scaled_x, full_test_y, best_betas, fit_intercept=True))
    plt.title('Cost vs iterations at alpha level %s' % alpha)
    plt.xlabel('iteration', fontsize = 14)
    plt.ylabel('cost', fontsize = 14)
    plt.plot(cost_history, label = '%s alpha' % alpha)
    #plt.plot(cost_history)
    #plt.ylim(0,10)
    plt.xlim(-1,6000)
    plt.legend()
    plt.show()

### This part is showing how different alphas affect the train and test sets
### This part is showing how different alphas affect the train and test sets
np.random.seed(20190201)
alphas_to_test = np.linspace(.14,.00005,num=100)
n_iter = 2000
improvement = .0005
train_error_al =[]
test_error_al= []
n_samples, n_features = full_train_scaled_x.shape

for alpha in alphas_to_test:
    theta = np.random.rand(n_features,1)
    theta_,cost_history, best_betas = grad_desc(full_train_scaled_x,full_train_y,theta,n_iter,alpha,improvement,fit_intercept=True)
    
    train_error_al.append(cost_history[np.argmin(cost_history)])
    test_error_al.append(predict(full_test_scaled_x, full_test_y, best_betas, fit_intercept=True))

plt.figure(figsize = (12,6))
plt.title("Training Error vs Test Error at Different Alpha Levels \n max iteration %s" % n_iter )
plt.plot(alphas_to_test, train_error_al, label='Train')   
plt.plot(alphas_to_test, test_error_al, label='Test')
plt.hlines(y = np.min(test_error_al), xmin=np.min(alphas_to_test), xmax= np.max(alphas_to_test),label = 'min test error')
plt.ylabel('error', fontsize=14)
plt.xlabel('alphas', fontsize=14)
#plt.xlim(left = .000005,right=.000030)
plt.ylim(0,1000)
plt.legend()
plt.show()

plt.figure(figsize = (12,6))
plt.title("Zoomed in Training Error vs Test Error at Different Alpha Levels \n max iteration %s" % n_iter)
plt.plot(alphas_to_test, train_error_al, label='Train')   
plt.plot(alphas_to_test, test_error_al, label='Test')
plt.hlines(y = np.min(test_error_al), xmin=np.min(alphas_to_test), xmax= np.max(alphas_to_test),label = 'min test error')
plt.ylabel('error', fontsize=14)
plt.xlabel('alphas', fontsize=14)
plt.xlim(left = 0.0001,right=.08)
plt.ylim(400,1000)
plt.legend()
plt.show()


print("""Yes, exp1 done""")
############################################################################
#
# # Experiment 2 
#1. Experiment with various thresholds for convergence. 
#2. Plot error results for train and test sets as a function of threshold and describe how varying the threshold affects error. 
#3. Pick your best threshold and plot train and test error (in one figure) as a function of number of gradient descent iterations.
#
#############################################################################
np.random.seed(20190201)
conver_test = np.linspace(15, 2, num = 100)
lr = .05
n_iter = 2000
train_error_ex2 =[]
test_error_ex2= []
#cost_history = []

for conver in conver_test:
    theta = np.random.rand(n_features,1)
    theta_,cost_history, best_betas = grad_desc(full_train_scaled_x,full_train_y,theta,n_iter,lr,conver,fit_intercept=True)
    train_error_ex2.append(cost_history[np.argmin(cost_history)])
    test_error_ex2.append(predict(full_test_scaled_x, full_test_y, best_betas, fit_intercept=True))

plt.figure(figsize = (12,6))
plt.title("Training Error vs Test Error at Different Extremem Convergence Levels \n This plot for illustrative purposes only")
plt.plot(conver_test, train_error_ex2, label='Train')
plt.plot(conver_test, test_error_ex2, label='Test')
plt.ylabel('error', fontsize=14)
plt.xlabel('convergence threshold', fontsize=14)
plt.legend()
plt.margins(.02)
plt.show()


np.random.seed(20190201)
conver_test = np.linspace(2, 10e-7, num = 200)
lr = .05
n_iter = 2000
train_error_cov =[]
test_error_cov= []
#cost_history = []

for conver in conver_test:
    theta = np.random.rand(n_features,1)
    theta_,cost_history, best_betas = grad_desc(full_train_scaled_x,full_train_y,theta,n_iter,lr,conver,fit_intercept=True)
    train_error_cov.append(cost_history[np.argmin(cost_history)])
    test_error_cov.append(predict(full_test_scaled_x, full_test_y, best_betas, fit_intercept=True))

cov_final = .005
plt.figure(figsize = (12,6))
plt.title("Training Error vs Test Error at Different Convergence Levels")
plt.plot(conver_test, train_error_cov, label='Train')
plt.plot(conver_test, test_error_cov, label='Test')
plt.ylabel('error', fontsize=14)
plt.xlabel('convergence threshold', fontsize=14)
plt.vlines(x= cov_final, ymin=400, ymax=770,label= 'convergence selection',linestyle = '--')
plt.hlines(y = np.min(test_error_cov), xmin=np.min(conver_test), xmax= np.max(conver_test),label = 'min test error')
plt.legend()
plt.show()

plt.margins(.02)
plt.figure(figsize = (12,6))
plt.title("Training Error vs Test Error at Different Convergence Levels")
plt.plot(conver_test, train_error_cov, label='Train')
plt.plot(conver_test, test_error_cov, label='Test')
plt.ylabel('error', fontsize=14)
plt.xlabel('convergence threshold', fontsize=14)
plt.hlines(y = np.min(test_error_cov), xmin=np.min(conver_test), xmax= np.max(conver_test),label = 'min test error')
plt.vlines(x= cov_final, ymin=400, ymax=770,label= 'convergence selection', linestyle = '--')
plt.xlim(left = 0,right=1)
plt.ylim(400,800)
plt.legend()
plt.show()

print("""Yes, exp2 done""")


############################################################################
#
# # Experiment 3
#1. pick 5 features randomly and model only on these five features. 
#2. Compare train and test error results for the case of using your original set of features (greater than 10) and five random features. 
#3. Report which five features did you select randomly.
#
#############################################################################

tot_col_num = len(test_split_df.columns[:-1])
n_rand_cols = 5
#this is how random columns were picked which produced the below column numbers
#rand_cols = np.random.choice(a=tot_col_num,size=n_rand_cols)
rand_cols = np.array([29, 47, 11, 16, 18])
print(f'The random columns choosen are, {train_split_df.iloc[:,rand_cols].columns}')

trained_rand_x = full_train_scaled_x[:,rand_cols]
test_rand_x = full_test_scaled_x[:,rand_cols]

np.random.seed(20190201)
lr = .05
conver = .005
n_iter = 1000

n_samples, n_features = trained_rand_x.shape
rand_train_error = []
rand_test_error = []
theta = np.random.rand(n_features,1)

#random selection
theta_,rand_cost_history, best_betas = grad_desc(trained_rand_x,full_train_y,theta,n_iter,lr,conver,fit_intercept=True)
rand_train_error.append(rand_cost_history[np.argmin(rand_cost_history[rand_cost_history > 0])])
rand_test_error.append(predict(test_rand_x, full_test_y, best_betas, fit_intercept=True))

#give me everything
fulln_samples, fulln_features = full_train_scaled_x.shape
full_train_error = []
full_test_error = []

theta = np.random.rand(fulln_features,1)
theta_,full_cost_history, best_betas = grad_desc(full_train_scaled_x,full_train_y,theta,n_iter,lr,conver,fit_intercept=True)
full_train_error.append(full_cost_history[np.argmin(full_cost_history > 0)])
full_test_error.append(predict(full_test_scaled_x, full_test_y, best_betas, fit_intercept=True))

plt.figure(figsize = (12,6))
plt.title("Full Dataset vs Random Cost", fontsize = 16)
plt.plot(rand_cost_history, label='Random Dataset Train')
plt.plot(full_cost_history, label='Full Dataset Train')
plt.ylabel('Error', fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize = (5,6))
plt.title("Full dataset vs Random Cost", fontsize = 16)
plt.plot(rand_test_error, label='Random Dataset Test', marker = 's',markersize = 10)
plt.plot(full_test_error, label='Full Dataset Test',marker='o',markersize = 10)
plt.ylabel('Error', fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.legend()
plt.show()
print("""Yes, exp3""")

############################################################################
#
# # Experiment 4
#1. Now pick five features that you think are best suited to predict the output, and retrain your model using these five features. 
#2. Compare to the case of using your original set of features and to random features case. 
#3. Did your choice of features provide better results than picking random features? 
#4. Why? Did your choice of features provide better results than using all features? Why?
#############################################################################

lr = .05
conver = .005
n_iter = 2000

n_samples, n_features = top_ntrain_xscaled.shape
my_train_error = []
my_test_error = []
theta = np.random.rand(n_features,1)

#My selection. This is based on the top 5 variables, most are correlated
theta_,my_cost_history, best_betas = grad_desc(top_ntrain_xscaled,full_train_y,theta,n_iter,lr,conver,fit_intercept=True)
my_train_error.append(my_cost_history[np.argmin(my_cost_history[my_cost_history > 0])])
my_test_error.append(predict(top_ntest_xscaled, full_test_y, best_betas, fit_intercept=True))

plt.figure(figsize = (12,6))
plt.title("My Five Features Vs Full Vs Random", fontsize = 16)
plt.plot(my_cost_history, label='My Dataset Train')
plt.plot(rand_cost_history, label='Random Dataset Train')
plt.plot(full_cost_history, label='Full Dataset Train')
plt.ylabel('Error', fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.legend()
plt.show()

plt.figure(figsize = (5,6))
plt.title("My Five vs Full Dataset vs Random Cost", fontsize = 16)
plt.plot(my_test_error, label='My Dataset Test', marker = 's',markersize = 10)
plt.plot(rand_test_error, label='Random Dataset Test', marker = 'x',markersize = 10)
plt.plot(full_test_error, label='Full Dataset Test',marker='o',markersize = 10)
plt.ylabel('Error', fontsize=14)
plt.xlabel('Iterations', fontsize=14)
plt.legend()
plt.show()

print("""Thank you for running my code. I hope you have a great day!""")