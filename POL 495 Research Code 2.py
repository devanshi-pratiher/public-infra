#!/usr/bin/env python
# coding: utf-8

# In[123]:


import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.metrics import precision_recall_fscore_support


from sklearn.model_selection import KFold
import numpy as np
import sys
import math
# Implementing a set of strategies of CP as a Python function that can be called by PHP:
# 
# a)	Adaptive <br>
# Part 1: Provider and Users/farmers (Tit for tat to both other users and the provider)
# 

# # Tit For Tat for Provider

# In[124]:


df = pd.read_csv("POL495.csv")
df


# In[125]:


provider= pd.get_dummies(df.Provider)


# In[126]:


df = df.join(provider)


# In[127]:


mid = df['Outcome']
df.drop(labels=['Outcome'], axis=1, inplace = True)


# In[128]:


df.insert(7, 'Outcome', mid)
df


# The farmer then observes outcomes of Round T
# <br>If threshold payment is 0:
# <br>-not enough farmers chose pay so threshold payment = 0. 
# <br>-even if the provider chose invest, infra. State will be 0
# 
# If threshold payment = 1:
# <br>-enough farmers chose to pay so threshold payment = 1. 
# <br>-If the provider also chose to be cooperative (invest), then infra. State will be 1
# 
# If threshold payment is 1:
# <br>-enough farmers chose to pay and threshold payment = 1. 
# <br>-If the provider also chose to be non-cooperative (keep), then infra. State will be 0

# In[129]:


farmer_pay = df['Threshold Payment']
provider_invest = df['invest']
provider_keep = df['keep']
group = df['Group #']
rounds = df['Rounds']

group_len = 24
round_len = 20

# Provider Nash Equilibrium <br>
# No matter the threshold value (farmer payment), the provider always chooses to keep and not invest. 

# In[246]:


#computer is the provider
def nashEquilibrium():
	r = 24*20
	for T in range(r):
		provider_keep.iat[T]= 1
#nashEquilibrium()


# Provider: Altruistic <br>
# No matter what the threshold value (farmer payment), provider always chooses to invest.

# In[243]:


def altruisitc():
	r = 24*20
	for T in range(r):
		provider_invest.iat[T]= 1
    
#altruisitc()


# Provider: Tit for Tat <br>
# Proivder observes outcomes in round T. <br>
# If threshold payment = 0: <br>
# - Provider will also keep money
# - Either way (even if provider invests), outcome is 0
# 
# If threshold payment is 1:
# - Provider also chose to be cooperative (invest)
# - Outcome is 1
# 

# In[247]:


def ttProvider():
    result1 = []
    decision = 0
    for pay in farmer_pay:
        if pay == 1:
            result1.append(1)
        elif pay == 0:
            result1.append(0)
        #arr = [[0]*pay]*decision
    return result1
#ttProvider()


# Involving next rounds:
# - If provider chose invest in Round T and infra. State was 0 in Round T, Provider responds by NOT investing in Round T+1
# - If the provider chose invest in Round T and infra. State was 1 in Round T, Provider responds by investing in Round T+1

# In[248]:


#choose which group to iterate the rounds in
grouped = df.groupby(group)
df_new = grouped.get_group(2)



# In[228]:


rounds_grouped = df_new['Rounds']
invest = df_new['invest']
outcome = df_new['Outcome']
keep = df_new['keep']



#show decisions of farmer in round T based on infra. State

def farmer_decison(T):
	infra_state = 0
	if (farmer_pay.iat[T] == 1 ):
		infra_state =1
	elif (farmer_pay.iat[T] == 0):
		infra_state = 0

	return infra_state

#show decisions of provider in round T based on infra. State

def provider_decison(T):
	infra_state =0
	if (farmer_pay.iat[T] == 0):
		infra_state = 0
	elif (farmer_pay.iat[T] == 1):
		infra_state = 1
	
	return infra_state

farmer_decision={}
provider_decision={}
infra_state=[]
farmer_decision_list={}
provider_decision_list={}
infra_state_list={}

def ttDecision(group):
	r = 24*20
	count = 0
	#infra_state[0]=0
	prev_count=0
	groupc=0;
	tcount=0
	
	for T in range(r):
		if (group != groupc):
			if count >= 20:
				count = 0
				groupc +=1
		count += 1
		prev_count = count - 1
		if (group == groupc and tcount <=20):
			f=farmer_decison(T)
			#print("farmer :",f)
			farmer_decision[T]=f #Farmer
			f=provider_decison(T)
			provider_decision[T]=f #Provider
			#print("provider_decision :",f)
			#Chech this
			
			if (provider_decision[T]==1):
				infra_state.append(1)
			elif (provider_decision[T]==0 and farmer_decision[T]==1):
				infra_state.append(0)
			elif (provider_decision[T]==0 and farmer_decision[T]==0):
				infra_state.append(0)
				
			tcount += 1
# Involving next rounds:
# - If provider chose invest in Round T and infra. State was 0 in Round T, Provider responds by NOT investing in Round T+1
# - If the provider chose invest in Round T and infra. State was 1 in Round T, Provider responds by investing in Round T+1

# - If farmer chose to pay in Round T and infra. state is 0, farmer hold (does not pay) in Round T+1
# - If farmer chose to pay in Round T and infra. state is 1, farmer pays in Round T+1

       
def ttDecisionT1(group, r1):

	T = r1-1#T
	T1 = r1 #T + 1
	pinvest=0
	fdecision=0
    #Provider Logic for Rounds

	if (provider_invest[T] == 1 and infra_state[T] == 0):
		provider_decision[T1] == 0
		pinvest=0
	elif (provider_invest[T] == 1 and infra_state[T] == 1):
		pinvest=1
		provider_decision[T1] == 1
		
	#Farmer	Logic for Rounds
	if (farmer_pay[T] == 1 and infra_state[T] == 0):
		fdecision=0
		farmer_decision[T1] == 0
	elif (farmer_pay[T] == 1 and infra_state[T] == 1):
		fdecision=1
		farmer_decision[T1] == 0

	if (pinvest == 0):
		print ("Provider Keep")
	else:
		print("Provider Invest")
	if (fdecision== 0):
		print("Farmers Hold")
	else:
		print("Farmer Pay")
		
	print("Infra State (T):", infra_state[T])
		
def ttAll():
	for group in range(0,24):
		ttDecision(group)
		for r in range(1,20):
			print ("---Group:",group,"------ Round (T+1)",r,"----------")
			ttDecisionT1(group, r)	#T+1
			
#def square(x: ndarray) -> ndarray:
#    '''
#    Square each element in the input ndarray.
#    '''
#    return np.power(x, 2)
    			
import math
 
def minimax (curDepth, nodeIndex,
             maxTurn, scores,
             targetDepth):
 
    # base case : targetDepth reached
    if (curDepth == targetDepth):
        return scores[nodeIndex]
     
    if (maxTurn):
        return max(minimax(curDepth + 1, nodeIndex * 2,
                    False, scores, targetDepth),
                   minimax(curDepth + 1, nodeIndex * 2 + 1,
                    False, scores, targetDepth))
     
    else:
        return min(minimax(curDepth + 1, nodeIndex * 2,
                     True, scores, targetDepth),
                   minimax(curDepth + 1, nodeIndex * 2 + 1,
                     True, scores, targetDepth))
     

#Main function: involves rounds and altruistic, tit-for-tat, and nash equilibrium
 	
def main():
	args = sys.argv[1:]
	print(args[0])
	if (int(args[0]) == 3):
		print ("The provider (always invest, i.e., altruistic)")
		altruisitc()
		ttAll();
	elif (int(args[0]) == 2):
		print ("The provider (Nash Eq.)")
		nashEquilibrium()
		ttAll();
	elif(int(args[0]) == 8):
		print ("Tit For Tat")
		ttAll();
	else:
		#minimax code
		# Driver code
		scores = [3, 5, 2, 9, 12, 5, 23, 23]
		for group in range(0,24):
			ttDecision(group)
		print(infra_state)
		treeDepth = math.log(len(scores), 2)
 
		print("The optimal value is : ", end, "")
		print(minimax(0, 0, True, scores, treeDepth))
		

		
	
if __name__ == '__main__':
    sys.exit(main())  # next section explains the use of sys.exit



