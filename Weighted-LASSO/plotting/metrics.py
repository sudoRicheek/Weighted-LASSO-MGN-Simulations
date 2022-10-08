import json
import os
from unicodedata import decimal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt



f1 = open('../out/simulation1.json')
f2 = open('../out/simulation2.json')
f3 = open('../out/simulation3.json')

data1 = json.load(f1)
data2 = json.load(f2)
data3 = json.load(f3)


measurements = ["375", "300", "150"]
sparsity =np.round(data1["P-LASSO"]["Sparsity"], decimals=2).astype('str').tolist() 

####################### Plain lasso RRMSE #############################
rrmse = np.round([data1["P-LASSO"]["RRMSE"], data2["P-LASSO"]["RRMSE"], data3["P-LASSO"]["RRMSE"]], decimals=5)


fig, ax = plt.subplots()
im = ax.imshow(rrmse)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(sparsity)), labels=sparsity)
ax.set_yticks(np.arange(len(measurements)), labels=measurements)

# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(measurements)):
    for j in range(len(sparsity)):
        text = ax.text(j, i, rrmse[i, j],
                       ha="center", va="center", color="w")

ax.set_title("RRMSE- Plain LASSO") 
fig.tight_layout()



##################################################################################

####################### Weighted lasso RRMSE #############################
rrmse = np.round([data1["W-LASSO"]["RRMSE"], data2["W-LASSO"]["RRMSE"], data3["W-LASSO"]["RRMSE"]], decimals=5)


fig, ax = plt.subplots()
im = ax.imshow(rrmse)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(sparsity)), labels=sparsity)
ax.set_yticks(np.arange(len(measurements)), labels=measurements)

# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(measurements)):
    for j in range(len(sparsity)):
        text = ax.text(j, i, rrmse[i, j],
                       ha="center", va="center", color="w")

ax.set_title("RRMSE- Weighted LASSO") 
fig.tight_layout()




##################################################################################
tp1 = np.array(data1["P-LASSO"]["TP"], dtype=np.int64)
tn1 = np.array(data1["P-LASSO"]["TN"], dtype=np.int64)
fp1 = np.array(data1["P-LASSO"]["FP"], dtype=np.int64)
fn1 = np.array(data1["P-LASSO"]["FN"], dtype=np.int64)

mcc1 = (tp1*tn1 - fp1*fn1)/np.sqrt((tp1+fp1)*(tp1+fn1)*(tn1+fp1)*(tn1+fn1))

tp2 = np.array(data2["P-LASSO"]["TP"], dtype=np.int64)
tn2 = np.array(data2["P-LASSO"]["TN"], dtype=np.int64)
fp2 = np.array(data2["P-LASSO"]["FP"], dtype=np.int64)
fn2 = np.array(data2["P-LASSO"]["FN"], dtype=np.int64)

mcc2 = (tp2*tn2 - fp2*fn2)/np.sqrt((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2))

tp3 = np.array(data3["P-LASSO"]["TP"], dtype=np.int64)
tn3 = np.array(data3["P-LASSO"]["TN"], dtype=np.int64)
fp3 = np.array(data3["P-LASSO"]["FP"], dtype=np.int64)
fn3 = np.array(data3["P-LASSO"]["FN"], dtype=np.int64)

mcc3 = (tp3*tn3 - fp3*fn3)/np.sqrt((tp3+fp3)*(tp3+fn3)*(tn3+fp3)*(tn3+fn3))

mcc = np.round([mcc1, mcc2, mcc3], decimals=5)


fig, ax = plt.subplots()
im = ax.imshow(mcc)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(sparsity)), labels=sparsity)
ax.set_yticks(np.arange(len(measurements)), labels=measurements)

# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(measurements)):
    for j in range(len(sparsity)):
        text = ax.text(j, i, mcc[i, j],
                       ha="center", va="center", color="w")

ax.set_title("MCC- Plain LASSO") 
fig.tight_layout()



##################################################################################

##################################################################################
tp1 = np.array(data1["W-LASSO"]["TP"], dtype=np.int64)
tn1 = np.array(data1["W-LASSO"]["TN"], dtype=np.int64)
fp1 = np.array(data1["W-LASSO"]["FP"], dtype=np.int64)
fn1 = np.array(data1["W-LASSO"]["FN"], dtype=np.int64)

mcc1 = (tp1*tn1 - fp1*fn1)/np.sqrt((tp1+fp1)*(tp1+fn1)*(tn1+fp1)*(tn1+fn1))

tp2 = np.array(data2["W-LASSO"]["TP"], dtype=np.int64)
tn2 = np.array(data2["W-LASSO"]["TN"], dtype=np.int64)
fp2 = np.array(data2["W-LASSO"]["FP"], dtype=np.int64)
fn2 = np.array(data2["W-LASSO"]["FN"], dtype=np.int64)

mcc2 = (tp2*tn2 - fp2*fn2)/np.sqrt((tp2+fp2)*(tp2+fn2)*(tn2+fp2)*(tn2+fn2))

tp3 = np.array(data3["W-LASSO"]["TP"], dtype=np.int64)
tn3 = np.array(data3["W-LASSO"]["TN"], dtype=np.int64)
fp3 = np.array(data3["W-LASSO"]["FP"], dtype=np.int64)
fn3 = np.array(data3["W-LASSO"]["FN"], dtype=np.int64)

mcc3 = (tp3*tn3 - fp3*fn3)/np.sqrt((tp3+fp3)*(tp3+fn3)*(tn3+fp3)*(tn3+fn3))

mcc = np.round([mcc1, mcc2, mcc3], decimals=5)


fig, ax = plt.subplots()
im = ax.imshow(mcc)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(sparsity)), labels=sparsity)
ax.set_yticks(np.arange(len(measurements)), labels=measurements)

# Rotate the tick labels and set their alignment.
# plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#          rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(measurements)):
    for j in range(len(sparsity)):
        text = ax.text(j, i, mcc[i, j],
                       ha="center", va="center", color="w")

ax.set_title("MCC- Weighted LASSO") 
fig.tight_layout()



##################################################################################

plt.show() ##### Show all plots

print((tn1+fn1))