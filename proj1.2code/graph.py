__author__ = 'Swetha'

import matplotlib.pyplot as plt


# line 2 points
#x2 = [0.05,0.05,0.06,0.07,0.08,0.08,0.08, 0.09]
#x3 = [50,100,150,150,200,250,300,325,350]
#y2 = [53,90,90,96,96,96,99,98]
#y3 = [53,92,90,96,95,96,99,98,96]
# plotting the line 2 points

#x2 = [125,125,125,125,125,150,200,250,300]
#y2=[]
x3 = [2000,3000, 3250, 3250,3500,3500,4000,5000]
y3 =[89,98,95,98,95,100,96,95]
plt.plot(x3, y3 , label = "Accuracy", color = "red")

# naming the x axis
plt.xlabel('Number of Epochs')
# naming the y axis
plt.ylabel('Accuracy ')
# giving a title to my graph
plt.title("change in accuracy wrt to number of epochs")

plt.legend()
plt.show()
