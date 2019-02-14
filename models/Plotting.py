import numpy as np

#import matplot library for plotting graphs
import matplotlib.pyplot as plt

#initialise the indices variable
inds = []

#This is the definition for plotting the line graph of Cross validation scores for Random Forest
def line_graph(score,name,xa,xb,xc):
    #Plot a line graph
    grid = np.arange(xa,xb,xc)
    for C in grid:
        inds.append(C)
    labels =[" Score "]    
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.plot(inds,score,'or-', linewidth=1) #Plot the ridge score in red with circle marker

    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Accuracy") #Y-axis label
    plt.xlabel(" Number of neighbors  "+ name) #X-axis label
    plt.title("Accuracy Score vs k value for kNN(approach2)") #Plot title
    plt.xlim(xa,xb) #set x axis range
    #plt.ylim(0.1,0.5) #Set yaxis range
    plt.legend(labels,loc="best")

    #Make sure labels and titles are inside plot area
    plt.tight_layout()

    #Save the chart
    plt.savefig("./Figures/" + name + "_line_graph.pdf")

    #Displays the plots. 
    #You must close the plot window for the code following each show()
    #to continue to run
    plt.show()